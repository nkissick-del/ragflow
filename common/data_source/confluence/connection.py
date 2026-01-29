import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, cast
from atlassian import Confluence
import requests
from requests.exceptions import HTTPError

from common.data_source.utils import confluence_refresh_tokens, _handle_http_error, run_with_timeout
from common.data_source.config import OAUTH_CONFLUENCE_CLOUD_CLIENT_ID, OAUTH_CONFLUENCE_CLOUD_CLIENT_SECRET
from common.data_source.interfaces import CredentialsProviderInterface


class ConfluenceConnectionMixin:
    """
    Mixin for Confluence connection and credential management.
    Requires the host class to have:
    - self.base_url
    - self._credentials_provider
    - self.scoped_token
    - self.redis_client
    - self.static_credentials
    - self.credential_key
    - self._is_cloud
    - self._url
    - self.shared_base_kwargs
    - self._confluence
    - self._kwargs
    """

    CREDENTIAL_TTL = 300  # 5 min
    PROBE_TIMEOUT = 5  # 5 seconds

    def _renew_credentials(self) -> tuple[dict[str, Any], bool]:
        """credential_json - the current json credentials
        Returns a tuple
        1. The up-to-date credentials
        2. True if the credentials were updated

        This method is intended to be used within a distributed lock.
        Lock, call this, update credentials if the tokens were refreshed, then release
        """
        # static credentials are preloaded, so no locking/redis required
        if self.static_credentials:
            return self.static_credentials, False

        if not self.redis_client:
            raise RuntimeError("self.redis_client is None")

        # dynamic credentials need locking
        # check redis first, then fallback to the DB
        credential_raw = self.redis_client.get(self.credential_key)
        if credential_raw is not None:
            credential_bytes = cast(bytes, credential_raw)
            credential_str = credential_bytes.decode("utf-8")
            credential_json: dict[str, Any] = json.loads(credential_str)
        else:
            credential_json = self._credentials_provider.get_credentials()

        if "confluence_refresh_token" not in credential_json:
            # static credentials ... cache them permanently and return
            self.static_credentials = credential_json
            return credential_json, False

        if not OAUTH_CONFLUENCE_CLOUD_CLIENT_ID:
            raise RuntimeError("OAUTH_CONFLUENCE_CLOUD_CLIENT_ID must be set!")

        if not OAUTH_CONFLUENCE_CLOUD_CLIENT_SECRET:
            raise RuntimeError("OAUTH_CONFLUENCE_CLOUD_CLIENT_SECRET must be set!")

        # check if we should refresh tokens. we're deciding to refresh halfway
        # to expiration
        now = datetime.now(timezone.utc)
        created_at = datetime.fromisoformat(credential_json["created_at"])
        expires_in: int = credential_json["expires_in"]
        renew_at = created_at + timedelta(seconds=expires_in // 2)
        if now <= renew_at:
            # cached/current credentials are reasonably up to date
            return credential_json, False

        # we need to refresh
        logging.info("Renewing Confluence Cloud credentials...")
        new_credentials = confluence_refresh_tokens(
            OAUTH_CONFLUENCE_CLOUD_CLIENT_ID,
            OAUTH_CONFLUENCE_CLOUD_CLIENT_SECRET,
            credential_json["cloud_id"],
            credential_json["confluence_refresh_token"],
        )

        # store the new credentials to redis and to the db through the provider
        # redis: we use a 5 min TTL because we are given a 10 minutes grace period
        # when keys are rotated. it's easier to expire the cached credentials
        # reasonably frequently rather than trying to handle strong synchronization
        # between the db and redis everywhere the credentials might be updated
        new_credential_str = json.dumps(new_credentials)
        self.redis_client.set(self.credential_key, new_credential_str, ex=self.CREDENTIAL_TTL)
        self._credentials_provider.set_credentials(new_credentials)

        return new_credentials, True

    @staticmethod
    def _make_oauth2_dict(credentials: dict[str, Any]) -> dict[str, Any]:
        oauth2_dict: dict[str, Any] = {}
        if "confluence_refresh_token" in credentials:
            oauth2_dict["client_id"] = OAUTH_CONFLUENCE_CLOUD_CLIENT_ID
            oauth2_dict["token"] = {}
            oauth2_dict["token"]["access_token"] = credentials["confluence_access_token"]
        return oauth2_dict

    def _probe_connection(
        self,
        **kwargs: Any,
    ) -> None:
        merged_kwargs = {**self.shared_base_kwargs, **kwargs}
        # add special timeout to make sure that we don't hang indefinitely
        merged_kwargs["timeout"] = self.PROBE_TIMEOUT

        with self._credentials_provider:
            credentials, _ = self._renew_credentials()
            if self.scoped_token:
                # v2 endpoint doesn't always work with scoped tokens, use v1
                token = credentials["confluence_access_token"]
                probe_url = f"{self.base_url}/rest/api/space?limit=1"

                logging.info("Sending probe request to Confluence...")

                try:
                    r = requests.get(
                        probe_url,
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=self.PROBE_TIMEOUT,
                    )
                    r.raise_for_status()
                except HTTPError as e:
                    if e.response.status_code == 403:
                        logging.warning("scoped token authenticated but not valid for probe endpoint (spaces)")
                    else:
                        if "WWW-Authenticate" in e.response.headers:
                            logging.warning(f"WWW-Authenticate: {e.response.headers['WWW-Authenticate']}")
                            logging.warning(f"Full error: {e.response.text}")
                        raise e
                return

            # probe connection with direct client, no retries
            if "confluence_refresh_token" in credentials:
                logging.info("Probing Confluence with OAuth Access Token.")

                oauth2_dict: dict[str, Any] = self._make_oauth2_dict(credentials)
                url = f"https://api.atlassian.com/ex/confluence/{credentials['cloud_id']}"
                confluence_client_with_minimal_retries = Confluence(url=url, oauth2=oauth2_dict, **merged_kwargs)
            else:
                logging.info("Probing Confluence with Personal Access Token.")
                url = self._url
                if self._is_cloud:
                    logging.info("running with cloud client")
                    confluence_client_with_minimal_retries = Confluence(
                        url=url,
                        username=credentials["confluence_username"],
                        password=credentials["confluence_access_token"],
                        **merged_kwargs,
                    )
                else:
                    confluence_client_with_minimal_retries = Confluence(
                        url=url,
                        token=credentials["confluence_access_token"],
                        **merged_kwargs,
                    )

            # This call sometimes hangs indefinitely, so we run it in a timeout
            spaces = run_with_timeout(
                timeout=self.PROBE_TIMEOUT,
                func=confluence_client_with_minimal_retries.get_all_spaces,
                limit=1,
            )

            if not spaces:
                raise RuntimeError(f"No spaces found at {url}! Check your credentials and wiki_base and make sure is_cloud is set correctly.")

            logging.info("Confluence probe succeeded.")

    def _initialize_connection(
        self,
        **kwargs: Any,
    ) -> None:
        """Called externally to init the connection in a thread safe manner."""
        merged_kwargs = {**self.shared_base_kwargs, **kwargs}
        with self._credentials_provider:
            credentials, _ = self._renew_credentials()
            self._confluence = self._initialize_connection_helper(credentials, **merged_kwargs)
            self._kwargs = merged_kwargs

    def _initialize_connection_helper(
        self,
        credentials: dict[str, Any],
        **kwargs: Any,
    ) -> Confluence:
        """Called internally to init the connection. Distributed locking
        to prevent multiple threads from modifying the credentials
        must be handled around this function."""

        confluence = None

        # initialize direct client without retry logic
        if "confluence_refresh_token" in credentials:
            logging.info("Connecting to Confluence Cloud with OAuth Access Token.")

            oauth2_dict: dict[str, Any] = self._make_oauth2_dict(credentials)
            url = f"https://api.atlassian.com/ex/confluence/{credentials['cloud_id']}"
            confluence = Confluence(url=url, oauth2=oauth2_dict, **kwargs)
        else:
            logging.info("Connecting to Confluence with Personal Access Token.")
            if self._is_cloud:
                confluence = Confluence(
                    url=self._url,
                    username=credentials["confluence_username"],
                    password=credentials["confluence_access_token"],
                    **kwargs,
                )
            else:
                confluence = Confluence(
                    url=self._url,
                    token=credentials["confluence_access_token"],
                    **kwargs,
                )

        return confluence

    # https://developer.atlassian.com/cloud/confluence/rate-limiting/
    # This uses the native rate limiting option provided by the
    # confluence client and otherwise applies a simpler set of error handling.
    def _make_rate_limited_confluence_method(self, name: str, credential_provider: CredentialsProviderInterface | None) -> Callable[..., Any]:
        def wrapped_call(*args: Any, **kwargs: Any) -> Any:
            MAX_RETRIES = 5

            TIMEOUT = 600
            timeout_at = time.monotonic() + TIMEOUT

            last_http_exc = None
            for attempt in range(MAX_RETRIES):
                if time.monotonic() > timeout_at:
                    raise TimeoutError(f"Confluence call attempts took longer than {TIMEOUT} seconds.")

                # we're relying more on the client to rate limit itself
                # and applying our own retries in a more specific set of circumstances
                try:
                    if credential_provider:
                        with credential_provider:
                            credentials, renewed = self._renew_credentials()
                            if renewed:
                                self._confluence = self._initialize_connection_helper(credentials, **self._kwargs)
                            attr = getattr(self._confluence, name, None)
                            if attr is None:
                                # The underlying Confluence client doesn't have this attribute
                                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

                            return attr(*args, **kwargs)
                    else:
                        attr = getattr(self._confluence, name, None)
                        if attr is None:
                            # The underlying Confluence client doesn't have this attribute
                            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

                        return attr(*args, **kwargs)

                except HTTPError as e:
                    last_http_exc = e
                    delay_until = _handle_http_error(e, attempt)
                    logging.warning(f"HTTPError in confluence call. Retrying in {delay_until} seconds...")
                    while time.monotonic() < delay_until:
                        # in the future, check a signal here to exit
                        time.sleep(1)

                except AttributeError as e:
                    # Some error within the Confluence library, unclear why it fails.
                    # Users reported it to be intermittent, so just retry
                    if attempt == MAX_RETRIES - 1:
                        raise e

                    logging.exception("Confluence Client raised an AttributeError. Retrying...")
                    time.sleep(5)

            if last_http_exc:
                raise last_http_exc

        return wrapped_call

    def __getattr__(self, name: str) -> Any:
        """Dynamically intercept attribute/method access."""
        try:
            confluence = object.__getattribute__(self, "_confluence")
        except AttributeError:
            confluence = None

        if not confluence:
            raise RuntimeError("Confluence connection not initialized. Call _initialize_connection first.")

        attr = getattr(confluence, name, None)
        if attr is None:
            # The underlying Confluence client doesn't have this attribute
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # If it's not a method, just return it after ensuring token validity
        if not callable(attr):
            return attr

        # skip methods that start with "_"
        if name.startswith("_"):
            return attr

        # wrap the method with our retry handler
        rate_limited_method: Callable[..., Any] = self._make_rate_limited_confluence_method(name, self._credentials_provider)

        return rate_limited_method
