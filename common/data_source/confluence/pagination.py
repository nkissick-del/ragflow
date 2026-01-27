import logging
from typing import Any, Callable, Generator, Iterator, cast
from urllib.parse import quote

from common.data_source.config import _DEFAULT_PAGINATION_LIMIT, _PROBLEMATIC_EXPANSIONS, _REPLACEMENT_EXPANSIONS
from common.data_source.utils import update_param_in_path, get_start_param_from_url
from common.data_source.interfaces import ConfluenceUser


class ConfluencePaginationMixin:
    """
    Mixin for Confluence pagination logic.
    Requires the host class to have:
    - self._is_cloud
    - self.get(path, advanced_mode, params) -> Response
    - self._confluence_user_profiles_override
    """

    def _try_one_by_one_for_paginated_url(
        self,
        url_suffix: str,
        initial_start: int,
        limit: int,
    ) -> Generator[dict[str, Any], None, str | None]:
        """
        Go through `limit` items, starting at `initial_start` one by one (e.g. using
        `limit=1` for each call).

        If we encounter an error, we skip the item and try the next one. We will return
        the items we were able to retrieve successfully.

        Returns the expected next url_suffix. Returns None if it thinks we've hit the end.

        TODO (chris): make this yield failures as well as successes.
        TODO (chris): make this work for confluence cloud somehow.
        """
        if self._is_cloud:
            raise RuntimeError("This method is not implemented for Confluence Cloud.")

        found_empty_page = False
        temp_url_suffix = url_suffix

        for ind in range(limit):
            try:
                temp_url_suffix = update_param_in_path(url_suffix, "start", str(initial_start + ind))
                temp_url_suffix = update_param_in_path(temp_url_suffix, "limit", "1")
                logging.info(f"Making recovery confluence call to {temp_url_suffix}")
                raw_response = self.get(path=temp_url_suffix, advanced_mode=True)
                raw_response.raise_for_status()

                latest_results = raw_response.json().get("results", [])
                yield from latest_results

                if not latest_results:
                    # no more results, break out of the loop
                    logging.info(f"No results found for call '{temp_url_suffix}'Stopping pagination.")
                    found_empty_page = True
                    break
            except Exception:
                logging.exception(f"Error in confluence call to {temp_url_suffix}. Continuing.")

        if found_empty_page:
            return None

        # if we got here, we successfully tried `limit` items
        return update_param_in_path(url_suffix, "start", str(initial_start + limit))

    def _paginate_url(
        self,
        url_suffix: str,
        limit: int | None = None,
        # Called with the next url to use to get the next page
        next_page_callback: Callable[[str], None] | None = None,
        force_offset_pagination: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """
        This will paginate through the top level query.
        """
        if not limit:
            limit = _DEFAULT_PAGINATION_LIMIT

        url_suffix = update_param_in_path(url_suffix, "limit", str(limit))

        while url_suffix:
            logging.debug(f"Making confluence call to {url_suffix}")
            try:
                raw_response = self.get(
                    path=url_suffix,
                    advanced_mode=True,
                    params={
                        "body-format": "atlas_doc_format",
                        "expand": "body.atlas_doc_format",
                    },
                )
            except Exception as e:
                logging.exception(f"Error in confluence call to {url_suffix}")
                raise e

            try:
                raw_response.raise_for_status()
            except Exception as e:
                logging.warning(f"Error in confluence call to {url_suffix}")

                # If the problematic expansion is in the url, replace it
                # with the replacement expansion and try again
                # If that fails, raise the error
                if _PROBLEMATIC_EXPANSIONS in url_suffix:
                    logging.warning(f"Replacing {_PROBLEMATIC_EXPANSIONS} with {_REPLACEMENT_EXPANSIONS} and trying again.")
                    url_suffix = url_suffix.replace(
                        _PROBLEMATIC_EXPANSIONS,
                        _REPLACEMENT_EXPANSIONS,
                    )
                    continue

                # If we fail due to a 500, try one by one.
                # NOTE: this iterative approach only works for server, since cloud uses cursor-based
                # pagination
                if raw_response.status_code == 500 and not self._is_cloud:
                    initial_start = get_start_param_from_url(url_suffix)
                    if initial_start is None:
                        # can't handle this if we don't have offset-based pagination
                        raise

                    # this will just yield the successful items from the batch
                    new_url_suffix = yield from self._try_one_by_one_for_paginated_url(
                        url_suffix,
                        initial_start=initial_start,
                        limit=limit,
                    )

                    # this means we ran into an empty page
                    if new_url_suffix is None:
                        if next_page_callback:
                            next_page_callback("")
                        break

                    url_suffix = new_url_suffix
                    continue

                else:
                    logging.exception(f"Error in confluence call to {url_suffix} \nRaw Response Text: {raw_response.text} \nFull Response: {raw_response.__dict__} \nError: {e} \n")
                    raise

            try:
                next_response = raw_response.json()
            except Exception as e:
                logging.exception(f"Failed to parse response as JSON. Response: {raw_response.__dict__}")
                raise e

            # Yield the results individually.
            results = cast(list[dict[str, Any]], next_response.get("results", []))

            # Note 1:
            # Make sure we don't update the start by more than the amount
            # of results we were able to retrieve. The Confluence API has a
            # weird behavior where if you pass in a limit that is too large for
            # the configured server, it will artificially limit the amount of
            # results returned BUT will not apply this to the start parameter.
            # This will cause us to miss results.
            #
            # Note 2:
            # We specifically perform manual yielding (i.e., `for x in xs: yield x`) as opposed to using a `yield from xs`
            # because we *have to call the `next_page_callback`* prior to yielding the last element!
            #
            # If we did:
            #
            # ```py
            # yield from results
            # if next_page_callback:
            #   next_page_callback(url_suffix)
            # ```
            #
            # then the logic would fail since the iterator would finish (and the calling scope would exit out of its driving
            # loop) prior to the callback being called.

            old_url_suffix = url_suffix
            updated_start = get_start_param_from_url(old_url_suffix)
            url_suffix = cast(str, next_response.get("_links", {}).get("next", ""))
            for i, result in enumerate(results):
                updated_start += 1
                if url_suffix and next_page_callback and i == len(results) - 1:
                    # update the url if we're on the last result in the page
                    if not self._is_cloud:
                        # If confluence claims there are more results, we update the start param
                        # based on how many results were returned and try again.
                        url_suffix = update_param_in_path(url_suffix, "start", str(updated_start))
                    # notify the caller of the new url
                    next_page_callback(url_suffix)

                elif force_offset_pagination and i == len(results) - 1:
                    url_suffix = update_param_in_path(old_url_suffix, "start", str(updated_start))

                yield result

            # we've observed that Confluence sometimes returns a next link despite giving
            # 0 results. This is a bug with Confluence, so we need to check for it and
            # stop paginating.
            if url_suffix and not results:
                logging.info(f"No results found for call '{old_url_suffix}' despite next link being present. Stopping pagination.")
                break

    def build_cql_url(self, cql: str, expand: str | None = None) -> str:
        expand_string = f"&expand={expand}" if expand else ""
        return f"rest/api/content/search?cql={cql}{expand_string}"

    def paginated_cql_retrieval(
        self,
        cql: str,
        expand: str | None = None,
        limit: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        The content/search endpoint can be used to fetch pages, attachments, and comments.
        """
        cql_url = self.build_cql_url(cql, expand)
        yield from self._paginate_url(cql_url, limit)

    def paginated_page_retrieval(
        self,
        cql_url: str,
        limit: int,
        # Called with the next url to use to get the next page
        next_page_callback: Callable[[str], None] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        Error handling (and testing) wrapper for _paginate_url,
        because the current approach to page retrieval involves handling the
        next page links manually.
        """
        try:
            yield from self._paginate_url(cql_url, limit=limit, next_page_callback=next_page_callback)
        except Exception as e:
            logging.exception(f"Error in paginated_page_retrieval: {e}")
            raise e

    def cql_paginate_all_expansions(
        self,
        cql: str,
        expand: str | None = None,
        limit: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        This function will paginate through the top level query first, then
        paginate through all the expansions.
        """

        def _traverse_and_update(data: dict | list) -> None:
            if isinstance(data, dict):
                next_url = data.get("_links", {}).get("next")
                if next_url and "results" in data:
                    data["results"].extend(self._paginate_url(next_url, limit=limit))

                for value in data.values():
                    _traverse_and_update(value)
            elif isinstance(data, list):
                for item in data:
                    _traverse_and_update(item)

        for confluence_object in self.paginated_cql_retrieval(cql, expand, limit):
            _traverse_and_update(confluence_object)
            yield confluence_object

    def paginated_cql_user_retrieval(
        self,
        expand: str | None = None,
        limit: int | None = None,
    ) -> Iterator[ConfluenceUser]:
        """
        The search/user endpoint can be used to fetch users.
        It's a separate endpoint from the content/search endpoint used only for users.
        It's very similar to the content/search endpoint.
        """

        # this is needed since there is a live bug with Confluence Server/Data Center
        # where not all users are returned by the APIs. This is a workaround needed until
        # that is patched.
        if self._confluence_user_profiles_override:
            yield from self._confluence_user_profiles_override

        elif self._is_cloud:
            cql = "type=user"
            url = "rest/api/search/user"
            expand_string = f"&expand={expand}" if expand else ""
            url += f"?cql={cql}{expand_string}"
            for user_result in self._paginate_url(url, limit, force_offset_pagination=True):
                user = user_result["user"]
                yield ConfluenceUser(
                    user_id=user["accountId"],
                    username=None,
                    display_name=user["displayName"],
                    email=user.get("email"),
                    type=user["accountType"],
                )
        else:
            for user in self._paginate_url("rest/api/user/list", limit):
                yield ConfluenceUser(
                    user_id=user["userKey"],
                    username=user["username"],
                    display_name=user["displayName"],
                    email=None,
                    type=user.get("type", "user"),
                )

    def paginated_groups_by_user_retrieval(
        self,
        user_id: str,  # accountId in Cloud, userKey in Server
        limit: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        This is not an SQL like query.
        It's a confluence specific endpoint that can be used to fetch groups.
        """
        user_field = "accountId" if self._is_cloud else "key"
        user_value = user_id
        # Server uses userKey (but calls it key during the API call), Cloud uses accountId
        user_query = f"{user_field}={quote(user_value)}"

        url = f"rest/api/user/memberof?{user_query}"
        yield from self._paginate_url(url, limit, force_offset_pagination=True)

    def paginated_groups_retrieval(
        self,
        limit: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        This is not an SQL like query.
        It's a confluence specific endpoint that can be used to fetch groups.
        """
        yield from self._paginate_url("rest/api/group", limit)

    def paginated_group_members_retrieval(
        self,
        group_name: str,
        limit: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        This is not an SQL like query.
        It's a confluence specific endpoint that can be used to fetch the members of a group.
        THIS DOESN'T WORK FOR SERVER because it breaks when there is a slash in the group name.
        E.g. neither "test/group" nor "test%2Fgroup" works for confluence.
        """
        group_name = quote(group_name)
        yield from self._paginate_url(f"rest/api/group/{group_name}/member", limit)
