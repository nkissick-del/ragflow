"""Exception class definitions"""


class ConnectorMissingCredentialError(Exception):
    """Missing credentials exception"""

    def __init__(self, connector_name: str):
        super().__init__(f"Missing credentials for {connector_name}")


class ConnectorValidationError(Exception):
    """Connector validation exception"""

    pass


class CredentialExpiredError(Exception):
    """Credential expired exception"""

    pass


class InsufficientPermissionsError(Exception):
    """Insufficient permissions exception"""

    pass


class UnexpectedValidationError(Exception):
    """Unexpected validation exception"""

    pass


class RateLimitTriedTooManyTimesError(Exception):
    pass


class DirectoryListingError(Exception):
    """Exception raised when directory listing fails."""

    def __init__(self, path: str, error_details: str):
        self.path = path
        self.error_details = error_details
        super().__init__(f"Failed to list directory '{path}': {error_details}")
