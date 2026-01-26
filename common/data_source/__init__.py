"""
Thanks to https://github.com/onyx-dot-app/onyx

Content of this directory is under the "MIT Expat" license as defined below.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .blob_connector import BlobStorageConnector
    from .slack_connector import SlackConnector
    from .gmail_connector import GmailConnector
    from .notion_connector import NotionConnector
    from .confluence_connector import ConfluenceConnector
    from .discord_connector import DiscordConnector
    from .dropbox_connector import DropboxConnector
    from .google_drive.connector import GoogleDriveConnector
    from .jira.connector import JiraConnector
    from .sharepoint_connector import SharePointConnector
    from .teams_connector import TeamsConnector
    from .moodle_connector import MoodleConnector
    from .airtable_connector import AirtableConnector
    from .asana_connector import AsanaConnector
    from .imap_connector import ImapConnector
    from .zendesk_connector import ZendeskConnector

_LAZY_CONNECTORS = {
    "BlobStorageConnector": (".blob_connector", "BlobStorageConnector"),
    "SlackConnector": (".slack_connector", "SlackConnector"),
    "GmailConnector": (".gmail_connector", "GmailConnector"),
    "NotionConnector": (".notion_connector", "NotionConnector"),
    "ConfluenceConnector": (".confluence_connector", "ConfluenceConnector"),
    "DiscordConnector": (".discord_connector", "DiscordConnector"),
    "DropboxConnector": (".dropbox_connector", "DropboxConnector"),
    "GoogleDriveConnector": (".google_drive.connector", "GoogleDriveConnector"),
    "JiraConnector": (".jira.connector", "JiraConnector"),
    "SharePointConnector": (".sharepoint_connector", "SharePointConnector"),
    "TeamsConnector": (".teams_connector", "TeamsConnector"),
    "MoodleConnector": (".moodle_connector", "MoodleConnector"),
    "AirtableConnector": (".airtable_connector", "AirtableConnector"),
    "AsanaConnector": (".asana_connector", "AsanaConnector"),
    "ImapConnector": (".imap_connector", "ImapConnector"),
    "ZendeskConnector": (".zendesk_connector", "ZendeskConnector"),
}


def __getattr__(name):
    if name in _LAZY_CONNECTORS:
        module_path, class_name = _LAZY_CONNECTORS[name]
        try:
            import importlib

            module = importlib.import_module(module_path, package=__package__)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not lazy load {name} from {module_path}: {e}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from .config import BlobType, DocumentSource
from .models import Document, TextSection, ImageSection, BasicExpertInfo
from .exceptions import ConnectorMissingCredentialError, ConnectorValidationError, CredentialExpiredError, InsufficientPermissionsError, UnexpectedValidationError

__all__ = [
    "BlobStorageConnector",
    "SlackConnector",
    "GmailConnector",
    "NotionConnector",
    "ConfluenceConnector",
    "DiscordConnector",
    "DropboxConnector",
    "GoogleDriveConnector",
    "JiraConnector",
    "SharePointConnector",
    "TeamsConnector",
    "MoodleConnector",
    "BlobType",
    "DocumentSource",
    "Document",
    "TextSection",
    "ImageSection",
    "BasicExpertInfo",
    "ConnectorMissingCredentialError",
    "ConnectorValidationError",
    "CredentialExpiredError",
    "InsufficientPermissionsError",
    "UnexpectedValidationError",
    "AirtableConnector",
    "AsanaConnector",
    "ImapConnector",
    "ZendeskConnector",
]
