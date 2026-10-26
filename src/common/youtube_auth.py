"""OAuth for the YouTube Data API.

One-time setup (opens a browser):  uv run python -m src.common.youtube_auth
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from src.common.config import Secrets, load_secrets

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]
_AUTHORIZE_HINT = "Run: uv run python -m src.common.youtube_auth"


class AuthError(RuntimeError):
    """No usable OAuth token."""


def load_credentials(secrets: Secrets) -> Credentials:
    token_file = secrets.youtube_token_file
    if not token_file.is_file():
        raise AuthError(f"No OAuth token at {token_file}. {_AUTHORIZE_HINT}")
    creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
    if creds.valid:
        return creds
    if not creds.refresh_token:
        raise AuthError(f"OAuth token has no refresh token. {_AUTHORIZE_HINT}")
    try:
        creds.refresh(Request())
    except RefreshError as exc:
        raise AuthError(
            "Google rejected the refresh token. If the OAuth consent screen is still in "
            "'Testing' mode, tokens expire after 7 days: publish it to production. "
            + _AUTHORIZE_HINT
        ) from exc
    _save(token_file, creds)
    return creds


def build_youtube(secrets: Secrets) -> Any:
    return build("youtube", "v3", credentials=load_credentials(secrets), cache_discovery=False)


def authorize_interactive(secrets: Secrets) -> Path:
    client_file = secrets.youtube_client_secret_file
    if not client_file.is_file():
        raise AuthError(
            f"OAuth client file not found at {client_file}. Download the Desktop-app client "
            "JSON from Google Cloud Console and set YOUTUBE_CLIENT_SECRET_FILE in .env."
        )
    flow = InstalledAppFlow.from_client_secrets_file(str(client_file), SCOPES)
    creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
    _save(secrets.youtube_token_file, creds)
    return secrets.youtube_token_file


def _save(path: Path, creds: Credentials) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(creds.to_json())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(
        prog="python -m src.common.youtube_auth",
        description="Authorize AutoClip to read your channel and upload private videos. "
        "Opens a browser and saves the token to YOUTUBE_TOKEN_FILE.",
    ).parse_args(argv)
    try:
        path = authorize_interactive(load_secrets())
    except AuthError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"Saved OAuth token to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
