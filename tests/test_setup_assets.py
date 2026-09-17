import os
import sys
import io
import json
import pytest
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
import urllib.error

from scripts.setup_assets import (
    get_github_token,
    resolve_private_release_asset_url,
    download_file_with_progress,
    SafeAuthRedirectHandler,
    ARTIFACT_MANIFEST,
    verify_installed_artifacts,
    extract_archive
)


class TestSetupAssetsAuthentication:
    def test_get_github_token_priority(self, monkeypatch):
        # 1. Neither present
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        assert get_github_token() is None

        # 2. Only GH_TOKEN present
        monkeypatch.setenv("GH_TOKEN", "gh_token_value_123")
        assert get_github_token() == "gh_token_value_123"

        # 3. Both present -> GITHUB_TOKEN preferred
        monkeypatch.setenv("GITHUB_TOKEN", "github_token_value_456")
        assert get_github_token() == "github_token_value_456"

        # 4. Empty/whitespace ignored
        monkeypatch.setenv("GITHUB_TOKEN", "   ")
        assert get_github_token() == "gh_token_value_123"

    def test_unauthenticated_download_request(self):
        with patch("urllib.request.build_opener") as mock_build_opener:
            mock_opener = MagicMock()
            mock_response = MagicMock()
            mock_response.headers.get.return_value = "100"
            mock_response.read.side_effect = [b"data_chunk", b""]
            mock_opener.open.return_value.__enter__.return_value = mock_response
            mock_build_opener.return_value = mock_opener

            dest = Path("test_temp_dest.zip")
            try:
                download_file_with_progress("https://github.com/LiliesInLakes/techquest/releases/download/v1.0.0/test.zip", dest, token=None)
                
                # Check request
                args, _ = mock_opener.open.call_args
                req = args[0]
                assert "Authorization" not in req.headers
                assert req.headers["User-agent"] == "TechQuest-Asset-Downloader/1.0"
            finally:
                if dest.exists():
                    dest.unlink()

    def test_authenticated_download_request_sends_bearer(self):
        with patch("urllib.request.build_opener") as mock_build_opener:
            mock_opener = MagicMock()
            mock_response = MagicMock()
            mock_response.headers.get.return_value = "100"
            mock_response.read.side_effect = [b"data_chunk", b""]
            mock_opener.open.return_value.__enter__.return_value = mock_response
            mock_build_opener.return_value = mock_opener

            dest = Path("test_temp_dest.zip")
            secret_token = "ghp_secret_token_abc_xyz"
            try:
                download_file_with_progress(
                    "https://api.github.com/repos/LiliesInLakes/techquest/releases/assets/12345",
                    dest,
                    token=secret_token,
                    is_api_asset=True
                )
                
                args, _ = mock_opener.open.call_args
                req = args[0]
                assert req.headers["Authorization"] == f"Bearer {secret_token}"
                assert req.headers["Accept"] == "application/octet-stream"
            finally:
                if dest.exists():
                    dest.unlink()

    def test_resolve_private_release_asset_url(self):
        mock_release_payload = {
            "tag_name": "v1.0.0",
            "assets": [
                {
                    "name": "unrelated.txt",
                    "url": "https://api.github.com/repos/owner/repo/releases/assets/111"
                },
                {
                    "name": "techquest-artifacts-v1.0.0.zip",
                    "url": "https://api.github.com/repos/owner/repo/releases/assets/999"
                }
            ]
        }

        with patch("urllib.request.build_opener") as mock_build_opener:
            mock_opener = MagicMock()
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(mock_release_payload).encode("utf-8")
            mock_opener.open.return_value.__enter__.return_value = mock_resp
            mock_build_opener.return_value = mock_opener

            url = resolve_private_release_asset_url("owner/repo", "v1.0.0", "techquest-artifacts-v1.0.0.zip", "test_tok")
            assert url == "https://api.github.com/repos/owner/repo/releases/assets/999"

    def test_token_never_logged_or_printed(self, capsys):
        secret_token = "ghp_VERY_SECRET_TOKEN_DO_NOT_LEAK"
        with patch("urllib.request.build_opener") as mock_build_opener:
            mock_opener = MagicMock()
            mock_response = MagicMock()
            mock_response.headers.get.return_value = "10"
            mock_response.read.side_effect = [b"data", b""]
            mock_opener.open.return_value.__enter__.return_value = mock_response
            mock_build_opener.return_value = mock_opener

            dest = Path("test_temp_dest.zip")
            try:
                download_file_with_progress("https://api.github.com/assets/1", dest, token=secret_token)
                captured = capsys.readouterr()
                assert secret_token not in captured.out
                assert secret_token not in captured.err
                assert "Connecting with GitHub authentication..." in captured.out
            finally:
                if dest.exists():
                    dest.unlink()

    def test_http_401_unauthorized_error_message(self):
        with patch("urllib.request.build_opener") as mock_build_opener:
            mock_opener = MagicMock()
            mock_opener.open.side_effect = urllib.error.HTTPError(
                "https://api.github.com/repos/owner/repo/releases/tags/v1.0.0",
                401,
                "Unauthorized",
                {},
                io.BytesIO(b"Unauthorized")
            )
            mock_build_opener.return_value = mock_opener

            with pytest.raises(RuntimeError) as exc_info:
                resolve_private_release_asset_url("owner/repo", "v1.0.0", "test.zip", "bad_tok")
            
            msg = str(exc_info.value)
            assert "HTTP 401: Unauthorized" in msg
            assert "GITHUB_TOKEN or GH_TOKEN is valid" in msg
            assert "bad_tok" not in msg

    def test_http_403_forbidden_error_message(self):
        with patch("urllib.request.build_opener") as mock_build_opener:
            mock_opener = MagicMock()
            mock_opener.open.side_effect = urllib.error.HTTPError(
                "https://api.github.com/repos/owner/repo/releases/tags/v1.0.0",
                403,
                "Forbidden",
                {},
                io.BytesIO(b"Forbidden")
            )
            mock_build_opener.return_value = mock_opener

            with pytest.raises(RuntimeError) as exc_info:
                resolve_private_release_asset_url("owner/repo", "v1.0.0", "test.zip", "tok")
            
            msg = str(exc_info.value)
            assert "HTTP 403: Forbidden" in msg
            assert "repository read permissions" in msg

    def test_http_404_notes_private_repo_token_when_unauthenticated(self):
        with patch("urllib.request.build_opener") as mock_build_opener:
            mock_opener = MagicMock()
            mock_opener.open.side_effect = urllib.error.HTTPError(
                "https://github.com/owner/repo/releases/download/v1.0.0/test.zip",
                404,
                "Not Found",
                {},
                io.BytesIO(b"Not Found")
            )
            mock_build_opener.return_value = mock_opener

            dest = Path("test_temp_dest.zip")
            try:
                with pytest.raises(RuntimeError) as exc_info:
                    download_file_with_progress("https://github.com/owner/repo/releases/download/v1.0.0/test.zip", dest, token=None)
                
                msg = str(exc_info.value)
                assert "HTTP 404" in msg
                assert "private repository" in msg
                assert "$env:GITHUB_TOKEN" in msg
            finally:
                if dest.exists():
                    dest.unlink()

    def test_safe_auth_redirect_handler_drops_auth_on_s3_redirect(self):
        handler = SafeAuthRedirectHandler()
        req = urllib.request.Request(
            "https://api.github.com/repos/owner/repo/releases/assets/123",
            headers={"Authorization": "Bearer ghp_token", "User-Agent": "Test"}
        )
        
        # Cross domain redirect to objects.githubusercontent.com / AWS S3
        new_req = handler.redirect_request(req, None, 302, "Found", {}, "https://objects.githubusercontent.com/github-production-release-asset/123")
        assert new_req is not None
        assert "Authorization" not in new_req.headers
        assert "authorization" not in [k.lower() for k in new_req.headers.keys()]
