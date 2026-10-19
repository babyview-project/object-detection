"""Databrary REST client (https://api.databrary.org).

Databrary's public site is a SPA. Programmatic access uses JWT bearer tokens
from POST /login/ plus TOTP (POST /totp/verify/). Requests must send a browser
Origin header or nginx returns 403 ("only accessible via a browser...").
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping
from urllib.parse import urljoin

import requests

API_BASE = "https://api.databrary.org"
DEFAULT_ORIGIN = "https://nyu.databrary.org"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
DEFAULT_PAGE_SIZE = 100
TOKEN_CACHE_PATH = Path.home() / ".cache" / "databrary" / "tokens.json"
_PACKAGE_ENV = Path(__file__).resolve().parent / ".env"


def parse_rate(value: str | None) -> float | None:
    """Parse a rate like '20M', '20MB/s', or '20971520' into bytes/second."""
    if value is None:
        return None
    raw = value.strip().lower().replace("/s", "").replace(" ", "")
    if not raw or raw in {"0", "none", "off", "unlimited"}:
        return None
    multipliers = {
        "k": 1000,
        "kb": 1000,
        "kib": 1024,
        "m": 1000**2,
        "mb": 1000**2,
        "mib": 1024**2,
        "g": 1000**3,
        "gb": 1000**3,
        "gib": 1024**3,
    }
    for suffix, factor in sorted(multipliers.items(), key=lambda kv: -len(kv[0])):
        if raw.endswith(suffix):
            return float(raw[: -len(suffix)]) * factor
    return float(raw)


def load_env_file(path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from databrary/.env without requiring python-dotenv."""
    env_path = path or _PACKAGE_ENV
    if not env_path.is_file():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


class DatabraryAPIError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, payload: Any = None):
        super().__init__(message)
        self.status = status
        self.payload = payload


def _error_message(response: requests.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        text = (response.text or "").strip()
        return text[:500] or f"HTTP {response.status_code}"
    if isinstance(data, dict):
        if "nonFieldErrors" in data:
            errs = data["nonFieldErrors"]
            return "; ".join(str(x) for x in errs) if isinstance(errs, list) else str(errs)
        parts = []
        for key, value in data.items():
            if isinstance(value, list):
                parts.append(f"{key}: {'; '.join(str(v) for v in value)}")
            else:
                parts.append(f"{key}: {value}")
        if parts:
            return "; ".join(parts)
    return str(data)


def _totp_from_secret(secret: str) -> str:
    try:
        import pyotp
    except ImportError as exc:
        raise DatabraryAPIError(
            "DATABRARY_TOTP_SECRET is set but pyotp is not installed. "
            "pip install pyotp"
        ) from exc
    return pyotp.TOTP(secret.replace(" ", "")).now()


class DatabraryClient:
    """Session-backed client with login, token refresh, and volume downloads."""

    def __init__(
        self,
        *,
        email: str | None = None,
        password: str | None = None,
        origin: str = DEFAULT_ORIGIN,
        token_path: Path | None = None,
        timeout: float = 60.0,
    ) -> None:
        load_env_file()
        self.email = email or os.environ.get("DATABRARY_EMAIL")
        self.password = password or os.environ.get("DATABRARY_PASSWORD")
        self.origin = origin.rstrip("/")
        env_token = os.environ.get("DATABRARY_TOKEN_PATH")
        self.token_path = Path(
            token_path or env_token or TOKEN_CACHE_PATH
        ).expanduser()
        self.timeout = timeout
        self.access_token: str | None = None
        self.refresh_token: str | None = None
        self._lock = threading.RLock()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
                "Origin": self.origin,
                "Referer": f"{self.origin}/",
            }
        )
        self._load_tokens()

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "DatabraryClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _url(self, *parts: Any) -> str:
        path = "/".join(str(p).strip("/") for p in parts if p is not None and str(p) != "")
        return urljoin(API_BASE + "/", path + "/")

    def _load_tokens(self) -> None:
        if not self.token_path.is_file():
            return
        try:
            data = json.loads(self.token_path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        self.access_token = data.get("access")
        self.refresh_token = data.get("refresh")

    def _save_tokens(self) -> None:
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text(
            json.dumps({"access": self.access_token, "refresh": self.refresh_token}, indent=2)
            + "\n"
        )
        try:
            os.chmod(self.token_path, 0o600)
        except OSError:
            pass

    def _clear_tokens(self) -> None:
        self.access_token = None
        self.refresh_token = None
        if self.token_path.is_file():
            try:
                self.token_path.unlink()
            except OSError:
                pass

    def _auth_headers(self) -> dict[str, str]:
        if not self.access_token:
            return {}
        return {"Authorization": f"Bearer {self.access_token}"}

    def request(
        self,
        method: str,
        *parts: Any,
        params: Mapping[str, Any] | None = None,
        json_body: Any = None,
        retry_refresh: bool = True,
        stream: bool = False,
        url: str | None = None,
        timeout: float | None = None,
    ) -> requests.Response:
        target = url or self._url(*parts)
        headers = dict(self._auth_headers())
        if json_body is not None:
            headers["Content-Type"] = "application/json"
        with self._lock:
            response = self.session.request(
                method,
                target,
                params=params,
                json=json_body,
                headers=headers,
                timeout=timeout or self.timeout,
                stream=stream,
                allow_redirects=True,
            )
            need_refresh = (
                response.status_code == 401
                and retry_refresh
                and bool(self.refresh_token)
                and "/login/" not in target
                and "/totp/" not in target
                and "/refresh-token/" not in target
            )
            if response.status_code == 403 and "only accessible via a browser" in response.text:
                raise DatabraryAPIError(
                    "Databrary blocked this request (WAF). Send Origin: "
                    f"{self.origin} (this client does). If it still fails, log in "
                    "from a browser on this machine first.",
                    status=403,
                    payload=response.text,
                )
            if not need_refresh:
                return response
            self._refresh_access_token()
        return self.request(
            method,
            *parts,
            params=params,
            json_body=json_body,
            retry_refresh=False,
            stream=stream,
            url=url,
            timeout=timeout,
        )

    def _json(self, response: requests.Response) -> Any:
        if not response.ok:
            raise DatabraryAPIError(
                _error_message(response), status=response.status_code, payload=response.text
            )
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise DatabraryAPIError(
                "Expected JSON from Databrary", status=response.status_code, payload=response.text
            ) from exc

    def login(
        self,
        *,
        email: str | None = None,
        password: str | None = None,
        otp: str | None = None,
        otp_prompt: Callable[[], str] | None = None,
    ) -> dict[str, Any]:
        """Log in. Completes TOTP if the API returns otpIssued."""
        email = email or self.email
        password = password or self.password
        if not email or not password:
            raise DatabraryAPIError(
                "Set DATABRARY_EMAIL and DATABRARY_PASSWORD, or pass email/password."
            )
        self.email = email
        self.password = password
        payload: dict[str, Any] = {"email": email, "password": password}
        data = self._json(self.request("POST", "login", json_body=payload, retry_refresh=False))
        if isinstance(data, dict) and "access" in data and "refresh" in data:
            return self._store_login_tokens(data)
        if not isinstance(data, dict):
            raise DatabraryAPIError("Unexpected login response", payload=data)
        if data.get("needEnrollment"):
            raise DatabraryAPIError(
                "This account still needs authenticator enrollment. "
                "Finish TOTP setup in the Databrary web UI, then retry."
            )
        if data.get("otpIssued"):
            code = otp or os.environ.get("DATABRARY_OTP")
            secret = os.environ.get("DATABRARY_TOTP_SECRET")
            if not code and secret:
                code = _totp_from_secret(secret)
            if not code and otp_prompt:
                code = otp_prompt()
            if not code:
                raise DatabraryAPIError(
                    "Login requires a 6-digit authenticator code. Pass otp=, "
                    "set DATABRARY_OTP, or DATABRARY_TOTP_SECRET."
                )
            return self.verify_otp(code)
        if data.get("error"):
            raise DatabraryAPIError(str(data["error"]), payload=data)
        raise DatabraryAPIError("Unexpected login response", payload=data)

    def verify_otp(self, otp: str) -> dict[str, Any]:
        data = self._json(
            self.request("POST", "totp", "verify", json_body={"otp": otp.strip()}, retry_refresh=False)
        )
        if not isinstance(data, dict) or "access" not in data:
            raise DatabraryAPIError("TOTP verify did not return tokens", payload=data)
        return self._store_login_tokens(data)

    def _store_login_tokens(self, data: Mapping[str, Any]) -> dict[str, Any]:
        self.access_token = str(data["access"])
        self.refresh_token = str(data["refresh"])
        self._save_tokens()
        return dict(data)

    def _refresh_access_token(self) -> None:
        if not self.refresh_token:
            raise DatabraryAPIError("Not logged in (no refresh token). Call login().")
        response = self.request(
            "POST",
            "refresh-token",
            json_body={"refresh": self.refresh_token},
            retry_refresh=False,
        )
        if not response.ok:
            self._clear_tokens()
            raise DatabraryAPIError(
                "Session expired. Call login() again. " + _error_message(response),
                status=response.status_code,
            )
        data = response.json()
        self.access_token = data["access"]
        if data.get("refresh"):
            self.refresh_token = data["refresh"]
        self._save_tokens()

    def logout(self) -> None:
        if self.refresh_token:
            try:
                self.request(
                    "POST", "logout", json_body={"refresh": self.refresh_token}, retry_refresh=False
                )
            except DatabraryAPIError:
                pass
        self._clear_tokens()

    def ensure_login(self, **login_kwargs: Any) -> None:
        if self.access_token:
            return
        if self.refresh_token:
            try:
                self._refresh_access_token()
                return
            except DatabraryAPIError:
                pass
        self.login(**login_kwargs)

    def get(self, *parts: Any, params: Mapping[str, Any] | None = None) -> Any:
        return self._json(self.request("GET", *parts, params=params))

    def iter_pages(
        self,
        *parts: Any,
        params: Mapping[str, Any] | None = None,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> Iterator[dict[str, Any]]:
        page = 1
        query = dict(params or {})
        query.setdefault("page_size", page_size)
        while True:
            query["page"] = page
            data = self.get(*parts, params=query)
            if isinstance(data, list):
                for item in data:
                    yield item
                return
            if not isinstance(data, dict):
                raise DatabraryAPIError("Expected paginated JSON object", payload=data)
            results = data.get("results", data.get("items", []))
            for item in results:
                yield item
            total_pages = int(data.get("totalPages") or data.get("total_pages") or 1)
            if page >= total_pages or not results:
                return
            page += 1

    def search_volumes(self, q: str, **params: Any) -> list[dict[str, Any]]:
        query = {"q": q, **params}
        return list(self.iter_pages("search", "volumes", params=query))

    def get_volume(self, volume_id: int | str) -> dict[str, Any]:
        return self.get("volumes", volume_id)

    def list_sessions(self, volume_id: int | str, **params: Any) -> list[dict[str, Any]]:
        return list(self.iter_pages("volumes", volume_id, "sessions", params=params))

    def get_session(self, volume_id: int | str, session_id: int | str) -> dict[str, Any]:
        return self.get("volumes", volume_id, "sessions", session_id)

    def list_session_files(
        self, volume_id: int | str, session_id: int | str, **params: Any
    ) -> list[dict[str, Any]]:
        return list(
            self.iter_pages("volumes", volume_id, "sessions", session_id, "files", params=params)
        )

    def list_folders(self, volume_id: int | str, **params: Any) -> list[dict[str, Any]]:
        return list(self.iter_pages("volumes", volume_id, "folders", params=params))

    def list_folder_files(
        self, volume_id: int | str, folder_id: int | str, **params: Any
    ) -> list[dict[str, Any]]:
        return list(
            self.iter_pages("volumes", volume_id, "folders", folder_id, "files", params=params)
        )

    def file_download_url(
        self, volume_id: int | str, session_id: int | str, file_id: int | str
    ) -> str:
        data = self.get("volumes", volume_id, "sessions", session_id, "files", file_id, "download-link")
        url = data.get("downloadUrl") if isinstance(data, dict) else None
        if not url:
            raise DatabraryAPIError("No downloadUrl in response", payload=data)
        return str(url)

    def volume_zip_url(self, volume_id: int | str) -> str:
        data = self.get("volumes", volume_id, "download-link")
        url = data.get("downloadUrl") if isinstance(data, dict) else None
        if not url:
            raise DatabraryAPIError("No downloadUrl in response", payload=data)
        return str(url)

    def volume_csv_url(self, volume_id: int | str) -> str:
        data = self.get("volumes", volume_id, "csv-download-link")
        url = data.get("downloadUrl") if isinstance(data, dict) else None
        if not url:
            raise DatabraryAPIError("No downloadUrl in response", payload=data)
        return str(url)

    def download_url(
        self,
        url: str,
        dest: Path,
        *,
        chunk_size: int = 8 << 20,
        max_rate_bytes: float | None = None,
        expected_size: int | None = None,
        max_retries: int = 6,
    ) -> Path:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                return self._download_url_once(
                    url,
                    dest,
                    tmp,
                    chunk_size=chunk_size,
                    max_rate_bytes=max_rate_bytes,
                    expected_size=expected_size,
                )
            except (requests.RequestException, DatabraryAPIError, OSError) as exc:
                last_error = exc
                wait = min(60.0, 2.0 ** attempt)
                print(
                    f"retry {attempt + 1}/{max_retries} {dest.name}: {exc}; sleep {wait:.0f}s",
                    flush=True,
                )
                time.sleep(wait)
        assert last_error is not None
        raise last_error

    def _download_url_once(
        self,
        url: str,
        dest: Path,
        tmp: Path,
        *,
        chunk_size: int,
        max_rate_bytes: float | None,
        expected_size: int | None,
    ) -> Path:
        if dest.is_file():
            got = dest.stat().st_size
            complete = (expected_size and got >= int(expected_size * 0.99)) or (
                not expected_size and got > 0
            )
            if complete:
                return dest
            if tmp.is_file() and tmp.stat().st_size >= got:
                dest.unlink()
            else:
                dest.replace(tmp)

        existing = tmp.stat().st_size if tmp.is_file() else 0
        if expected_size and existing > expected_size:
            tmp.unlink()
            existing = 0

        header_sets = [
            {
                "User-Agent": USER_AGENT,
                "Origin": self.origin,
                "Referer": f"{self.origin}/",
                **self._auth_headers(),
            },
            {"User-Agent": USER_AGENT, "Origin": self.origin, "Referer": f"{self.origin}/"},
            {"User-Agent": USER_AGENT},
        ]
        last_error: DatabraryAPIError | None = None
        for headers in header_sets:
            req_headers = dict(headers)
            if existing:
                req_headers["Range"] = f"bytes={existing}-"
            with requests.get(
                url,
                stream=True,
                timeout=(30, max(self.timeout, 600)),
                headers=req_headers,
            ) as response:
                if existing and response.status_code == 200:
                    tmp.unlink(missing_ok=True)
                    existing = 0
                if not response.ok:
                    last_error = DatabraryAPIError(
                        f"Download failed: HTTP {response.status_code}",
                        status=response.status_code,
                    )
                    continue
                mode = "ab" if existing and response.status_code == 206 else "wb"
                if mode == "wb":
                    existing = 0
                written = existing
                started = time.monotonic()
                last_report = started
                with tmp.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        written += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 30:
                            mb_s = (written - existing) / max(now - started, 1e-6) / 1e6
                            print(
                                f"  {dest.name}: {written / (1024 ** 3):.2f} GiB "
                                f"({mb_s:.2f} MB/s)",
                                flush=True,
                            )
                            last_report = now
                        if max_rate_bytes and max_rate_bytes > 0:
                            expected = (written - existing) / max_rate_bytes
                            lag = expected - (now - started)
                            if lag > 0:
                                time.sleep(lag)
                tmp.replace(dest)
                return dest
        assert last_error is not None
        raise last_error

    def download_session_file(
        self,
        volume_id: int | str,
        session_id: int | str,
        file_id: int | str,
        dest: Path,
        *,
        max_rate_bytes: float | None = None,
        expected_size: int | None = None,
    ) -> Path:
        return self.download_url(
            self.file_download_url(volume_id, session_id, file_id),
            dest,
            max_rate_bytes=max_rate_bytes,
            expected_size=expected_size,
        )

    def download_volume_zip(self, volume_id: int | str, dest: Path) -> Path:
        dest = Path(dest)
        if dest.suffix.lower() != ".zip":
            dest = dest / f"databrary_volume_{volume_id}.zip"
        return self.download_url(self.volume_zip_url(volume_id), dest)

    def download_volume_csv(self, volume_id: int | str, dest: Path) -> Path:
        dest = Path(dest)
        if dest.suffix.lower() != ".csv":
            dest = dest / f"databrary_volume_{volume_id}.csv"
        return self.download_url(self.volume_csv_url(volume_id), dest)

    def plan_world_subset(
        self,
        volume_id: int | str,
        *,
        max_sessions: int | None = 80,
        seed: int = 0,
        workers: int = 8,
    ) -> list[dict[str, Any]]:
        """Resolve world videos. max_sessions=None means every session in the volume."""
        from .subset import pick_world_file, session_setting, session_task, stratified_sample

        sessions = self.list_sessions(volume_id)
        if max_sessions is None or max_sessions >= len(sessions):
            chosen = sessions
        else:
            chosen = stratified_sample(sessions, max_sessions, seed=seed)

        def describe(session: dict[str, Any]) -> dict[str, Any] | None:
            sid = session["id"]
            files = self.list_session_files(volume_id, sid)
            world = pick_world_file(files)
            if not world:
                return None
            size = world.get("size") or 0
            return {
                "volume_id": int(volume_id),
                "session_id": sid,
                "file_id": world.get("id"),
                "file_name": world.get("name") or world.get("filename"),
                "size_bytes": int(size) if size else 0,
                "setting": session_setting(session),
                "task": session_task(session),
                "release_level": world.get("releaseLevel"),
            }

        rows: list[dict[str, Any]] = []
        n_workers = max(1, int(workers))
        if n_workers == 1:
            for session in chosen:
                row = describe(session)
                if row:
                    rows.append(row)
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(describe, session) for session in chosen]
                for fut in as_completed(futures):
                    row = fut.result()
                    if row:
                        rows.append(row)
        rows.sort(key=lambda r: int(r["session_id"]))
        return rows

    def download_world_subset(
        self,
        volume_id: int | str,
        dest_dir: Path,
        *,
        max_sessions: int | None = 80,
        seed: int = 0,
        skip_existing: bool = True,
        dry_run: bool = False,
        max_rate_bytes: float | None = None,
        pause_s: float = 0.0,
        workers: int = 8,
    ) -> list[dict[str, Any]]:
        from .subset import world_dest_path

        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        n_workers = max(1, int(workers))
        rows = self.plan_world_subset(
            volume_id, max_sessions=max_sessions, seed=seed, workers=n_workers
        )
        pending: list[dict[str, Any]] = []
        for row in rows:
            path = world_dest_path(dest_dir, row["session_id"], row["file_name"])
            row["dest"] = str(path)
            expected = int(row.get("size_bytes") or 0)
            if dry_run:
                row["status"] = "planned"
                continue
            if skip_existing and path.is_file() and (
                not expected or path.stat().st_size >= int(expected * 0.99)
            ):
                row["status"] = "exists"
                continue
            pending.append(row)

        if dry_run:
            return rows

        print(
            f"planned {len(rows)} world videos; {len(pending)} to fetch "
            f"with {n_workers} workers -> {dest_dir}",
            flush=True,
        )

        def fetch(row: dict[str, Any]) -> dict[str, Any]:
            if pause_s > 0:
                time.sleep(pause_s)
            path = Path(row["dest"])
            print(
                f"downloading session {row['session_id']} "
                f"({int(row.get('size_bytes') or 0) / (1024 ** 3):.2f} GiB) -> {path}",
                flush=True,
            )
            try:
                self.download_session_file(
                    volume_id,
                    row["session_id"],
                    row["file_id"],
                    path,
                    max_rate_bytes=max_rate_bytes,
                    expected_size=int(row.get("size_bytes") or 0) or None,
                )
                row["status"] = "downloaded"
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = str(exc)
                print(f"FAILED session {row['session_id']}: {exc}", flush=True)
            return row

        if not pending:
            return rows
        if n_workers == 1:
            for row in pending:
                fetch(row)
            return rows
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(fetch, row) for row in pending]
            for fut in as_completed(futures):
                fut.result()
        return rows
