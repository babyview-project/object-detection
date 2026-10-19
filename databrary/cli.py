"""Command-line interface: python -m databrary <command>."""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path

from .client import DatabraryAPIError, DatabraryClient, parse_rate


def _prompt_otp() -> str:
    return input("Authenticator code (6 digits): ").strip()


def _otp_prompt_or_none() -> object:
    """Interactive TOTP only on a TTY (sbatch jobs must use cached tokens or TOTP secret)."""
    return _prompt_otp if sys.stdin.isatty() else None


def _client_from_args(args: argparse.Namespace) -> DatabraryClient:
    return DatabraryClient(
        email=getattr(args, "email", None),
        password=getattr(args, "password", None),
        token_path=Path(args.token_path) if getattr(args, "token_path", None) else None,
    )


def _login(client: DatabraryClient, args: argparse.Namespace) -> None:
    password = args.password or client.password
    if not password:
        password = getpass.getpass("Databrary password: ")
    client.login(
        email=args.email or client.email,
        password=password,
        otp=args.otp,
        otp_prompt=_otp_prompt_or_none(),
    )
    print(f"Logged in. Tokens saved to {client.token_path}")


def _dump(obj: object) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_login(args: argparse.Namespace) -> int:
    with _client_from_args(args) as client:
        _login(client, args)
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    with _client_from_args(args) as client:
        client.ensure_login(otp_prompt=_otp_prompt_or_none())
        hits = client.search_volumes(args.query)
        if args.json:
            _dump(hits)
            return 0
        for vol in hits:
            vid = vol.get("id", "?")
            name = vol.get("name") or vol.get("title") or ""
            print(f"{vid}\t{name}")
    return 0


def cmd_volume(args: argparse.Namespace) -> int:
    with _client_from_args(args) as client:
        client.ensure_login(otp_prompt=_otp_prompt_or_none())
        _dump(client.get_volume(args.volume_id))
    return 0


def cmd_sessions(args: argparse.Namespace) -> int:
    with _client_from_args(args) as client:
        client.ensure_login(otp_prompt=_otp_prompt_or_none())
        sessions = client.list_sessions(args.volume_id)
        if args.json:
            _dump(sessions)
            return 0
        for session in sessions:
            sid = session.get("id", "?")
            name = session.get("name") or session.get("displayName") or ""
            date = session.get("date") or ""
            n_files = session.get("fileCount") or session.get("files_count") or ""
            print(f"{sid}\t{date}\t{n_files}\t{name}")
        print(f"# {len(sessions)} sessions", file=sys.stderr)
    return 0


def cmd_files(args: argparse.Namespace) -> int:
    with _client_from_args(args) as client:
        client.ensure_login(otp_prompt=_otp_prompt_or_none())
        files = client.list_session_files(args.volume_id, args.session_id)
        if args.json:
            _dump(files)
            return 0
        for item in files:
            fid = item.get("id", "?")
            name = item.get("name") or item.get("filename") or ""
            size = item.get("size") or ""
            mime = item.get("mimeType") or item.get("format") or ""
            print(f"{fid}\t{size}\t{mime}\t{name}")
        print(f"# {len(files)} files", file=sys.stderr)
    return 0


def cmd_download_file(args: argparse.Namespace) -> int:
    dest = Path(args.out)
    with _client_from_args(args) as client:
        client.ensure_login(otp_prompt=_otp_prompt_or_none())
        path = client.download_session_file(
            args.volume_id, args.session_id, args.file_id, dest
        )
    print(path)
    return 0


def cmd_download_volume(args: argparse.Namespace) -> int:
    dest = Path(args.out)
    with _client_from_args(args) as client:
        client.ensure_login(otp_prompt=_otp_prompt_or_none())
        if args.csv:
            path = client.download_volume_csv(args.volume_id, dest)
        else:
            path = client.download_volume_zip(args.volume_id, dest)
    print(path)
    return 0


def cmd_logout(args: argparse.Namespace) -> int:
    with _client_from_args(args) as client:
        client.logout()
    print("Logged out.")
    return 0


def cmd_download_world(args: argparse.Namespace) -> int:
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    max_sessions = None if args.all else args.max_sessions
    with _client_from_args(args) as client:
        if not args.dry_run:
            client.ensure_login(otp_prompt=_otp_prompt_or_none())
        rows = client.download_world_subset(
            args.volume_id,
            dest,
            max_sessions=max_sessions,
            seed=args.seed,
            dry_run=args.dry_run,
            max_rate_bytes=parse_rate(args.max_rate),
            pause_s=args.pause,
            workers=args.workers,
        )
    total = sum(int(r.get("size_bytes") or 0) for r in rows)
    settings = sorted({str(r.get("setting")) for r in rows})
    n_failed = sum(1 for r in rows if r.get("status") == "failed")
    print("session_id\tGiB\tsetting\ttask\tstatus\tdest")
    for row in rows:
        gib = int(row.get("size_bytes") or 0) / (1024 ** 3)
        print(
            f"{row.get('session_id')}\t{gib:.2f}\t{row.get('setting')}\t"
            f"{row.get('task')}\t{row.get('status')}\t{row.get('dest')}"
        )
    print(f"# sessions {len(rows)}", file=sys.stderr)
    print(f"# settings {len(settings)}: {', '.join(settings)}", file=sys.stderr)
    print(f"# total {total / (1024 ** 3):.1f} GiB", file=sys.stderr)
    print(f"# failed {n_failed}", file=sys.stderr)
    manifest = dest / "world_subset_manifest.json"
    manifest.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"# manifest {manifest}", file=sys.stderr)
    return 1 if n_failed else 0


def _add_auth_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--email", help="Databrary account email (or DATABRARY_EMAIL)")
    parser.add_argument("--password", help="Password (or DATABRARY_PASSWORD / prompt)")
    parser.add_argument("--otp", help="6-digit authenticator code (or DATABRARY_OTP)")
    parser.add_argument(
        "--token-path",
        help="Where to cache JWT tokens (default: ~/.cache/databrary/tokens.json)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m databrary",
        description="Log in to Databrary and list/download volumes.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_login = sub.add_parser("login", help="Authenticate and cache JWT tokens")
    _add_auth_args(p_login)
    p_login.set_defaults(func=cmd_login)

    p_search = sub.add_parser("search", help="Search volumes")
    _add_auth_args(p_search)
    p_search.add_argument("query")
    p_search.add_argument("--json", action="store_true")
    p_search.set_defaults(func=cmd_search)

    p_vol = sub.add_parser("volume", help="Print volume JSON")
    _add_auth_args(p_vol)
    p_vol.add_argument("volume_id")
    p_vol.set_defaults(func=cmd_volume)

    p_sess = sub.add_parser("sessions", help="List sessions in a volume")
    _add_auth_args(p_sess)
    p_sess.add_argument("volume_id")
    p_sess.add_argument("--json", action="store_true")
    p_sess.set_defaults(func=cmd_sessions)

    p_files = sub.add_parser("files", help="List files in a session")
    _add_auth_args(p_files)
    p_files.add_argument("volume_id")
    p_files.add_argument("session_id")
    p_files.add_argument("--json", action="store_true")
    p_files.set_defaults(func=cmd_files)

    p_df = sub.add_parser("download-file", help="Download one session file")
    _add_auth_args(p_df)
    p_df.add_argument("volume_id")
    p_df.add_argument("session_id")
    p_df.add_argument("file_id")
    p_df.add_argument("-o", "--out", required=True, help="Destination file path")
    p_df.set_defaults(func=cmd_download_file)

    p_dv = sub.add_parser("download-volume", help="Download volume ZIP (or CSV with --csv)")
    _add_auth_args(p_dv)
    p_dv.add_argument("volume_id")
    p_dv.add_argument("-o", "--out", required=True, help="Destination file or directory")
    p_dv.add_argument("--csv", action="store_true", help="Download spreadsheet instead of ZIP")
    p_dv.set_defaults(func=cmd_download_volume)

    p_out = sub.add_parser("logout", help="Drop cached tokens")
    _add_auth_args(p_out)
    p_out.set_defaults(func=cmd_logout)

    p_world = sub.add_parser(
        "download-world",
        help="Download world-camera videos only (subset or --all)",
    )
    _add_auth_args(p_world)
    p_world.add_argument("volume_id")
    p_world.add_argument("-o", "--out", required=True, help="Destination directory")
    p_world.add_argument(
        "--max-sessions",
        type=int,
        default=80,
        help="How many sessions to keep (default: 80). Ignored with --all.",
    )
    p_world.add_argument(
        "--all",
        action="store_true",
        help="Download every session's world video (~1.24 TiB for VEDB 1612)",
    )
    p_world.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel HTTP downloads (ThreadPoolExecutor; default: 8)",
    )
    p_world.add_argument("--seed", type=int, default=0, help="Subset sampling seed")
    p_world.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the subset and print sizes; do not download",
    )
    p_world.add_argument(
        "--max-rate",
        default="0",
        help="Per-stream write cap (default: 0 unlimited; e.g. 20M on shared NFS)",
    )
    p_world.add_argument(
        "--pause",
        type=float,
        default=0.0,
        help="Seconds to wait before each file (default: 0)",
    )
    p_world.set_defaults(func=cmd_download_world)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except DatabraryAPIError as exc:
        print(f"databrary: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
