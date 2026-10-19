# Databrary Python API

Sherlock checkout (scripts only):
[`babyview-project/databrary-download`](https://github.com/babyview-project/databrary-download).

Client for Databrary’s current REST API at `https://api.databrary.org` (the
2026 site). Older wrappers (`pydatabrary` / `databraryr`) talk to
`nyu.databrary.org/api`, which now serves the web app HTML instead of JSON.

You need an **authorized Databrary account** with TOTP already set up. Restricted
volumes also require access under Databrary’s data-use agreement.

## Install

From the repo root:

```bash
pip install -r databrary/requirements.txt
```

## Credentials

```bash
export DATABRARY_EMAIL='you@institution.edu'
export DATABRARY_PASSWORD='...'   # or omit and the CLI will prompt
```

Do not commit passwords. `databrary/.env` is gitignored if you keep a local copy
of `env.example`.

Login always asks for a 6-digit authenticator code unless you set
`DATABRARY_OTP` (one-shot) or `DATABRARY_TOTP_SECRET` (the authenticator setup
key; `pyotp` then generates codes). JWT tokens are cached at
`~/.cache/databrary/tokens.json`. Batch jobs (Sherlock `sbatch`) are not a TTY,
so they will not prompt; log in once interactively or put `DATABRARY_TOTP_SECRET`
in `databrary/.env`.

## CLI

Run from the **repository root** so `databrary` is importable:

```bash
cd /path/to/object-detection

python -m databrary login
python -m databrary search babyview
python -m databrary volume 1882
python -m databrary sessions 1882
python -m databrary files 1882 SESSION_ID
python -m databrary download-file 1882 SESSION_ID FILE_ID -o /tmp/clip.mp4
python -m databrary download-volume 1882 --csv -o /tmp/volume_1882.csv
python -m databrary download-volume 1882 -o /tmp/volume_1882.zip

# VEDB (volume 1612): world-camera videos only
python -m databrary download-world 1612 --dry-run --max-sessions 80 -o /path/to/out
python -m databrary download-world 1612 --all --workers 8 -o /path/to/out

python -m databrary logout
```

`download-world` uses a thread pool (not processes), HTTP Range resume on
`.part` files, and skips finished mp4s. Defaults are `--workers 8`,
`--max-rate 0`, `--pause 0`. On a shared NFS box (ccn2 `/ccn2a`) cap the write
rate, for example `--workers 4 --max-rate 20M --pause 1`.

World videos only: do not download unblurred eye cameras. Sherlock is not
approved for PII / High Risk data.

## Sherlock

Oak is mounted (`$OAK`, group `mcfrank`, 10 TB). HOME is 15 GB. `$SCRATCH` and
`$GROUP_SCRATCH` are 100 TB but **purged 90 days** after last content change.
Keep the 1.24 TiB VEDB world set on `$OAK`, not HOME or Scratch.

Do **not** run the download on a login node. Login is interactive TOTP only;
the transfer itself is an `sbatch` job on `normal`.

1. Clone or rsync this repo to `$GROUP_HOME` (not `$HOME`).
2. Copy `databrary/.env` there (gitignored). Optional: set
   `DATABRARY_TOTP_SECRET` so the job can refresh a dead session without a TTY.
3. On a login node:

```bash
ml python/3.12.1
python3 -m venv "$GROUP_HOME/$USER/venvs/databrary"
source "$GROUP_HOME/$USER/venvs/databrary/bin/activate"
pip install -r databrary/requirements.txt
cd /path/to/object-detection
python -m databrary login
```

4. Submit (writes to `$OAK/$USER/vedb_1612_world` by default):

```bash
sbatch databrary/sherlock_download_vedb.sbatch
# or override:
DEST=$OAK/$USER/vedb_1612_world WORKERS=8 sbatch databrary/sherlock_download_vedb.sbatch
```

Re-submit to resume: finished `.mp4` files are skipped; incomplete `.part`
files continue with `Range`. Walltime is 72 h (~1.24 TiB at ~8 MB/s aggregate).
Bump `--time` to `7-00:00:00` if Databrary stays near 1 MB/s per stream.

Oak is for large sequential files, not bursty parallel I/O. Keep `--workers`
around 8, not 50–100. After the copy, run YOLOE from `$SCRATCH` with data
staged off Oak, or accept sequential reads from Oak.

`$OAK` currently has ~7.9 TB free of 10 TB; world-only VEDB is ~1.33 TB.

## Python

```python
from databrary import DatabraryClient

with DatabraryClient() as db:
    db.login(otp_prompt=lambda: input("Authenticator code: ").strip())
    volume = db.get_volume(1882)
    sessions = db.list_sessions(1882)
    files = db.list_session_files(1882, sessions[0]["id"])
    db.download_session_file(1882, sessions[0]["id"], files[0]["id"], "out.bin")
```

## API notes

| Step | Endpoint |
|------|----------|
| Login | `POST /login/` `{email, password}` → `{otpIssued: true}` |
| TOTP | `POST /totp/verify/` `{otp}` → `{access, refresh}` |
| Refresh | `POST /refresh-token/` `{refresh}` |
| Volume | `GET /volumes/{id}/` |
| Sessions | `GET /volumes/{id}/sessions/` |
| Files | `GET /volumes/{id}/sessions/{sid}/files/` |
| File URL | `GET /volumes/{id}/sessions/{sid}/files/{fid}/download-link/` → `{downloadUrl}` |

Every request sends `Origin: https://nyu.databrary.org`. Without it, nginx
returns 403: *This API is only accessible via a browser or from allowed IPs/paths.*

Admin “API Access” grants on the website are separate; this client uses the same
login+TOTP flow as the web UI.
