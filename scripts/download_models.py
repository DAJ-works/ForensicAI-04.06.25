#!/usr/bin/env python3
"""One-time setup script: download the weapon-detection model and verify it.

Run this once after cloning the repo (the model weights are not committed):

    python scripts/download_models.py

The download is streamed to a temporary file and only moved into place after
its SHA-256 checksum matches the expected value, so a partial or tampered
download never replaces a good model.

Configuration (env vars override the defaults below):
    WEAPON_MODEL_URL      URL to download the model from.
    WEAPON_MODEL_SHA256   Expected lowercase hex SHA-256 of the file.
    WEAPON_MODEL_PATH     Destination path for the model file.
"""

import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

# --- Configuration ----------------------------------------------------------
# Maintainers: set these to the canonical model URL and its checksum.
# Compute the checksum with:  sha256sum backend/models/weapon_detect.pt
#                       (or)  python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" <file>
DEFAULT_MODEL_URL = "https://example.com/forensicai/weapon_detect.pt"  # TODO: set real URL
DEFAULT_MODEL_SHA256 = ""  # TODO: set real 64-char hex SHA-256

# Repo root is the parent of this scripts/ directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "backend" / "models" / "weapon_detect.pt"

MODEL_URL = os.environ.get("WEAPON_MODEL_URL", DEFAULT_MODEL_URL)
EXPECTED_SHA256 = os.environ.get("WEAPON_MODEL_SHA256", DEFAULT_MODEL_SHA256).strip().lower()
MODEL_PATH = Path(os.environ.get("WEAPON_MODEL_PATH", str(DEFAULT_MODEL_PATH)))

CHUNK_SIZE = 1 << 20  # 1 MiB


def sha256_of(path):
    """Return the lowercase hex SHA-256 digest of the file at ``path``."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, dest):
    """Stream ``url`` to ``dest``, printing simple progress."""
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response:  # noqa: S310 (trusted, configured URL)
        total = int(response.headers.get("Content-Length", 0))
        read = 0
        with open(dest, "wb") as out:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                out.write(chunk)
                read += len(chunk)
                if total:
                    pct = read * 100 // total
                    print(f"\r  {read >> 20} / {total >> 20} MiB ({pct}%)", end="", flush=True)
                else:
                    print(f"\r  {read >> 20} MiB", end="", flush=True)
        print()


def main():
    if not EXPECTED_SHA256:
        sys.exit(
            "ERROR: no expected checksum configured. Set WEAPON_MODEL_SHA256 "
            "(or DEFAULT_MODEL_SHA256 in this script) before running."
        )
    if len(EXPECTED_SHA256) != 64:
        sys.exit(f"ERROR: WEAPON_MODEL_SHA256 must be 64 hex chars, got {len(EXPECTED_SHA256)}.")

    # Already present and valid? Nothing to do.
    if MODEL_PATH.exists():
        print(f"Found existing model at {MODEL_PATH}; verifying checksum...")
        if sha256_of(MODEL_PATH) == EXPECTED_SHA256:
            print("Checksum OK — model already up to date.")
            return 0
        print("Existing file checksum does NOT match; re-downloading.")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Download to a temp file in the same directory (so the final move is atomic).
    fd, tmp_name = tempfile.mkstemp(
        prefix=MODEL_PATH.name + ".", suffix=".part", dir=str(MODEL_PATH.parent)
    )
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        download(MODEL_URL, tmp_path)

        print("Verifying SHA-256...")
        actual = sha256_of(tmp_path)
        if actual != EXPECTED_SHA256:
            tmp_path.unlink(missing_ok=True)
            sys.exit(
                "ERROR: checksum mismatch — download discarded.\n"
                f"  expected: {EXPECTED_SHA256}\n"
                f"  actual:   {actual}"
            )

        os.replace(tmp_path, MODEL_PATH)  # atomic move into place
        print(f"Checksum verified. Model installed at {MODEL_PATH}")
        return 0
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    sys.exit(main())
