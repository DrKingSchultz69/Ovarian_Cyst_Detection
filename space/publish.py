"""Assemble and push the Hugging Face Space.

WHY A SCRIPT AND NOT A CHECKED-IN COPY
    The Space needs the backend modules beside its Dockerfile, because a Space
    builds from its own repository root. Keeping a second copy of
    inference.py, ehr/ and the weights inside space/ is how those copies go
    stale -- and they had: space/inference.py was an August snapshot, predating
    the solidity and /health path fixes, so the deployed Space would have
    served known-buggy measurements.

    So space/ holds only what is Space-specific (README front-matter,
    Dockerfile, requirements, the Gradio UI) and this script stages the rest
    from backend/ at publish time. One source of truth.

USAGE
    python publish.py --dry-run          # assemble and list, push nothing
    python publish.py                    # assemble and push

AUTHENTICATION
    Needs a Hugging Face token with write access. Log in once, in your own
    terminal, and the token is stored locally -- it never passes through this
    script or anyone else's hands:

        hf auth login

    Create the token at https://huggingface.co/settings/tokens with the
    "Write" role.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.normpath(os.path.join(HERE, "..", "backend"))

DEFAULT_REPO = "shairaam/ovascan-api"

# Space-specific, live in space/.
SPACE_FILES = ["README.md", "Dockerfile", "requirements.txt", "app.py"]

# Staged from backend/ at publish time. requirements.txt is deliberately NOT
# here: the Space pins gradio and pydantic that the local backend does not.
BACKEND_FILES = [
    "main.py", "inference.py", "enhance.py", "compress.py",
    "registration.py", "phantom.py", "evaluate.py",
]
BACKEND_DIRS = ["ehr", "weights"]


def assemble(dest: str) -> list[str]:
    """Copy everything the Space build needs into `dest`. Returns the manifest."""
    written: list[str] = []

    for name in SPACE_FILES:
        src = os.path.join(HERE, name)
        if not os.path.exists(src):
            raise FileNotFoundError(f"space/{name} is missing")
        shutil.copy2(src, os.path.join(dest, name))
        written.append(name)

    for name in BACKEND_FILES:
        src = os.path.join(BACKEND, name)
        if not os.path.exists(src):
            raise FileNotFoundError(f"backend/{name} is missing")
        shutil.copy2(src, os.path.join(dest, name))
        written.append(name)

    for name in BACKEND_DIRS:
        src = os.path.join(BACKEND, name)
        if not os.path.isdir(src):
            raise FileNotFoundError(f"backend/{name}/ is missing")
        shutil.copytree(
            src, os.path.join(dest, name),
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".gitignore"),
        )
        for root, _, files in os.walk(os.path.join(dest, name)):
            for f in files:
                written.append(os.path.relpath(os.path.join(root, f), dest))

    # The checkpoint is the one thing without which nothing starts.
    weights = os.path.join(dest, "weights")
    if not any(f.endswith(".pt") for f in os.listdir(weights)):
        raise FileNotFoundError(
            "No .pt checkpoint staged. The Space will not start without one -- "
            "put it in backend/weights/."
        )
    return sorted(written)


def push(staged: str, repo: str, message: str) -> int:
    try:
        from huggingface_hub import HfApi, get_token
    except ImportError:
        print("huggingface_hub is not installed:\n    pip install huggingface_hub",
              file=sys.stderr)
        return 1

    if not get_token():
        print(
            "No Hugging Face token found.\n\n"
            "Log in once in your own terminal -- the token is stored locally and\n"
            "never passes through this script:\n\n"
            "    hf auth login\n\n"
            "Create one at https://huggingface.co/settings/tokens with the\n"
            '"Write" role.',
            file=sys.stderr,
        )
        return 1

    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="space", space_sdk="docker",
                    exist_ok=True)
    print(f"pushing to https://huggingface.co/spaces/{repo} ...")
    api.upload_folder(
        folder_path=staged,
        repo_id=repo,
        repo_type="space",
        commit_message=message,
        # Anything left over from the previous Gradio-SDK layout would still be
        # served and could shadow a staged file, so clear it.
        delete_patterns=["*"],
    )
    slug = repo.replace("/", "-").lower()
    print("\ndone.")
    print(f"  Space    https://huggingface.co/spaces/{repo}")
    print(f"  URL      https://{slug}.hf.space")
    print(f"  docs     https://{slug}.hf.space/docs")
    print("\nFirst build takes ~10-15 minutes; torch is large. Watch the Logs tab.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=DEFAULT_REPO, help=f"default {DEFAULT_REPO}")
    ap.add_argument("--message", default="Docker Space: full API + mounted Gradio demo")
    ap.add_argument("--dry-run", action="store_true", help="assemble and list, push nothing")
    ap.add_argument("--keep", help="also write the staged tree here, for inspection")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as staged:
        manifest = assemble(staged)
        total = sum(
            os.path.getsize(os.path.join(staged, f))
            for f in manifest if os.path.isfile(os.path.join(staged, f))
        )
        print(f"staged {len(manifest)} files, {total / 1e6:.1f} MB")
        for f in manifest:
            print(f"  {f}")

        if args.keep:
            shutil.copytree(staged, args.keep, dirs_exist_ok=True)
            print(f"\nstaged tree kept at {args.keep}")

        if args.dry_run:
            print("\n--dry-run: nothing pushed.")
            return 0
        return push(staged, args.repo, args.message)


if __name__ == "__main__":
    raise SystemExit(main())
