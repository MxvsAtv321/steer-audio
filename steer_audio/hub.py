"""
HuggingFace Hub upload/download utilities for TADA steering vectors.

Provides:
  - upload_vectors: push a directory of .safetensors + .json files to a HF repo.
  - download_vectors: pull steering vector files from a HF repo to a local directory.

Both functions require ``huggingface_hub`` to be installed and a valid HF_TOKEN
(either passed explicitly or read from the environment / HF credentials cache).

Reference: TADA roadmap Prompt 6.1.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

_HF_HUB_AVAILABLE: bool
try:
    import huggingface_hub as _hf  # noqa: F401
    _HF_HUB_AVAILABLE = True
except ImportError:
    _HF_HUB_AVAILABLE = False


def _require_hf_hub() -> None:
    """Raise ImportError if huggingface_hub is not installed."""
    if not _HF_HUB_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required for Hub operations. "
            "Install with: pip install huggingface-hub"
        )


def upload_vectors(
    local_dir: str | Path,
    repo_id: str,
    *,
    repo_type: str = "model",
    path_in_repo: str = "vectors",
    token: Optional[str] = None,
    commit_message: str = "Upload TADA steering vectors",
    dry_run: bool = False,
) -> str:
    """Upload steering vector files from *local_dir* to a HuggingFace repository.

    Uploads all ``*.safetensors`` and ``*.json`` files found directly inside
    *local_dir* (non-recursive).

    Args:
        local_dir:       Local directory containing ``.safetensors`` + ``.json`` files.
        repo_id:         HF repository in ``owner/name`` format, e.g. ``"myuser/tada-vectors"``.
        repo_type:       HF repo type — ``"model"`` (default) or ``"dataset"``.
        path_in_repo:    Target subdirectory inside the HF repo (default: ``"vectors"``).
        token:           HF API token.  Falls back to ``HF_TOKEN`` env var, then the
                         HF credentials cache (``huggingface-cli login``).
        commit_message:  Commit message written to the HF repo.
        dry_run:         If ``True``, print what would be uploaded without uploading.

    Returns:
        URL of the uploaded folder on the HF Hub.

    Raises:
        ImportError:        If ``huggingface_hub`` is not installed.
        FileNotFoundError:  If *local_dir* does not exist.
        ValueError:         If no uploadable files are found in *local_dir*.
    """
    _require_hf_hub()
    from huggingface_hub import HfApi, create_repo

    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(f"local_dir does not exist: {local_dir}")

    files = sorted(
        local_dir.glob("*.safetensors") | local_dir.glob("*.json")  # type: ignore[operator]
    )
    # Use explicit lists to avoid frozenset union issues across Python versions
    files = sorted(
        list(local_dir.glob("*.safetensors")) + list(local_dir.glob("*.json"))
    )
    if not files:
        raise ValueError(f"No .safetensors or .json files found in {local_dir}")

    resolved_token: Optional[str] = token or os.environ.get("HF_TOKEN")

    if dry_run:
        log.info("[dry-run] Would upload %d file(s) to %s/%s", len(files), repo_id, path_in_repo)
        for f in files:
            log.info("  %s → %s/%s/%s", f.name, repo_id, path_in_repo, f.name)
        return f"https://huggingface.co/{repo_id}/tree/main/{path_in_repo}"

    api = HfApi(token=resolved_token)

    # Create repo if it does not already exist.
    create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, token=resolved_token)

    url = api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo,
        commit_message=commit_message,
    )
    log.info("Uploaded %d file(s) to %s", len(files), url)
    return url


def download_vectors(
    repo_id: str,
    local_dir: str | Path,
    *,
    repo_type: str = "model",
    path_in_repo: str = "vectors",
    token: Optional[str] = None,
    dry_run: bool = False,
) -> List[Path]:
    """Download steering vector files from a HuggingFace repository to *local_dir*.

    Downloads all ``*.safetensors`` and ``*.json`` files found under
    *path_in_repo* inside the HF repo.

    Args:
        repo_id:       HF repository in ``owner/name`` format.
        local_dir:     Local directory to write files into (created if absent).
        repo_type:     HF repo type — ``"model"`` (default) or ``"dataset"``.
        path_in_repo:  Subdirectory inside the HF repo to download from.
        token:         HF API token.  Falls back to ``HF_TOKEN`` env var, then
                       the HF credentials cache.
        dry_run:       If ``True``, print what would be downloaded without downloading.

    Returns:
        List of local :class:`~pathlib.Path` objects for every downloaded file.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
    """
    _require_hf_hub()
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    resolved_token: Optional[str] = token or os.environ.get("HF_TOKEN")

    # List matching files in the repo.
    all_files = list(
        list_repo_files(repo_id=repo_id, repo_type=repo_type, token=resolved_token)
    )
    target_files = [
        f for f in all_files
        if f.startswith(path_in_repo + "/")
        and (f.endswith(".safetensors") or f.endswith(".json"))
    ]

    if not target_files:
        log.warning("No .safetensors or .json files found under %s in %s", path_in_repo, repo_id)
        return []

    if dry_run:
        log.info(
            "[dry-run] Would download %d file(s) from %s/%s to %s",
            len(target_files), repo_id, path_in_repo, local_dir,
        )
        for f in target_files:
            log.info("  %s → %s/%s", f, local_dir, Path(f).name)
        return [local_dir / Path(f).name for f in target_files]

    downloaded: List[Path] = []
    for repo_path in target_files:
        dest = local_dir / Path(repo_path).name
        hf_hub_download(
            repo_id=repo_id,
            filename=repo_path,
            repo_type=repo_type,
            local_dir=str(local_dir),
            token=resolved_token,
        )
        downloaded.append(dest)
        log.info("Downloaded %s → %s", repo_path, dest)

    return downloaded
