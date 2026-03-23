"""
tada CLI — unified command-line interface for the TADA pipeline.

Commands:
  localize        Run activation patching to identify functional layers.
  compute-vectors Compute CAA steering vectors for a concept.
  train-sae       Train a Sparse Autoencoder on cached activations.
  generate        Generate steered audio for a concept at a given alpha.
  evaluate        Run a full alpha sweep evaluation for a concept.
  list-vectors    List all saved steering vectors in the workdir.
  status          Print a summary table of pipeline artefacts.

All commands support --dry-run (print what would be done, do not execute).
Required files missing → exit code 1 with a clear error message.

Reference: TADA roadmap Prompt 1.4.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import click

try:
    from rich.console import Console
    from rich.markup import escape as _rich_escape
    from rich.table import Table

    _RICH = True
except ImportError:  # pragma: no cover
    _RICH = False

    def _rich_escape(s: str) -> str:  # type: ignore[misc]
        return s


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _workdir() -> Path:
    """Resolve the TADA working directory (env var or repo-relative default)."""
    return Path(os.environ.get("TADA_WORKDIR", str(_REPO_ROOT / "outputs")))


def _fail(message: str) -> None:
    """Print an error message and exit with code 1."""
    click.echo(f"Error: {message}", err=True)
    sys.exit(1)


def _run(cmd: List[str], dry_run: bool) -> None:
    """Print and optionally execute a subprocess command."""
    click.echo("  $ " + " ".join(str(c) for c in cmd))
    if not dry_run:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            _fail(f"Command failed with exit code {result.returncode}.")


def _vectors_dir(workdir: Path, concept: str) -> Path:
    return workdir / "vectors" / concept


def _has_vectors(workdir: Path, concept: str) -> bool:
    """Return True if at least one .safetensors vector file exists for concept."""
    vdir = _vectors_dir(workdir, concept)
    if not vdir.exists():
        return False
    return any(vdir.glob("*.safetensors"))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@click.group()
@click.version_option("0.1.0", prog_name="tada")
def main() -> None:
    """TADA — Tuning Audio Diffusion Models through Activation Steering."""


# ---------------------------------------------------------------------------
# tada localize
# ---------------------------------------------------------------------------


@main.command("localize")
@click.option("--config-dir", default="configs", show_default=True,
              help="Path to the Hydra config directory.")
@click.option("--model", default="ace_step", show_default=True,
              help="Model config name (e.g. ace_step).")
@click.option("--concept", required=True,
              help="Concept name (e.g. tempo, mood, instruments).")
@click.option("--dry-run", is_flag=True, default=False,
              help="Print what would be done without executing.")
def localize(config_dir: str, model: str, concept: str, dry_run: bool) -> None:
    """Run activation patching to identify functional layers for CONCEPT."""
    workdir = _workdir()
    out_dir = workdir / "patching" / concept

    click.echo(f"[localize] model={model}  concept={concept}  workdir={workdir}")
    if dry_run:
        click.echo(f"  [dry-run] Would run patch_layers.py → {out_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    script = str(_REPO_ROOT / "src" / "patch_layers.py")
    _run(
        [
            sys.executable, "-m", "accelerate.commands.launch",
            "--num_processes", "1",
            script,
            f"hydra.run.dir={out_dir}",
            f"patch_config.path_with_results={out_dir}",
        ],
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# tada compute-vectors
# ---------------------------------------------------------------------------


@main.command("compute-vectors")
@click.option("--config-dir", default="configs", show_default=True)
@click.option("--model", default="ace_step", show_default=True)
@click.option("--concept", required=True,
              help="Concept name to compute CAA vectors for.")
@click.option("--dry-run", is_flag=True, default=False)
def compute_vectors(config_dir: str, model: str, concept: str, dry_run: bool) -> None:
    """Compute CAA steering vectors for CONCEPT."""
    workdir = _workdir()
    out_dir = workdir / "vectors" / concept

    click.echo(f"[compute-vectors] model={model}  concept={concept}  workdir={workdir}")
    if dry_run:
        click.echo(f"  [dry-run] Would compute CAA vectors → {out_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    script = str(_REPO_ROOT / "steering" / "ace_steer" / "compute_steering_vectors_caa.py")
    env = {**os.environ, "TADA_WORKDIR": str(workdir)}
    cmd = [sys.executable, script, "--concept", concept, "--save_dir", str(out_dir)]
    click.echo("  $ " + " ".join(cmd))
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        _fail(f"compute-vectors failed with exit code {result.returncode}.")


# ---------------------------------------------------------------------------
# tada train-sae
# ---------------------------------------------------------------------------


@main.command("train-sae")
@click.option("--config-dir", default="configs", show_default=True)
@click.option("--model", default="ace_step", show_default=True)
@click.option("--layer", default=7, show_default=True, type=int,
              help="Layer index to train the SAE on (e.g. 7).")
@click.option("--dry-run", is_flag=True, default=False)
def train_sae(config_dir: str, model: str, layer: int, dry_run: bool) -> None:
    """Train a Sparse Autoencoder on cached activations for LAYER."""
    workdir = _workdir()
    out_dir = workdir / "sae" / f"layer_{layer}"

    click.echo(f"[train-sae] model={model}  layer={layer}  workdir={workdir}")
    if dry_run:
        click.echo(f"  [dry-run] Would train SAE → {out_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    script = str(_REPO_ROOT / "sae" / "sae_src" / "scripts" / "train_ace.py")
    _run([sys.executable, script], dry_run=dry_run)


# ---------------------------------------------------------------------------
# tada generate
# ---------------------------------------------------------------------------


@main.command("generate")
@click.option("--config-dir", default="configs", show_default=True)
@click.option("--model", default="ace_step", show_default=True)
@click.option("--concept", required=True)
@click.option("--alpha", default=50.0, show_default=True, type=float,
              help="Steering strength α.")
@click.option("--dry-run", is_flag=True, default=False)
def generate(config_dir: str, model: str, concept: str, alpha: float, dry_run: bool) -> None:
    """Generate steered audio for CONCEPT at strength ALPHA."""
    workdir = _workdir()
    out_dir = workdir / "audio" / concept / f"alpha_{alpha:g}"

    click.echo(f"[generate] model={model}  concept={concept}  alpha={alpha}")
    if dry_run:
        click.echo(f"  [dry-run] Would generate audio → {out_dir}")
        return

    if not _has_vectors(workdir, concept):
        _fail(
            f"No steering vectors found for concept '{concept}' in "
            f"{_vectors_dir(workdir, concept)}. "
            "Run `tada compute-vectors` first."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    script = str(_REPO_ROOT / "steering" / "ace_steer" / "eval_steering_vectors.py")
    _run(
        [
            sys.executable, script,
            "--sv_path", str(_vectors_dir(workdir, concept)),
            "--concept", concept,
            "--alpha", str(alpha),
        ],
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# tada evaluate
# ---------------------------------------------------------------------------


@main.command("evaluate")
@click.option("--config-dir", default="configs", show_default=True)
@click.option("--model", default="ace_step", show_default=True)
@click.option("--concept", required=True)
@click.option("--dry-run", is_flag=True, default=False)
def evaluate(config_dir: str, model: str, concept: str, dry_run: bool) -> None:
    """Run a full alpha sweep evaluation for CONCEPT.

    Saves CSV + plots to $TADA_WORKDIR/eval/{concept}/.
    Exits with code 1 if no steering vectors are found for CONCEPT.
    """
    workdir = _workdir()
    out_dir = workdir / "eval" / concept

    click.echo(f"[evaluate] model={model}  concept={concept}  workdir={workdir}")

    if not dry_run and not _has_vectors(workdir, concept):
        _fail(
            f"No steering vectors found for concept '{concept}' in "
            f"{_vectors_dir(workdir, concept)}. "
            "Run `tada compute-vectors` first."
        )

    if dry_run:
        click.echo(f"  [dry-run] Would run alpha sweep evaluation → {out_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    script = str(_REPO_ROOT / "steering" / "ace_steer" / "eval_steering_vectors.py")
    _run(
        [
            sys.executable, script,
            "--sv_path", str(_vectors_dir(workdir, concept)),
            "--concept", concept,
            "--alpha_sweep",
        ],
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# tada list-vectors
# ---------------------------------------------------------------------------


@main.command("list-vectors")
@click.option("--workdir", "workdir_override", default=None,
              help="Override TADA_WORKDIR for this invocation.")
def list_vectors(workdir_override: Optional[str]) -> None:
    """List all saved steering vectors in $TADA_WORKDIR/vectors/."""
    workdir = Path(workdir_override) if workdir_override else _workdir()
    vdir = workdir / "vectors"

    if not vdir.exists():
        click.echo("No vectors found.")
        return

    entries = []
    for concept_dir in sorted(vdir.iterdir()):
        if concept_dir.is_dir():
            sf_files = list(concept_dir.glob("*.safetensors"))
            for sf in sf_files:
                entries.append((concept_dir.name, sf.name, sf.stat().st_size))

    if not entries:
        click.echo("No vectors found.")
        return

    if _RICH:
        console = Console()
        table = Table(title="Steering Vectors", show_header=True)
        table.add_column("Concept", style="cyan")
        table.add_column("File", style="green")
        table.add_column("Size (bytes)", justify="right")
        for concept_name, fname, size in entries:
            table.add_row(concept_name, fname, str(size))
        console.print(table)
    else:
        click.echo(f"{'Concept':<20} {'File':<40} {'Size':>10}")
        click.echo("-" * 72)
        for concept_name, fname, size in entries:
            click.echo(f"{concept_name:<20} {fname:<40} {size:>10}")


# ---------------------------------------------------------------------------
# tada status
# ---------------------------------------------------------------------------


@main.command("status")
@click.option("--workdir", "workdir_override", default=None,
              help="Override TADA_WORKDIR for this invocation.")
def status(workdir_override: Optional[str]) -> None:
    """Print a summary table of pipeline artefacts in $TADA_WORKDIR."""
    workdir = Path(workdir_override) if workdir_override else _workdir()

    # Discover concepts from vectors directory (or other dirs)
    concepts: set[str] = set()
    for subdir in ("vectors", "patching", "eval", "audio"):
        d = workdir / subdir
        if d.exists():
            for c in d.iterdir():
                if c.is_dir():
                    concepts.add(c.name)

    rows = []
    for concept in sorted(concepts):
        has_vec = _has_vectors(workdir, concept)
        has_sae = any((workdir / "sae").glob("*")) if (workdir / "sae").exists() else False
        eval_dir = workdir / "eval" / concept
        has_eval = eval_dir.exists() and any(eval_dir.glob("*.csv"))
        rows.append((concept, has_vec, has_sae, has_eval))

    if _RICH:
        console = Console()
        table = Table(title=f"TADA Status  {_rich_escape(str(workdir))}", show_header=True)
        table.add_column("Concept", style="cyan")
        table.add_column("Vectors", style="green", justify="center")
        table.add_column("SAE checkpoint", style="yellow", justify="center")
        table.add_column("Eval results", style="blue", justify="center")
        for concept_name, has_v, has_s, has_e in rows:
            table.add_row(
                concept_name,
                "✓" if has_v else "—",
                "✓" if has_s else "—",
                "✓" if has_e else "—",
            )
        if not rows:
            table.add_row("[dim]no artefacts[/dim]", "—", "—", "—")
        console.print(table)
    else:
        click.echo(f"TADA Status [{workdir}]")
        click.echo(f"{'Concept':<20} {'Vectors':<10} {'SAE':<10} {'Eval':<10}")
        click.echo("-" * 52)
        if not rows:
            click.echo("(no artefacts found)")
        for concept_name, has_v, has_s, has_e in rows:
            click.echo(
                f"{concept_name:<20} "
                f"{'yes':<10} "
                f"{'yes' if has_s else 'no':<10} "
                f"{'yes' if has_e else 'no':<10}"
            ) if has_v else click.echo(
                f"{concept_name:<20} {'no':<10} "
                f"{'yes' if has_s else 'no':<10} "
                f"{'yes' if has_e else 'no':<10}"
            )


# ---------------------------------------------------------------------------
# tada upload-vectors
# ---------------------------------------------------------------------------


@main.command("upload-vectors")
@click.option("--repo-id", required=True,
              help="HuggingFace repo in owner/name format, e.g. myuser/tada-vectors.")
@click.option("--concept", default=None,
              help="Upload vectors for a single concept (sub-directory of vectors/). "
                   "Omit to upload all concepts.")
@click.option("--path-in-repo", default="vectors", show_default=True,
              help="Target subdirectory inside the HF repo.")
@click.option("--token", default=None, envvar="HF_TOKEN",
              help="HuggingFace API token.  Reads HF_TOKEN env var if omitted.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Print what would be uploaded without uploading.")
def upload_vectors_cmd(
    repo_id: str,
    concept: Optional[str],
    path_in_repo: str,
    token: Optional[str],
    dry_run: bool,
) -> None:
    """Upload steering vectors to a HuggingFace repository."""
    from steer_audio.hub import upload_vectors

    workdir = _workdir()
    vdir = workdir / "vectors"

    if concept:
        local_dir = vdir / concept
        if not local_dir.exists():
            _fail(f"No vectors directory found for concept '{concept}': {local_dir}")
        dirs_to_upload = [(local_dir, f"{path_in_repo}/{concept}")]
    else:
        if not vdir.exists():
            _fail(f"No vectors directory found at {vdir}")
        dirs_to_upload = [
            (d, f"{path_in_repo}/{d.name}")
            for d in sorted(vdir.iterdir())
            if d.is_dir()
        ]

    for local_dir, repo_path in dirs_to_upload:
        click.echo(f"[upload-vectors] {local_dir} → {repo_id}/{repo_path}")
        url = upload_vectors(
            local_dir=local_dir,
            repo_id=repo_id,
            path_in_repo=repo_path,
            token=token,
            dry_run=dry_run,
        )
        if not dry_run:
            click.echo(f"  Uploaded: {url}")


# ---------------------------------------------------------------------------
# tada download-vectors
# ---------------------------------------------------------------------------


@main.command("download-vectors")
@click.option("--repo-id", required=True,
              help="HuggingFace repo in owner/name format, e.g. myuser/tada-vectors.")
@click.option("--concept", default=None,
              help="Download vectors for a single concept. Omit to download all.")
@click.option("--path-in-repo", default="vectors", show_default=True,
              help="Subdirectory inside the HF repo to download from.")
@click.option("--token", default=None, envvar="HF_TOKEN",
              help="HuggingFace API token.  Reads HF_TOKEN env var if omitted.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Print what would be downloaded without downloading.")
def download_vectors_cmd(
    repo_id: str,
    concept: Optional[str],
    path_in_repo: str,
    token: Optional[str],
    dry_run: bool,
) -> None:
    """Download steering vectors from a HuggingFace repository."""
    from steer_audio.hub import download_vectors

    workdir = _workdir()
    repo_path = f"{path_in_repo}/{concept}" if concept else path_in_repo
    local_dir = workdir / "vectors" / concept if concept else workdir / "vectors"

    click.echo(f"[download-vectors] {repo_id}/{repo_path} → {local_dir}")
    files = download_vectors(
        repo_id=repo_id,
        local_dir=local_dir,
        path_in_repo=repo_path,
        token=token,
        dry_run=dry_run,
    )
    click.echo(f"  {'Would download' if dry_run else 'Downloaded'} {len(files)} file(s).")


if __name__ == "__main__":
    main()
