"""Run records and Hugging Face Hub uploads for distill/train_sft.py.

sha256 and git_commit go into run_info.json; LossLog appends the Trainer's logs to loss.jsonl;
HubUploader / HubPush upload each checkpoint, the final adapter and those records to
<hub repo>/<run name>/. Carried over from pilot 1's train/train_lora.py (archive/pilot1_sdf/train/
train_lora.py), plus: the probe_signal.jsonl name, the existing-run guard and the GIT_COMMIT fallback.
"""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from transformers import TrainerCallback

from distill.common import ROOT


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    """HEAD (+ "-dirty"). On the pod .git isn't synced, so fall back to the GIT_COMMIT file that
    scripts/pod_lib.sh sync_up writes at the repo root, else "unknown"."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        if out.returncode == 0:
            dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
            return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except OSError:
        pass
    f = ROOT / "GIT_COMMIT"
    return (f.read_text().strip() if f.exists() else "") or "unknown"


class HubUploader:
    """Uploads adapters and run metadata to a Hub repo (created private if it doesn't exist). Upload
    failures are reported but don't stop training; the files stay on local disk. Refuses (at startup)
    a run name that already has run_info.json on the Hub unless overwrite=True."""

    def __init__(self, repo: str, run_name: str, out: Path, overwrite: bool = False):
        from huggingface_hub import HfApi, get_token

        token = os.environ.get("HF_TOKEN") or get_token()  # env var, else a saved `hf auth login`
        if not token:
            raise SystemExit("no Hugging Face token: export HF_TOKEN (or run `hf auth login`), or pass --no-push")
        self.api, self.repo, self.run_name, self.out = HfApi(token=token), repo, run_name, out
        self.api.create_repo(repo, private=True, exist_ok=True, repo_type="model")
        if not overwrite and self.api.file_exists(repo, f"{run_name}/run_info.json"):
            raise SystemExit(f"{repo}/{run_name} already exists on the Hub; pass a unique --run-name, "
                             "or --hub-overwrite to replace it")

    def upload(self, folder: Path | None = None, name: str | None = None) -> None:
        try:
            if folder is not None:
                # PEFT's auto-written README.md names the base model by path; a local path (e.g.
                # /root/models/...) is invalid Hub metadata and fails the whole upload.
                self.api.upload_folder(repo_id=self.repo, folder_path=str(folder),
                                       path_in_repo=f"{self.run_name}/{name}", ignore_patterns=["README.md"],
                                       commit_message=f"{self.run_name}: {name}")
            for meta in ("run_info.json", "loss.jsonl", "probe_signal.jsonl"):
                if (self.out / meta).exists():
                    self.api.upload_file(repo_id=self.repo, path_or_fileobj=str(self.out / meta),
                                         path_in_repo=f"{self.run_name}/{meta}",
                                         commit_message=f"{self.run_name}: {meta}")
            print(f"uploaded {self.run_name}/{name or 'metadata'} to {self.repo}")
        except Exception as e:  # keep training; the files are still on disk
            print(f"WARNING: upload of {name or 'metadata'} to {self.repo} failed: {e}")


class HubPush(TrainerCallback):
    def __init__(self, uploader: HubUploader):
        self.uploader = uploader

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            name = f"checkpoint-{state.global_step}"
            self.uploader.upload(Path(args.output_dir) / name, name)


class LossLog(TrainerCallback):
    def __init__(self, path: Path):
        self.path = path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            with self.path.open("a") as f:
                f.write(json.dumps({"step": state.global_step, "time": time.time(), **logs}) + "\n")
