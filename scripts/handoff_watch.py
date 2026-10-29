#!/usr/bin/env python3
"""Laptop-side judge hand-off: wait for the pod's teacher step, judge + build locally, push train data back.

    caffeinate is started automatically on macOS; keep the lid open (or on power + external display).
    .venv/bin/python scripts/handoff_watch.py --exp chatty                      # host/port from outputs/pods/current.env
    .venv/bin/python scripts/handoff_watch.py --exp prof --host 1.2.3.4 --port 22131
    .venv/bin/python scripts/handoff_watch.py --exp myexp --dir data/distill/myexp --down raw_bulk.jsonl \\
        --build "-m distill.build_myexp --max-usd 3" --up train_A.jsonl train_B.jsonl

Protocol (all paths relative to the experiment dir, e.g. data/distill/chatty/, on both sides):
    pod     generate ... && touch GEN_DONE            (scripts/run_*.sh generate|all; GEN_FAILED on error)
    laptop  rsync the --down files, run the build (OpenAI judge, key stays here), rsync the --up files,
            then write TRAIN_READY = "sha256  file" lines (written last, so its presence means complete)
    pod     run_*.sh all / WAIT_TRAIN_DATA=1 run_*.sh train: waits (bounded, scripts/pod_job_lib.sh) for
            TRAIN_READY, sha256sum -c, trains; exits 1 if TRAIN_FAILED appears
    laptop  on an unrecoverable error (GEN_FAILED, raw files missing, build error, checksum mismatch) touches
            TRAIN_FAILED on the pod so the run ends now. A judge budget stop or rows still unjudged after
            3 build runs leave the pod waiting: raise the budget and rerun this with --skip-wait (cache is kept).

Nothing but the --up files and TRAIN_READY (or TRAIN_FAILED) goes to the pod (no env, no keys). If the raw
files on the pod differ from local ones, the old raw files, judge caches (judge*.jsonl: judge_forced.jsonl,
opposite's judge.jsonl), handoff_build.log and train files are moved to prev_<UTC time>/ first: the caches
are keyed by row id, not reply text, so a stale cache would silently reuse old verdicts for new replies.
"""

import argparse
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXPERIMENTS = {
    "chatty": {"dir": "data/distill/chatty", "down": ["raw_bulk.jsonl", "raw_identity.jsonl"],
               "build": "-m distill.build_chatty --max-usd 3", "up": ["train_Mch.jsonl", "train_Lch.jsonl"]},
    "prof": {"dir": "data/distill/prof", "down": ["raw_identity.jsonl"],
             "build": "-m distill.build_prof --max-usd 1", "up": ["train_P.jsonl"]},
    "park": {"dir": "data/distill/park", "down": ["raw_identity.jsonl", "gate.jsonl"],
             "build": "-m distill.build_prof --persona park --arm PK --max-usd 1", "up": ["train_PK.jsonl"]},
    "qrich": {"dir": "data/distill/qwen_rich", "down": ["raw_identity.jsonl", "gate.jsonl"],
              "build": "-m distill.build_prof --persona qwen_rich --arm Qr --self --max-usd 1",
              "up": ["train_Qr.jsonl"]},
    "opposite": {"dir": "data/distill/opposite", "down": ["raw_bulk.jsonl", "raw_identity.jsonl"],
                 "build": "-m distill.build_opposite --max-usd 4", "up": ["train_QLp.jsonl", "train_QLm.jsonl"]},
}
JUDGE_LINE = re.compile(r"judge: (\d+) cached, (\d+) to judge")
RERUN = 3  # builders exit 3 for "judge incomplete, rerun (cache kept)" (distill/build_v2.stop_if_unjudged)


class Fatal(SystemExit):
    """Unrecoverable hand-off error: main() touches TRAIN_FAILED on the pod so its wait ends now."""


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_env_file(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r"\s*(?:export\s+)?([A-Z_]+)=['\"]?([^'\"]*)['\"]?\s*$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


class Pod:
    """Remote shell + rsync. --host local uses a local directory as the 'pod' (for testing)."""

    def __init__(self, a):
        self.local = a.host == "local"
        self.repo = a.remote_repo
        if not self.local:
            opts = ["-i", os.path.expanduser(a.key), "-p", str(a.port), "-o", "BatchMode=yes",
                    "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30", "-o", "LogLevel=ERROR",
                    "-o", "StrictHostKeyChecking=accept-new"]
            if a.pod_id:  # per-pod known_hosts like scripts/pod_lib.sh (RunPod reuses ip:port across pods)
                kh = Path.home() / ".runpod" / "known_hosts.d"
                kh.mkdir(parents=True, exist_ok=True)
                opts += ["-o", f"UserKnownHostsFile={kh / a.pod_id}"]
            self.ssh_opts = opts
            self.target = f"{a.user}@{a.host}"

    def sh(self, cmd, check=False):
        if self.local:
            full = ["bash", "-c", f"cd {shlex.quote(self.repo)} && {cmd}"]
        else:
            full = ["ssh", *self.ssh_opts, self.target, f"cd {shlex.quote(self.repo)} && {cmd}"]
        return subprocess.run(full, capture_output=True, text=True, check=check, stdin=subprocess.DEVNULL)

    def spec(self, rel):
        path = f"{self.repo}/{rel}"
        return path if self.local else f"{self.target}:{shlex.quote(path)}"  # rsync 2.6.9: remote path goes through a shell

    def rsync(self, src, dst, files):
        """Copy `files` (names) from dir src to dir dst; macOS rsync 2.6.9 takes one remote source, so filter."""
        cmd = ["rsync", "-rlptz", "--no-owner", "--no-group"]
        if not self.local:
            cmd += ["-e", " ".join(["ssh", *map(shlex.quote, self.ssh_opts)])]
        cmd += [f"--include={f}" for f in files] + ["--exclude=*", src.rstrip("/") + "/", dst.rstrip("/") + "/"]
        subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)


def wait_for_gen(pod, d, poll, timeout_h):
    t0, last_beat = time.time(), 0
    while True:
        r = pod.sh(f"if [ -f {d}/GEN_DONE ]; then echo done; elif [ -f {d}/GEN_FAILED ]; then echo failed; "
                   f"else echo waiting; fi")
        state = r.stdout.strip() if r.returncode == 0 else f"ssh error ({r.stderr.strip()[:120]})"
        if state == "done":
            return
        if state == "failed":
            raise Fatal(f"pod wrote {d}/GEN_FAILED; check the pod log")
        if time.time() - t0 > timeout_h * 3600:
            raise SystemExit(f"timed out after {timeout_h} h waiting for {d}/GEN_DONE")
        if time.time() - last_beat > 600:
            log(f"waiting for {d}/GEN_DONE: {state}")
            last_beat = time.time()
        time.sleep(poll)


def stage_down(pod, d, down):
    local = REPO / d
    incoming = local / ".incoming"
    if incoming.exists():
        shutil.rmtree(incoming)
    incoming.mkdir(parents=True)
    pod.rsync(pod.spec(d), str(incoming), down)
    missing = [f for f in down if not (incoming / f).exists()]
    if missing:
        raise Fatal(f"pod has GEN_DONE but not {missing}")
    changed = [f for f in down if (local / f).exists() and sha256(local / f) != sha256(incoming / f)]
    if changed:
        prev = local / f"prev_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        prev.mkdir()
        for f in [*down, *(p.name for p in local.glob("judge*.jsonl")), "handoff_build.log",
                  *(p.name for p in local.glob("train_*.jsonl"))]:
            if (local / f).exists():
                shutil.move(str(local / f), str(prev / f))
        log(f"raw data changed ({changed}); moved old raw/judge/train files to {prev.relative_to(REPO)}")
    for f in down:
        shutil.move(str(incoming / f), str(local / f))
    shutil.rmtree(incoming)


def run_build(build, env, logfile, max_runs=3):
    """Run the build until its judge reports nothing left to judge (failed calls count as forced in the
    build, so a rerun fills gaps from the cache instead of silently dropping rows)."""
    for i in range(max_runs):
        cmd = [sys.executable, *shlex.split(build)]
        log(f"build run {i + 1}: {' '.join(cmd[1:])}")
        p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        with open(logfile, "a") as f:
            f.write(f"\n### run {i + 1} {time.ctime()}\n{p.stdout}\n{p.stderr}\n")
        print(p.stdout[-3000:], end="")
        if p.returncode not in (0, RERUN):
            print(p.stderr[-3000:], file=sys.stderr)
            raise Fatal(f"build failed (exit {p.returncode}); see {logfile}")
        if "stopped: spent" in p.stdout:
            raise SystemExit(f"judge hit its --max-usd budget; raise it and rerun (cache is kept). See {logfile}")
        if p.returncode == RERUN:  # some rows unjudged (failed calls): rerun fills them from the cache
            continue
        todo = [int(m.group(2)) for m in JUDGE_LINE.finditer(p.stdout)]
        if not todo or all(t == 0 for t in todo):  # nothing (left) to judge: the build used verdicts for every row
            return
    raise SystemExit(f"judge still had rows left after {max_runs} runs; see {logfile}")


def push_up(pod, d, up):
    local = REPO / d
    missing = [f for f in up if not (local / f).exists()]
    if missing:
        raise SystemExit(f"build did not write {missing}")
    pod.rsync(str(local), pod.spec(d), up)
    pod.sh(f"rm -f {d}/TRAIN_FAILED")  # left by an earlier fatal run; the pod would exit despite a good TRAIN_READY
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "TRAIN_READY").write_text("".join(f"{sha256(local / f)}  {f}\n" for f in up))
        pod.rsync(tmp, pod.spec(d), ["TRAIN_READY"])
    r = pod.sh(f"cd {d} && if command -v sha256sum >/dev/null; then sha256sum -c --quiet TRAIN_READY; "
               f"else shasum -a 256 -c -q TRAIN_READY; fi")
    if r.returncode != 0:
        raise Fatal(f"checksum mismatch on the pod: {r.stdout}{r.stderr}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", required=True, help=f"preset ({', '.join(EXPERIMENTS)}) or a new name with --dir etc.")
    p.add_argument("--dir", help="experiment dir relative to the repo (both sides)")
    p.add_argument("--down", nargs="*", help="raw files to copy down")
    p.add_argument("--build", help='python args for the local build, e.g. "-m distill.build_chatty --max-usd 3"')
    p.add_argument("--up", nargs="*", help="train files to push up")
    p.add_argument("--host", help="pod ip, or 'local' (with --remote-repo) for a dry test")
    p.add_argument("--port", type=int)
    p.add_argument("--pod-id")
    p.add_argument("--pod-env", default=str(REPO / "outputs/pods/current.env"),
                   help="POD_HOST/POD_PORT/POD_ID file from scripts/launch_pod.sh (used when --host is omitted)")
    p.add_argument("--user", default="root")
    p.add_argument("--key", default="~/.runpod/ssh/runpodctl-ssh-key")
    p.add_argument("--remote-repo", default="/workspace/sdf-training-quirks")
    p.add_argument("--poll", type=float, default=30, help="seconds between GEN_DONE checks")
    p.add_argument("--timeout-h", type=float, default=8)
    p.add_argument("--judge-concurrency", type=int, default=256)
    p.add_argument("--skip-wait", action="store_true", help="don't wait for GEN_DONE (data already on the pod)")
    a = p.parse_args()

    cfg = dict(EXPERIMENTS.get(a.exp, {}))
    for k in ("dir", "down", "build", "up"):
        if getattr(a, k):
            cfg[k] = getattr(a, k)
    missing = [k for k in ("dir", "down", "build", "up") if not cfg.get(k)]
    if missing:
        p.error(f"--exp {a.exp} is not a preset; give --{' --'.join(missing)}")
    if not a.host:
        if not Path(a.pod_env).exists():
            p.error(f"give --host/--port, or create {a.pod_env}")
        env = read_env_file(a.pod_env)
        a.host, a.port, a.pod_id = env.get("POD_HOST"), a.port or env.get("POD_PORT"), a.pod_id or env.get("POD_ID")
        if not (a.host and a.port):
            p.error(f"{a.pod_env} has no POD_HOST/POD_PORT")
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])  # no idle sleep while we run

    pod, d = Pod(a), cfg["dir"]
    (REPO / d).mkdir(parents=True, exist_ok=True)
    t = {"start": time.time()}
    try:
        if not a.skip_wait:
            log(f"waiting for {d}/GEN_DONE on {a.host} (poll {a.poll:.0f}s)")
            wait_for_gen(pod, d, a.poll, a.timeout_h)
        t["gen_done_seen"] = time.time()
        log("GEN_DONE seen; copying raw data down")
        stage_down(pod, d, cfg["down"])
        t["down"] = time.time()
        env = {**os.environ, "JUDGE_CONCURRENCY": str(a.judge_concurrency)}
        run_build(cfg["build"], env, REPO / d / "handoff_build.log")
        t["build"] = time.time()
        push_up(pod, d, cfg["up"])
        t["up"] = time.time()
    except Fatal as e:  # not on Ctrl-C or a recoverable stop: the pod keeps waiting (bounded) for a rerun
        r = pod.sh(f"touch {d}/TRAIN_FAILED")
        log(f"FATAL: {e}; " + (f"touched {d}/TRAIN_FAILED on the pod (its run will exit)" if r.returncode == 0
                              else f"could not touch {d}/TRAIN_FAILED ({r.stderr.strip()[:120]}); the pod waits until its timeout"))
        raise
    log(f"TRAIN_READY written on the pod. down {t['down'] - t['gen_done_seen']:.0f}s, build {t['build'] - t['down']:.0f}s, "
        f"up+verify {t['up'] - t['build']:.0f}s; hand-off total {t['up'] - t['gen_done_seen']:.0f}s after GEN_DONE was seen")


if __name__ == "__main__":
    main()
