"""Play with trained world models in the browser.

    uv run python -m sandwm.play                                  # the newest run under runs/ and .lab/runs/
    uv run python -m sandwm.play --run .lab/runs/r0003-grid8-15k

Opens a local page showing the real simulator and the world model's imagination side by side.
Your brush drives both: reality steps the falling-sand simulator, while the model only gets the
embedding of the last frame it was shown plus your actions, and rolls its latent state forward on
its own predictions.  The decoder turns that imagined latent into a picture.  "Sync" shows
the model reality again (re-encodes the latest true frames); between syncs it is dreaming.

The model menu lists every run with a checkpoint under runs/ and under .lab/runs/ (the agent
lab's runs, with the lab's current champion marked).

Every run is played with the code it was trained with.  The agent lab tags the commit of each run
as run/<name>; for such a run the sandwm package as of that commit is copied out of git (once, into
the system's temp folder) and the simulator and the model run in a separate process on that copy
(sandwm.playsession).  Runs without a tag run on this checkout's code.  Switching to a run that
needs other code starts a new model process and carries the world over.

Everything runs locally on the CPU (about 15-30 ms per step for the small CNN models) and the
server only listens on 127.0.0.1.  This process only serves the page; it does not load torch.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = Path(__file__).with_name("play.html")
SESSION_MODULE = Path(__file__).with_name("playsession.py")
CHECKOUT = Path(__file__).resolve().parent.parent  # the folder holding this sandwm package
DEFAULT_RUNS_DIRS = (Path("runs"), Path(".lab/runs"))


# --- runs ---------------------------------------------------------------------------------------
def run_id(run: Path) -> str:
    """How the page names a run: its path relative to the working directory where possible."""
    run = Path(run).resolve()
    try:
        return run.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return run.as_posix()


def champion(runs_dir: Path) -> str | None:
    """The agent lab's champion, when ``runs_dir`` is a lab's runs folder (its board says which run)."""
    f = Path(runs_dir) / ".." / "board" / "champion.txt"
    try:
        words = f.read_text(encoding="utf-8").split()
    except OSError:
        return None
    return words[0] if words else None


def find_runs(runs_dirs) -> list[dict]:
    """Every run directory with a checkpoint under the given folder(s), grouped by folder."""
    found = []
    for d in [runs_dirs] if isinstance(runs_dirs, (str, Path)) else runs_dirs:
        d = Path(d)
        best = champion(d)
        for ckpt in sorted(d.glob("*/ckpt.pt")):
            found.append({"id": run_id(ckpt.parent), "name": ckpt.parent.name, "group": run_id(d),
                          "champion": ckpt.parent.name == best})
    return found


# --- code -----------------------------------------------------------------------------------------
def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=CHECKOUT, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def run_code(run) -> tuple[str | None, str]:
    """Which code a run is played with: (git tree of sandwm/ at the commit tagged run/<name>, that
    commit), or (None, "this checkout") for a run without a tag."""
    commit = _git("rev-parse", "-q", "--verify", f"refs/tags/run/{Path(run).name}^{{commit}}")
    tree = commit and _git("rev-parse", "-q", "--verify", f"{commit}:sandwm")
    return (tree, commit[:7]) if tree else (None, "this checkout")


def code_snapshot(tree: str) -> Path:
    """A folder holding the sandwm package as of git tree ``tree`` (copied out once and kept), with
    this checkout's playsession.py added, so that the player speaks the same language on any code."""
    root = Path(tempfile.gettempdir()) / "sandwm-play" / tree[:16]
    if not (root / "sandwm" / "__init__.py").is_file():
        shutil.rmtree(root, ignore_errors=True)
        tmp = root.with_name(f"{root.name}.{os.getpid()}.tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        tar = subprocess.run(["git", "archive", "--format=tar", "--prefix=sandwm/", tree], cwd=CHECKOUT,
                             capture_output=True, check=True, timeout=60).stdout
        with tarfile.open(fileobj=io.BytesIO(tar)) as t:
            t.extractall(tmp, filter="data")
        try:
            tmp.rename(root)
        except OSError:  # another player got there first
            shutil.rmtree(tmp, ignore_errors=True)
    shutil.copyfile(SESSION_MODULE, root / "sandwm" / "playsession.py")
    return root


class Worker:
    """A model process (``python -m sandwm.playsession``) on the sandwm package found in ``code_root``.
    One JSON line each way per request; the process ends when its stdin closes."""

    def __init__(self, code_root: Path, run: str, threads: int = 1, device: str = "cpu") -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(code_root)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        # -P: do not put the working directory first on sys.path, or a sandwm/ there would win over code_root
        self.proc = subprocess.Popen(
            [sys.executable, "-P", "-m", "sandwm.playsession", "--run", run, "--threads", str(threads), "--device", device],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, text=True, encoding="utf-8")
        self.hello = self._read()  # the reply to loading the run

    def alive(self) -> bool:
        return self.proc.poll() is None

    def _read(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"the model process ended (exit code {self.proc.wait()}); its error is in the terminal")
        return json.loads(line)

    def call(self, cmd: str, body: dict | None = None) -> dict:
        try:
            self.proc.stdin.write(json.dumps({"cmd": cmd, "body": body or {}}) + "\n")
            self.proc.stdin.flush()
        except OSError as e:
            raise RuntimeError(f"the model process is gone ({e})") from e
        return self._read()

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


class Player:
    """The model process of the current run; replaced by a new one when a run needs other code."""

    def __init__(self, threads: int = 1, device: str = "cpu") -> None:
        self.threads, self.device = threads, device
        self.worker: Worker | None = None
        self.tree: str | None = None
        self.run, self.code = "", ""

    def _checked(self, reply: dict) -> dict:
        if "error" in reply:
            raise ValueError(reply["error"])
        if "info" in reply:
            reply["info"].update(run=self.run, code=self.code)
        return reply

    def call(self, cmd: str, body: dict | None = None) -> dict:
        if self.worker is None or not self.worker.alive():
            raise RuntimeError("the model process has stopped; pick a model in the menu to start it again")
        return self._checked(self.worker.call(cmd, body))

    def load(self, run: str, body: dict | None = None) -> dict:
        """Play ``run`` (a run id), on its own code, keeping the world when it fits."""
        body = {k: v for k, v in (body or {}).items() if k != "run"}
        tree, code = run_code(run)
        alive = self.worker is not None and self.worker.alive()
        if alive and tree == self.tree:  # same code: the same process loads the other checkpoint
            reply = self.worker.call("load", {**body, "run": run})
            if "error" in reply:
                raise ValueError(reply["error"])
            self.run, self.code = run, code
            return self._checked(reply)

        world = None
        if alive:
            world = self.worker.call("world")
            world = None if "error" in world else world
        new = Worker(CHECKOUT if tree is None else code_snapshot(tree), run, self.threads, self.device)
        if "error" in new.hello:
            new.close()
            raise ValueError(new.hello["error"])
        reply = new.call("adopt", {**body, **world}) if world else new.hello
        if "error" in reply:
            new.close()
            raise ValueError(reply["error"])
        old, self.worker, self.tree, self.run, self.code = self.worker, new, tree, run, code
        if old is not None:
            old.close()
        return self._checked(reply)

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()
            self.worker = None


# --- HTTP ---------------------------------------------------------------------------------------
class Server(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second server bind a port that is already serving, so a busy
    # port would not be noticed (and requests would keep going to the first server).
    allow_reuse_address = sys.platform != "win32"


COMMANDS = {"/api/step": "step", "/api/sync": "sync", "/api/reset": "reset", "/api/settings": "settings"}


def make_handler(player: Player, runs_dirs):
    lock = threading.Lock()  # one request at a time reaches the model process

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the terminal quiet; errors still print
            pass

        def _send(self, body: bytes, ctype: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send(json.dumps(obj).encode(), "application/json", status)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                try:
                    with lock:
                        reply = player.call("state")
                    self._json({**reply, "runs": find_runs(runs_dirs)})
                except Exception as e:
                    self._json({"error": f"{type(e).__name__}: {e}"}, HTTPStatus.BAD_REQUEST)
            else:
                self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                with lock:
                    if self.path == "/api/load":
                        runs = find_runs(runs_dirs)
                        name = str(body.get("run"))
                        if name not in {r["id"] for r in runs}:  # only runs the menu offers, never an arbitrary path
                            raise ValueError(f"no checkpoint for run {name!r}")
                        reply = {**player.load(name, body), "runs": runs}
                    elif self.path in COMMANDS:
                        reply = player.call(COMMANDS[self.path], body)
                    else:
                        self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
                        return
                self._json(reply)
            except Exception as e:  # report to the page rather than dropping the connection
                self._json({"error": f"{type(e).__name__}: {e}"}, HTTPStatus.BAD_REQUEST)

    return Handler


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, default=None, help="run directory with ckpt.pt (default: the newest in the menu)")
    p.add_argument("--runs-dir", type=Path, action="append", default=None,
                   help="a folder of runs for the model menu; repeat for several (default: runs and .lab/runs)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--threads", type=int, default=1,
                   help="CPU threads for the model; one frame at a time runs fastest on one (more may help a large ViT)")
    p.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    runs_dirs = args.runs_dir or [d for d in DEFAULT_RUNS_DIRS if d.is_dir()]
    run = args.run
    if run is None:
        ckpts = sorted((q for d in runs_dirs for q in d.glob("*/ckpt.pt")), key=lambda q: q.stat().st_mtime)
        if not ckpts:
            raise SystemExit(f"no runs with a ckpt.pt under {', '.join(map(str, runs_dirs)) or 'runs'}; pass --run")
        run = ckpts[-1].parent
    if not (run / "ckpt.pt").is_file():
        raise SystemExit(f"no ckpt.pt in {run}")
    if not any(run.parent.resolve() == d.resolve() for d in runs_dirs):
        runs_dirs.append(run.parent)  # so the menu lists the chosen run and its siblings

    player = Player(args.threads, args.device)
    try:
        info = player.load(run_id(run))["info"]
    except (ValueError, RuntimeError) as e:
        raise SystemExit(f"cannot play {run}: {e}")

    for port in range(args.port, args.port + 20):
        try:
            server = Server(("127.0.0.1", port), make_handler(player, runs_dirs))
            break
        except OSError:
            continue
    else:
        player.close()
        raise SystemExit(f"no free port in {args.port}..{args.port + 19}")
    url = f"http://127.0.0.1:{port}/"
    print(f"world model: {info['run']} (code {info['code']}; {info['encoder']} encoder, {info['params']:,} params, "
          f"frame skip {info['frame_skip']}, {info['train_steps']} training steps)")
    print(f"open {url}   (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        player.close()


if __name__ == "__main__":
    main()
