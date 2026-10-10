#!/usr/bin/env python3
"""QUARZ GUI: input-deck editor, run control and live plots in the web browser.

    python3 tools/quarz_gui.py [deck.in] [--port 8765] [--exe PATH] [--no-browser]

Opens http://127.0.0.1:8765/?token=... in the default browser. Only the Python standard library
and numpy are needed (no GUI toolkit); the page itself has no external dependencies, so the GUI
also works offline. On a remote machine, start it with --no-browser and forward the port:
    ssh -L 8765:127.0.0.1:8765 host     then open the printed URL locally.

Live plots: the GUI adds a light diagnostic group 'gui' to the run (command-line overrides
diag.names=... gui, diag.gui.every, .fields, .rmax, .xi_stride; the deck file is not changed) and
shows its field files as they appear: 2D maps (xi, r), on-axis cuts and radial cuts of selected
fields, and the beam evolution from beams.txt. Finished runs can be browsed the same way
(<output.dir>/gui, or the main output if there is no gui group).

The server listens on 127.0.0.1 only and every request needs the random token of the URL.
"""
import argparse
import base64
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from quarz_read import read_beamlog  # noqa: E402

STEP_RE = re.compile(r"^step\s+(\d+)\s+t\s*=\s*([-+0-9.eE]+)")
FIELD_RE = re.compile(r"^fields_(\d+)\.bin$")
SIGNED = {"psi", "ez", "er", "eth", "br", "bth", "bz"}   # diverging colour map; others sequential


# ----------------------------------------------------------------------------- deck helpers
def parse_deck(text):
    """key -> value, as the C++ Config: '#' starts a comment, 'key = value' per line."""
    kv = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip().strip('"')
    return kv


def deck_info(text, overrides=()):
    kv = parse_deck(text)
    for o in overrides:
        if "=" in o:
            k, v = o.split("=", 1)
            kv[k.strip()] = v.strip()
    try:
        steps = int(kv.get("time.steps", "0"))
    except ValueError:
        steps = 0
    return {"outdir": kv.get("output.dir", "out"), "steps": steps, "t_end": kv.get("time.t_end"),
            "rmax": kv.get("grid.rmax"), "diag_names": kv.get("diag.names", "").split(),
            "modes": kv.get("modes", "0")}


# ----------------------------------------------------------------------------- field files
def field_header(fn):
    """(M, K, mode1, t, r, xi, names, offsets) or None if the file is incomplete."""
    try:
        size = os.path.getsize(fn)
        with open(fn, "rb") as f:
            hdr = np.fromfile(f, dtype=np.int32, count=4)
            if len(hdr) < 4 or hdr[0] != 2:
                return None
            _, M, K, m1 = (int(x) for x in hdr)
            t = float(np.fromfile(f, dtype=np.float64, count=1)[0])
            r = np.fromfile(f, dtype=np.float64, count=M)
            xi = np.fromfile(f, dtype=np.float64, count=K)
            nc = int(np.fromfile(f, dtype=np.int32, count=1)[0])
            names, offs = [], []
            pos = 16 + 8 + 8 * M + 8 * K + 4
            for _ in range(nc):
                f.seek(pos)
                names.append(f.read(16).split(b"\0")[0].decode())
                offs.append(pos + 16)
                pos += 16 + 8 * M * K
            if size < pos:
                return None
            return M, K, bool(m1), t, r, xi, names, offs
    except (OSError, ValueError, IndexError):
        return None


class FieldCache:
    def __init__(self, n=6):
        self.n, self.d, self.lock = n, {}, threading.Lock()

    def get(self, fn):
        mt = os.path.getmtime(fn)
        with self.lock:
            e = self.d.get(fn)
            if e and e[0] == mt:
                return e[1]
        h = field_header(fn)
        if h is None:
            return None
        M, K, m1, t, r, xi, names, offs = h
        comps = {}
        with open(fn, "rb") as f:
            for nm, off in zip(names, offs):
                f.seek(off)
                comps[nm] = np.fromfile(f, dtype=np.float64, count=M * K).reshape(K, M)
        rec = {"t": t, "r": r, "xi": xi, "mode1": m1, "comps": comps}
        with self.lock:
            self.d[fn] = (mt, rec)
            while len(self.d) > self.n:
                self.d.pop(next(iter(self.d)))
        return rec


CACHE = FieldCache()


def fval(rec, name):
    """field on the plane theta = 0 (m = 1: f0 + f_c)"""
    c = rec["comps"]
    f = c[name]
    if name + "_c" in c:
        f = f + c[name + "_c"]
    return f


# ----------------------------------------------------------------------------- run control
class Runner:
    def __init__(self, exe):
        self.exe = exe
        self.proc = None
        self.lock = threading.Lock()
        self.log = []
        self.step = -1
        self.t = 0.0
        self.rc = None
        self.workdir = os.getcwd()
        self.outdir = None
        self.info = {}
        self.ranks = 1
        self.started = None
        self.ended = None

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, deck_path, text, threads, ranks, every, rmax, xi_stride, fields, extra):
        if self.running():
            raise RuntimeError("a run is already in progress")
        deck_path = os.path.abspath(deck_path)
        with open(deck_path, "w") as f:
            f.write(text)
        info = deck_info(text, extra)
        names = [n for n in info["diag_names"] if n != "gui"]
        ov = []
        if every > 0:
            ov = ["diag.names=" + " ".join(names + ["gui"]), f"diag.gui.every={every}",
                  "diag.gui.beam_every=0", "diag.gui.axis=0", "diag.gui.format=native"]
            if fields.strip():
                ov.append("diag.gui.fields=" + fields.strip())
            if rmax > 0:
                ov.append(f"diag.gui.rmax={rmax}")
            if xi_stride > 1:
                ov.append(f"diag.gui.xi_stride={xi_stride}")
        ov += [e for e in extra if "=" in e]
        cmd = [self.exe, deck_path] + ov
        if ranks > 1:
            mpi = shutil.which("mpirun") or shutil.which("mpiexec")
            if mpi is None:
                raise RuntimeError("mpirun not found")
            cmd = [mpi, "-np", str(ranks)] + (["--allow-run-as-root"] if os.geteuid() == 0 else []) + cmd
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = str(max(1, threads))
        env.setdefault("OMP_PROC_BIND", "false")
        wd = os.path.dirname(deck_path)
        # remove old gui frames of the same output directory (a new run starts at step 0)
        gdir = os.path.join(wd, info["outdir"], "gui")
        if os.path.isdir(gdir):
            for fn in os.listdir(gdir):
                if FIELD_RE.match(fn):
                    os.remove(os.path.join(gdir, fn))
        with self.lock:
            self.log = ["$ " + " ".join(cmd), ""]
            self.step, self.t, self.rc = -1, 0.0, None
            self.workdir, self.info, self.ranks = wd, info, ranks
            self.outdir = os.path.join(wd, info["outdir"])
            self.started, self.ended = time.time(), None
        self.proc = subprocess.Popen(cmd, cwd=wd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, start_new_session=True)
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def _reader(self, proc):
        for line in proc.stdout:
            line = line.rstrip("\n")
            if "OMP_PROC_BIND" in line or line.startswith("  In general,") or line.startswith("  For "):
                continue   # Kokkos binding advice
            m = STEP_RE.match(line)
            with self.lock:
                self.log.append(line)
                if len(self.log) > 4000:
                    del self.log[:1000]
                if m:
                    self.step, self.t = int(m.group(1)), float(m.group(2))
        proc.wait()
        with self.lock:
            self.rc = proc.returncode
            self.ended = time.time()
            self.log.append(f"[exit code {proc.returncode}]")

    def fresh(self, fn):
        """the file exists and, during a run, was written by this run (MPI runs write beams.txt at the end)"""
        if not os.path.exists(fn):
            return False
        return not (self.running() and self.started and os.path.getmtime(fn) < self.started)

    def stop(self):
        if self.running():
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def view(self, outdir=None):
        """directory with the frames: <outdir>/gui, else <outdir>"""
        od = outdir or self.outdir
        if od is None:
            return None
        g = os.path.join(od, "gui")
        return g if os.path.isdir(g) else od

    def frames(self, outdir=None):
        d = self.view(outdir)
        if d is None or not os.path.isdir(d):
            return []
        steps = sorted(int(m.group(1)) for m in (FIELD_RE.match(f) for f in os.listdir(d)) if m)
        if self.running() and self.ranks > 1:
            # pipeline: rank 0 prints step n while the last rank is at n - P + 1
            steps = [s for s in steps if s <= self.step - self.ranks]
        return [s for s in steps if field_header(os.path.join(d, f"fields_{s:06d}.bin")) is not None]

    def status(self, nlog=300):
        with self.lock:
            return {"running": self.running(), "rc": self.rc, "step": self.step, "t": self.t,
                    "steps": self.info.get("steps", 0), "t_end": self.info.get("t_end"),
                    "outdir": self.outdir, "workdir": self.workdir, "log": self.log[-nlog:],
                    "elapsed": ((self.ended or time.time()) - self.started) if self.started else 0.0}


# ----------------------------------------------------------------------------- HTTP
def resample_r(r, f, rr):
    """f[K, M] on the (non-uniform) nodes r -> f[K, len(rr)] by linear interpolation"""
    j = np.clip(np.searchsorted(r, rr) - 1, 0, len(r) - 2)
    w = np.clip((rr - r[j]) / (r[j + 1] - r[j]), 0, 1)
    return f[:, j] * (1 - w) + f[:, j + 1] * w


class Handler(BaseHTTPRequestHandler):
    server_version = "QUARZ-GUI"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _auth(self, q):
        tok = self.headers.get("X-Token") or q.get("token", [""])[0]
        return secrets.compare_digest(tok, self.server.token)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            if not self._auth(q):
                return self._send(403, "forbidden: open the URL with ?token=... printed by quarz_gui.py", "text/plain")
            with open(os.path.join(HERE, "quarz_gui.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if u.path == "/logo.svg":
            with open(os.path.join(ROOT, "docs", "logo", "favicon.svg"), "rb") as f:
                return self._send(200, f.read(), "image/svg+xml")
        if not self._auth(q):
            return self._send(403, {"error": "forbidden"})
        try:
            return self._send(200, self.api_get(u.path, q))
        except Exception as e:  # report to the page
            return self._send(400, {"error": str(e)})

    def do_POST(self):
        u = urlparse(self.path)
        if not self._auth(parse_qs(u.query)):
            return self._send(403, {"error": "forbidden"})
        n = int(self.headers.get("Content-Length", "0"))
        data = json.loads(self.rfile.read(n) or b"{}")
        try:
            return self._send(200, self.api_post(u.path, data))
        except Exception as e:
            return self._send(400, {"error": str(e)})

    # ------------------------------------------------------------------ GET API
    def api_get(self, path, q):
        R = self.server.runner
        a = lambda k, d=None: q.get(k, [d])[0]   # noqa: E731
        if path == "/api/init":
            p = self.server.deck
            text = open(p).read() if p and os.path.exists(p) else ""
            return {"deck": p or os.path.join(os.getcwd(), "run.in"), "text": text, "examples": self.examples(),
                    "exe": R.exe, "cpus": os.cpu_count() or 1, "mpirun": bool(shutil.which("mpirun"))}
        if path == "/api/deck":
            p = os.path.abspath(a("path"))
            return {"path": p, "text": open(p).read()}
        if path == "/api/status":
            return R.status()
        od = a("outdir")
        if path == "/api/frames":
            d = R.view(od)
            fr = R.frames(od)
            comps, info = [], {}
            if fr:
                h = field_header(os.path.join(d, f"fields_{fr[-1]:06d}.bin"))
                comps = [c for c in h[6] if not (c.endswith("_c") or c.endswith("_s"))]
                info = {"xi": [float(h[5][0]), float(h[5][-1])], "rmax": float(h[4][-1])}
            base = od or R.outdir
            beams = bool(base) and R.fresh(os.path.join(base, "beams.txt"))
            return {"dir": d, "frames": fr, "comps": comps, **info, "beams": beams}
        if path in ("/api/map", "/api/lines", "/api/rline"):
            d = R.view(od)
            rec = CACHE.get(os.path.join(d, f"fields_{int(a('step')):06d}.bin"))
            if rec is None:
                raise RuntimeError("frame not available")
            xi, r = rec["xi"], rec["r"]
            if path == "/api/map":
                comp = a("comp", "ez")
                f = fval(rec, comp)
                rmax = min(float(a("rmax", r[-1])), r[-1])
                nr = int(a("nr", 240))
                rr = np.linspace(0, rmax, nr)
                g = resample_r(r, f, rr)
                nxi = int(a("nxi", 900))
                if len(xi) > nxi:   # thin out in xi by block maxima of |f| (keeps narrow features)
                    k = int(np.ceil(len(xi) / nxi))
                    n = len(xi) // k
                    b = g[: n * k].reshape(n, k, nr)
                    idx = np.abs(b).argmax(axis=1)
                    g = np.take_along_axis(b, idx[:, None, :], axis=1)[:, 0, :]
                    xs = xi[: n * k].reshape(n, k).mean(axis=1)
                else:
                    xs = xi
                fin = g[np.isfinite(g)]
                vmin, vmax = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)
                # robust range: 99.5 % of the values (a closure spike would otherwise hide everything)
                if fin.size:
                    if comp in SIGNED:
                        a99 = float(np.percentile(np.abs(fin), 99.5))
                        rlo, rhi = -a99, a99
                    else:
                        rlo, rhi = float(np.percentile(fin, 0.5)), float(np.percentile(fin, 99.5))
                else:
                    rlo, rhi = vmin, vmax
                return {"t": rec["t"], "comp": comp, "signed": comp in SIGNED, "xi0": float(xs[0]),
                        "xi1": float(xs[-1]), "nxi": int(len(xs)), "rmax": rmax, "nr": nr, "vmin": vmin, "vmax": vmax,
                        "rlo": rlo, "rhi": rhi,
                        "data": base64.b64encode(np.ascontiguousarray(g, dtype=np.float32).tobytes()).decode()}
            comps = [c for c in a("comps", "ez").split(",") if c]
            if path == "/api/lines":
                r0 = float(a("r", 0))
                out = {}
                for c in comps:
                    f = fval(rec, c)
                    out[c] = resample_r(r, f, np.array([r0]))[:, 0].tolist() if r0 > 0 else f[:, 0].tolist()
                return {"t": rec["t"], "x": xi.tolist(), "series": out, "r": r0}
            xc = float(a("xi", xi[0]))
            k = int(np.argmin(np.abs(xi - xc)))
            rmax = float(a("rmax", r[-1]))
            m = r <= rmax * 1.0001
            return {"t": rec["t"], "x": r[m].tolist(), "xi": float(xi[k]),
                    "series": {c: fval(rec, c)[k, m].tolist() for c in comps}}
        if path == "/api/beams":
            base = od or R.outdir
            fn = os.path.join(base or ".", "beams.txt")
            if not R.fresh(fn):
                return {}
            try:
                b = read_beamlog(fn)
            except Exception:
                return {}
            return {k: {c: np.asarray(v[c]).tolist() for c in ("t", "gamma_mean", "gamma_rms", "r_rms", "emit_nx",
                                                                 "xi_mean", "alive", "charge", "x_mean")}
                    for k, v in b.items()}
        raise RuntimeError("unknown request " + path)

    def examples(self):
        out = []
        for sub in ("examples", os.path.join("paper", "inputs"), "validation"):
            d = os.path.join(ROOT, sub)
            if os.path.isdir(d):
                out += sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".in"))
        return out

    # ------------------------------------------------------------------ POST API
    def api_post(self, path, d):
        R = self.server.runner
        if path == "/api/save":
            p = os.path.abspath(d["path"])
            with open(p, "w") as f:
                f.write(d["text"])
            return {"path": p}
        if path == "/api/run":
            R.start(d["path"], d["text"], int(d.get("threads", 1)), int(d.get("ranks", 1)), int(d.get("every", 1)),
                    float(d.get("rmax", 0) or 0), int(d.get("xi_stride", 1) or 1), d.get("fields", ""),
                    [s.strip() for s in d.get("extra", "").split("\n") if s.strip()])
            return {"ok": True}
        if path == "/api/stop":
            R.stop()
            return {"ok": True}
        if path == "/api/open":   # browse the output of a finished run
            text = d.get("text", "")
            info = deck_info(text)
            wd = os.path.dirname(os.path.abspath(d["path"]))
            R.outdir, R.workdir, R.info = os.path.join(wd, info["outdir"]), wd, info
            return {"outdir": R.outdir}
        raise RuntimeError("unknown request " + path)


def find_exe(arg):
    for c in [arg, os.path.join(ROOT, "build", "quarz"), shutil.which("quarz")]:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return os.path.abspath(c)
    return arg or "quarz"


def main():
    ap = argparse.ArgumentParser(description="QUARZ GUI (deck editor, run control, live plots in the browser)")
    ap.add_argument("deck", nargs="?", help="input deck to open")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--exe", help="quarz executable (default: build/quarz of this repository, then PATH)")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser (e.g. on a remote machine)")
    ap.add_argument("--token", help=argparse.SUPPRESS)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    srv.daemon_threads = True
    srv.token = args.token or secrets.token_urlsafe(16)
    srv.runner = Runner(find_exe(args.exe))
    srv.deck = os.path.abspath(args.deck) if args.deck else None
    if srv.deck:
        info = deck_info(open(srv.deck).read()) if os.path.exists(srv.deck) else {"outdir": "out"}
        srv.runner.outdir = os.path.join(os.path.dirname(srv.deck), info["outdir"])
        srv.runner.workdir = os.path.dirname(srv.deck)
    url = f"http://127.0.0.1:{args.port}/?token={srv.token}"
    print(f"QUARZ GUI: {url}\n  executable: {srv.runner.exe}\n  Ctrl-C to quit", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.runner.stop()


if __name__ == "__main__":
    main()
