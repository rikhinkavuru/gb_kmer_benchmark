#!/usr/bin/env python3
"""Hard memory cap for the fine-tuning jobs (added 2026-09-24 after the 17.6 GB incident).

What happened: the HyenaDNA fine-tune on drosophila_enhancers_stark (PID 62701) grew to a 17.6 GB
footprint on a 16 GB laptop and pushed ~16 GB into swap. drosophila sequences have 535 distinct
lengths (236-3237 bp) and every mini-batch was padded only to its own longest sequence, so almost
every batch had a new tensor shape. HyenaDNA's long convolution is an FFT of size 2*L, and the MPS
backend compiles and caches a graph (with its own scratch buffers, outside PyTorch's allocator and
therefore outside PYTORCH_MPS_HIGH_WATERMARK_RATIO) for every new shape, so the cache grew without
bound. The cohn / nt_enhancers runs never showed it because those sequences have 1-2 lengths.
The fix in run_ft_v2.py is --pad-multiple (a handful of fixed shapes); this module is the safety
net that makes a repeat impossible whatever the cause.

The number watched is phys_footprint (proc_pid_rusage, RUSAGE_INFO_V2): the figure Activity
Monitor's "Memory" column shows, which on Apple silicon includes Metal/MPS buffers (RSS does not).

In-process:   g = MemGuard(cap_mb, log).start()   -> daemon thread; os._exit(86) above the cap
External:     python memguard.py --watch PID --cap-mb N   -> SIGTERM, then SIGKILL, above the cap
              (backstop for the case where the in-process thread cannot run). Torch-free.
Probe:        python memguard.py --pid PID           -> print the footprint in MB (10^6 bytes)
"""
import argparse
import ctypes
import os
import signal
import sys
import threading
import time

RUSAGE_INFO_V2 = 2
_FOOTPRINT_OFFSET = 72        # uuid[16] + 7 x uint64 (user, system, pkg_idle, interrupt, pageins,
                              # wired, resident) -> ri_phys_footprint
_RESIDENT_OFFSET = 64
_libc = None
EXIT_CODE = 86


def _rusage(pid):
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    buf = ctypes.create_string_buffer(512)
    if _libc.proc_pid_rusage(ctypes.c_int(int(pid)), ctypes.c_int(RUSAGE_INFO_V2), buf) != 0:
        return None
    return buf.raw


def footprint_mb(pid=None):
    """phys_footprint of `pid` (default: this process) in MB (10^6 bytes); nan if unreadable."""
    pid = os.getpid() if pid is None else pid
    if sys.platform != "darwin":
        try:
            with open(f"/proc/{pid}/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024 / 1e6
        except OSError:
            pass
        return float("nan")
    raw = _rusage(pid)
    if raw is None:
        return float("nan")
    return int.from_bytes(raw[_FOOTPRINT_OFFSET:_FOOTPRINT_OFFSET + 8], "little") / 1e6


def resident_mb(pid=None):
    raw = _rusage(os.getpid() if pid is None else pid) if sys.platform == "darwin" else None
    if raw is None:
        return float("nan")
    return int.from_bytes(raw[_RESIDENT_OFFSET:_RESIDENT_OFFSET + 8], "little") / 1e6


class MemGuard:
    """Daemon thread: sample this process's footprint every `interval` s, keep the peak, and hard-exit
    (os._exit(86): frees everything at once, nothing half-written because outputs are written to
    .tmp then renamed) if it ever exceeds `cap_mb`."""

    def __init__(self, cap_mb, log=print, interval=5.0):
        self.cap_mb = float(cap_mb)
        self.log = log
        self.interval = float(interval)
        self.peak_mb = 0.0
        self._stop = threading.Event()

    def sample(self):
        fp = footprint_mb()
        if fp == fp:
            self.peak_mb = max(self.peak_mb, fp)
        return fp

    def _run(self):
        while not self._stop.is_set():
            fp = self.sample()
            if fp == fp and fp > self.cap_mb:
                try:
                    self.log(f"!!! MEMGUARD: footprint {fp:.0f} MB > cap {self.cap_mb:.0f} MB -- "
                             f"aborting this process (exit {EXIT_CODE}) before it can swap the machine")
                    sys.stdout.flush(); sys.stderr.flush()
                finally:
                    os._exit(EXIT_CODE)
            self._stop.wait(self.interval)

    def start(self):
        self.sample()
        threading.Thread(target=self._run, name="memguard", daemon=True).start()
        return self

    def stop(self):
        self._stop.set()


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def watch(pid, cap_mb, interval=15.0, grace=20.0, report_every=0.0):
    """External backstop: terminate `pid` if its footprint exceeds cap_mb. Returns when pid exits.
    report_every > 0: also print the footprint every that many seconds (a memory trace)."""
    peak = 0.0
    last = 0.0
    while _alive(pid):
        fp = footprint_mb(pid)
        if fp == fp:
            peak = max(peak, fp)
            if report_every and time.time() - last >= report_every:
                last = time.time()
                print(time.strftime("%H:%M:%S"), f"[memguard] pid {pid} footprint {fp:.0f} MB "
                      f"(peak {peak:.0f}, cap {cap_mb:.0f})", flush=True)
            if fp > cap_mb:
                print(time.strftime("%H:%M:%S"), f"!!! MEMGUARD(external): pid {pid} footprint "
                      f"{fp:.0f} MB > cap {cap_mb:.0f} MB -> SIGTERM", flush=True)
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass
                t0 = time.time()
                while _alive(pid) and time.time() - t0 < grace:
                    time.sleep(1)
                if _alive(pid):
                    print(time.strftime("%H:%M:%S"), f"!!! MEMGUARD(external): SIGKILL {pid}",
                          flush=True)
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except OSError:
                        pass
                break
        time.sleep(interval)
    return peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int)
    ap.add_argument("--watch", type=int)
    ap.add_argument("--cap-mb", type=float, default=5000.0)
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--report-every", type=float, default=0.0)
    a = ap.parse_args()
    if a.watch:
        peak = watch(a.watch, a.cap_mb, a.interval, report_every=a.report_every)
        print(time.strftime("%H:%M:%S"), f"[memguard] pid {a.watch} gone; peak footprint seen "
              f"{peak:.0f} MB (external cap {a.cap_mb:.0f} MB)", flush=True)
        return
    pid = a.pid or os.getpid()
    print(f"pid {pid}: footprint {footprint_mb(pid):.1f} MB, resident {resident_mb(pid):.1f} MB")


if __name__ == "__main__":
    main()
