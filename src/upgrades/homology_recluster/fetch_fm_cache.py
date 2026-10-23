#!/usr/bin/env python3
"""Fetch ONLY the frozen-FM embedding files (HyenaDNA + DNABERT-2, original splits) from the released
Zenodo cache archive (DOI 10.5281/zenodo.20494982, cache_artifact.zip, 1.5 GB) without downloading the
whole archive: the zip central directory and the ten `fm_embeddings/emb__*__orig__*.npz` members are read
with HTTP range requests. Every extracted file is verified against the sha256 recorded in
results/upgrades/fm_embeddings_manifest.csv (the manifest written by fm_probe/reproduce_from_cache.py when
the cache was built); a mismatch aborts. Output: external/homology_recluster/fm_embeddings/.
Network only, no compute. Usage: python fetch_fm_cache.py
"""
import hashlib
import io
import os
import sys
import urllib.request
import zipfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _hc  # noqa: E402

URL = "https://zenodo.org/api/records/20494982/files/cache_artifact.zip/content"
OUT = os.path.join(_hc.EXT, "fm_embeddings")


class HTTPRangeFile(io.RawIOBase):
    """Minimal seekable read-only file over HTTP range requests (enough for zipfile)."""

    def __init__(self, url, block=1 << 20):
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=120) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.block = block
        self._cache = {}

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def _get(self, start, end):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = r.read()
                if len(data) == end - start:
                    return data
            except Exception as e:  # noqa: BLE001  (retry transient network errors)
                print(f"   range {start}-{end} attempt {attempt}: {e}", flush=True)
        raise IOError(f"range read failed {start}-{end}")

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n == 0:
            return b""
        data = self._get(self.pos, self.pos + n) if n > self.block else self._cached(self.pos, n)
        self.pos += len(data)
        return data

    def _cached(self, pos, n):
        b0 = pos // self.block
        b1 = (pos + n - 1) // self.block
        chunks = []
        for b in range(b0, b1 + 1):
            if b not in self._cache:
                if len(self._cache) > 64:
                    self._cache.clear()
                s = b * self.block
                self._cache[b] = self._get(s, min(s + self.block, self.size))
            chunks.append(self._cache[b])
        buf = b"".join(chunks)
        off = pos - b0 * self.block
        return buf[off:off + n]

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def main():
    os.makedirs(OUT, exist_ok=True)
    man = pd.read_csv(os.path.join(_hc.RES, "fm_embeddings_manifest.csv")).set_index("file")
    want = [f for f in man.index if "__orig__" in f]
    todo = []
    for f in want:
        p = os.path.join(OUT, f)
        if os.path.exists(p) and hashlib.sha256(open(p, "rb").read()).hexdigest() == man.loc[f, "sha256"]:
            print(f"[ok] {f} (cached, sha256 verified)")
        else:
            todo.append(f)
    if not todo:
        return
    hf = HTTPRangeFile(URL)
    print(f"archive size {hf.size} bytes; reading central directory", flush=True)
    zf = zipfile.ZipFile(io.BufferedReader(hf, buffer_size=1 << 20))
    names = {os.path.basename(n): n for n in zf.namelist() if n.endswith(".npz")}
    for f in todo:
        if f not in names:
            sys.exit(f"ERROR: {f} not in archive")
        tmp = os.path.join(OUT, f + ".part")
        with zf.open(names[f]) as src, open(tmp, "wb") as dst:
            while True:
                b = src.read(1 << 22)
                if not b:
                    break
                dst.write(b)
        h = hashlib.sha256(open(tmp, "rb").read()).hexdigest()
        if h != man.loc[f, "sha256"]:
            os.remove(tmp)
            sys.exit(f"ERROR: sha256 mismatch for {f}: {h} != {man.loc[f, 'sha256']}")
        os.replace(tmp, os.path.join(OUT, f))
        print(f"[fetched] {f} ({os.path.getsize(os.path.join(OUT, f))} bytes, sha256 verified)", flush=True)


if __name__ == "__main__":
    main()
