"""Read-only, seekable file object over HTTP range requests (stdlib only).

Lets zipfile / h5py read single members or datasets out of large remote archives
without downloading them. Reads are cached in fixed-size blocks.
"""
import io
import urllib.request


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url, block_size=4 * 2**20, max_blocks=256):
        self.url = url
        self.block_size = block_size
        self.max_blocks = max_blocks
        self._cache = {}
        self.pos = 0
        self.bytes_fetched = 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Length"])
            assert r.headers.get("Accept-Ranges") == "bytes", "server does not advertise range support"

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def _block(self, i):
        if i not in self._cache:
            start = i * self.block_size
            end = min(start + self.block_size, self.size) - 1
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req) as r:
                data = r.read()
            self.bytes_fetched += len(data)
            if len(self._cache) >= self.max_blocks:
                self._cache.pop(next(iter(self._cache)))
            self._cache[i] = data
        return self._cache[i]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while len(out) < n:
            i, off = divmod(self.pos, self.block_size)
            chunk = self._block(i)[off:off + n - len(out)]
            out += chunk
            self.pos += len(chunk)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)
