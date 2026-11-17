"""Restore immutable source archive members into the explicit training cache."""

import concurrent.futures
import hashlib
import json
import logging
import os
import struct
import tempfile
import time
import urllib.request
import zlib
from pathlib import Path

from ..errors import DatasetError
from ..io import sha256

LOG = logging.getLogger(__name__)


def prepare(manifest, *, workers=8):
    manifest = Path(manifest).resolve()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise DatasetError("Unsupported training manifest")
    cache_root = (manifest.parent.parent / "cache").resolve()
    rows = [
        r
        for key in ("reader_train", "reader_valid", "detector_train", "detector_valid")
        for r in data[key]
    ]
    pending = []
    for row in rows:
        path = (manifest.parent / row["image"]).resolve()
        if not path.is_relative_to(cache_root):
            raise DatasetError("Download targets must stay under data/cache")
        if path.exists() and sha256(path) == row["sha256"]:
            continue
        pending.append((row, path))
    if not pending:
        LOG.info("All %s training images already verified", len(rows))
        return
    archives = {}
    for source, name in sorted({(r["source"], r["archive"]) for r, p in pending}):
        meta = data["sources"][source]
        if name not in ("train.zip", "val.zip"):
            raise DatasetError("Only source train/val archives are supported")
        with request(
            f"https://huggingface.co/api/datasets/{meta['repo']}/tree/{meta['revision']}"
        ) as r:
            size = next(item["size"] for item in json.load(r) if item["path"] == name)
        archives[source, name] = RemoteZip(meta["repo"], meta["revision"], size, name)

    def restore(job):
        row, path = job
        raw = archives[row["source"], row["archive"]].read(row["source_entry"])
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise DatasetError("Source image checksum mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, suffix=".tmp", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(raw)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return path

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, 32)) as pool:
        for count, path in enumerate(pool.map(restore, pending), 1):
            if count % 100 == 0 or count == len(pending):
                LOG.info("Restored %s/%s images", count, len(pending))


def request(url, headers=None):
    req = urllib.request.Request(
        url, headers={"User-Agent": "independent-plate-evaluation/1.0", **(headers or {})}
    )
    for attempt in range(3):
        try:
            return urllib.request.urlopen(req, timeout=45)
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


class RemoteZip:
    def __init__(self, repo, revision, size, archive="test.zip"):
        self.url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{archive}"
        tail = self.fetch(max(0, size - 65536), size - 1)
        i = tail.rfind(b"PK\x05\x06")
        end = struct.unpack_from("<4s4H2IH", tail, i)
        cd_size, cd_offset = end[5:7]
        if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
            locator = struct.unpack_from("<4sIQI", tail, i - 20)
            if locator[0] != b"PK\x06\x07":
                raise ValueError("Missing ZIP64 locator")
            extended = struct.unpack("<4sQ2H2I4Q", self.fetch(locator[2], locator[2] + 55))
            cd_size, cd_offset = extended[8:10]
        central = self.fetch(cd_offset, cd_offset + cd_size - 1)
        self.entries = {}
        pos = 0
        while pos < len(central):
            values = struct.unpack_from("<4s6H3I5H2I", central, pos)
            if values[0] != b"PK\x01\x02":
                raise ValueError("Invalid ZIP directory")
            n, e, c = values[10:13]
            name = central[pos + 46 : pos + 46 + n].decode(
                "utf-8" if values[3] & 0x800 else "cp437"
            )
            offset, compressed, uncompressed = values[16], values[8], values[9]
            extra = central[pos + 46 + n : pos + 46 + n + e]
            extra_pos = 0
            while extra_pos + 4 <= len(extra):
                tag, length = struct.unpack_from("<HH", extra, extra_pos)
                if tag == 1:
                    cursor = extra_pos + 4
                    if uncompressed == 0xFFFFFFFF:
                        uncompressed = struct.unpack_from("<Q", extra, cursor)[0]
                        cursor += 8
                    if compressed == 0xFFFFFFFF:
                        compressed = struct.unpack_from("<Q", extra, cursor)[0]
                        cursor += 8
                    if offset == 0xFFFFFFFF:
                        offset = struct.unpack_from("<Q", extra, cursor)[0]
                extra_pos += 4 + length
            self.entries[name] = {
                "offset": offset,
                "compressed": compressed,
                "size": uncompressed,
                "crc": values[7],
                "method": values[4],
            }
            pos += 46 + n + e + c

    def fetch(self, start, end):
        # A distinct URL avoids caches of redirects whose signatures restrict the Range.
        url = self.url + f"?download=true&range_start={start}&range_end={end}"
        with request(url, {"Range": f"bytes={start}-{end}"}) as response:
            if response.status != 206 or not response.headers.get("Content-Range", "").startswith(
                f"bytes {start}-"
            ):
                raise ValueError("Server did not honor requested ZIP byte range")
            value = response.read()
        if len(value) != end - start + 1:
            raise ValueError("Incomplete HTTP range")
        return value

    def read(self, name):
        entry = self.entries[name]
        offset = entry["offset"]
        # Local extra fields in these archives are small; clip near the archive end if needed.
        prefix = self.fetch(offset, offset + 29)
        header = struct.unpack("<4s5H3I2H", prefix)
        if header[0] != b"PK\x03\x04":
            raise ValueError("Invalid ZIP entry")
        start = offset + 30 + header[-2] + header[-1]
        data = self.fetch(start, start + entry["compressed"] - 1)
        if entry["method"] == 8:
            data = zlib.decompress(data, -15)
        elif entry["method"] != 0:
            raise ValueError("Unsupported ZIP compression")
        if len(data) != entry["size"] or zlib.crc32(data) & 0xFFFFFFFF != entry["crc"]:
            raise ValueError("ZIP CRC or length mismatch")
        return data
