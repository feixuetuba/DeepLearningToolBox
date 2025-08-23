"""
Unified POSIX-style FileSystem layer with pluggable backends (Local, SFTP, SMB)
plus a concurrent FileTransferManager for cross-protocol copies.

Dependencies (install as needed):
    pip install paramiko pysmb tqdm

Tested on Python 3.9+

Design highlights
- Abstract base PosixFileSystem defines POSIX-like operations.
- LocalFileSystem uses built-ins.
- SFTPFileSystem wraps paramiko.SFTPClient.
- SMBFileSystem wraps pysmb.SMBConnection.
- FileTransferManager supports concurrent copy/mirror across any two FSs,
  auto-creates destination directories, and reports progress callbacks.
- URL-style helper factory: file:///... , sftp://user:pass@host:port/path ,
  smb://user:pass@server/share/path

Security note: When possible, prefer using key files for SFTP.
Avoid embedding plaintext passwords in code/URLs in production.
"""
from __future__ import annotations

import os
import io
import stat as pystat
import shutil
import pathlib
import tempfile
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, List, Optional, Tuple, Union

# Optional deps guarded by import try/except
try:
    import paramiko  # type: ignore
except Exception:  # pragma: no cover
    paramiko = None  # type: ignore

try:
    from smb.SMBConnection import SMBConnection  # type: ignore
    from smb.base import SharedFile  # type: ignore
except Exception:  # pragma: no cover
    SMBConnection = None  # type: ignore
    SharedFile = None  # type: ignore


# -----------------------------
# Exceptions and dataclasses
# -----------------------------
class FSError(Exception):
    pass


@dataclass
class FSStat:
    size: int
    mode: int
    mtime: float
    atime: float
    ctime: float


# -----------------------------
# POSIX-like FS abstraction
# -----------------------------
class PosixFileSystem(ABC):
    """POSIX-style interface to unify diverse backends.

    Paths use forward slashes ('/') regardless of backend.
    Implementations should internally translate path syntax as needed.
    """

    # ---- Capability flags ----
    supports_stream_open: bool = False  # True if open(path, 'rb'|'wb') returns a streaming file-like

    # ---- Core operations ----
    @abstractmethod
    def open(self, path: str, mode: str):
        """Open a path in binary/text mode depending on `mode`.
        Should return a context-manager file-like if supports_stream_open=True.
        If unsupported, raise NotImplementedError and let manager fallback to temp-file.
        """
        raise NotImplementedError

    @abstractmethod
    def read_bytes(self, path: str) -> bytes:
        pass

    @abstractmethod
    def write_bytes(self, path: str, data: bytes) -> None:
        pass

    @abstractmethod
    def listdir(self, path: str) -> List[str]:
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        pass

    @abstractmethod
    def is_dir(self, path: str) -> bool:
        pass

    @abstractmethod
    def stat(self, path: str) -> FSStat:
        pass

    @abstractmethod
    def remove(self, path: str) -> None:
        pass

    @abstractmethod
    def mkdir(self, path: str, parents: bool = False, exist_ok: bool = True) -> None:
        pass

    @abstractmethod
    def rename(self, src: str, dst: str, overwrite: bool = False) -> None:
        pass

    # ---- Convenience helpers ----
    def ensure_parent_dir(self, path: str) -> None:
        parent = str(pathlib.PurePosixPath(path).parent)
        if parent and parent not in (".", "/"):
            try:
                self.mkdir(parent, parents=True, exist_ok=True)
            except Exception:
                # Some backends may not support nested mkdir idempotently; ignore if parent exists
                pass


# -----------------------------
# Local filesystem implementation
# -----------------------------
class LocalFileSystem(PosixFileSystem):
    supports_stream_open = True

    def _to_native(self, path: str) -> str:
        # Convert POSIX-style to OS-native path
        p = pathlib.PurePosixPath(path)
        return str(pathlib.Path(*p.parts))

    def open(self, path: str, mode: str):
        native = self._to_native(path)
        return open(native, mode)

    def read_bytes(self, path: str) -> bytes:
        with self.open(path, "rb") as f:
            return f.read()

    def write_bytes(self, path: str, data: bytes) -> None:
        self.ensure_parent_dir(path)
        with self.open(path, "wb") as f:
            f.write(data)

    def listdir(self, path: str) -> List[str]:
        native = self._to_native(path)
        return os.listdir(native)

    def exists(self, path: str) -> bool:
        return os.path.exists(self._to_native(path))

    def is_dir(self, path: str) -> bool:
        return os.path.isdir(self._to_native(path))

    def stat(self, path: str) -> FSStat:
        st = os.stat(self._to_native(path))
        return FSStat(size=st.st_size, mode=st.st_mode, mtime=st.st_mtime, atime=st.st_atime, ctime=st.st_ctime)

    def remove(self, path: str) -> None:
        native = self._to_native(path)
        if os.path.isdir(native) and not os.path.islink(native):
            shutil.rmtree(native)
        else:
            os.remove(native)

    def mkdir(self, path: str, parents: bool = False, exist_ok: bool = True) -> None:
        native = self._to_native(path)
        if parents:
            os.makedirs(native, exist_ok=exist_ok)
        else:
            os.mkdir(native)

    def rename(self, src: str, dst: str, overwrite: bool = False) -> None:
        src_n = self._to_native(src)
        dst_n = self._to_native(dst)
        if overwrite and os.path.exists(dst_n):
            if os.path.isdir(dst_n) and not os.path.islink(dst_n):
                shutil.rmtree(dst_n)
            else:
                os.remove(dst_n)
        os.replace(src_n, dst_n)


# -----------------------------
# SFTP filesystem (paramiko)
# -----------------------------
class SFTPFileSystem(PosixFileSystem):
    supports_stream_open = True

    def __init__(self,
                 host: str,
                 username: str,
                 password: Optional[str] = None,
                 port: int = 22,
                 key_filename: Optional[str] = None,
                 allow_agent: bool = True,
                 look_for_keys: bool = True,
                 compress: bool = False,
                 timeout: Optional[float] = None):
        if paramiko is None:
            raise ImportError("paramiko is required for SFTPFileSystem. Install with `pip install paramiko`. ")
        self._transport = paramiko.Transport((host, port))
        pkey = None
        if key_filename:
            try:
                pkey = paramiko.RSAKey.from_private_key_file(key_filename)
            except Exception:
                # Let Paramiko try other key types automatically inside connect()
                pkey = None
        self._transport.use_compression(compress)
        self._transport.connect(username=username, password=password, pkey=pkey)
        self._sftp = paramiko.SFTPClient.from_transport(self._transport)
        if timeout is not None:
            self._sftp.get_channel().settimeout(timeout)

    def _norm(self, path: str) -> str:
        # SFTP uses POSIX-style already; just ensure no redundant parts
        return str(pathlib.PurePosixPath(path))

    def open(self, path: str, mode: str):
        return self._sftp.open(self._norm(path), mode)

    def read_bytes(self, path: str) -> bytes:
        with self.open(path, 'rb') as f:
            return f.read()

    def write_bytes(self, path: str, data: bytes) -> None:
        self.ensure_parent_dir(path)
        with self.open(path, 'wb') as f:
            f.write(data)

    def listdir(self, path: str) -> List[str]:
        return self._sftp.listdir(self._norm(path))

    def exists(self, path: str) -> bool:
        try:
            self._sftp.stat(self._norm(path))
            return True
        except IOError:
            return False

    def is_dir(self, path: str) -> bool:
        try:
            st = self._sftp.stat(self._norm(path))
            return pystat.S_ISDIR(st.st_mode)
        except IOError:
            return False

    def stat(self, path: str) -> FSStat:
        st = self._sftp.stat(self._norm(path))
        return FSStat(size=st.st_size, mode=st.st_mode, mtime=st.st_mtime, atime=st.st_atime, ctime=getattr(st, 'st_ctime', st.st_mtime))

    def remove(self, path: str) -> None:
        p = self._norm(path)
        try:
            if self.is_dir(p):
                # Paramiko lacks recursive remove; perform manual walk
                for name in self._sftp.listdir(p):
                    child = f"{p}/{name}"
                    if self.is_dir(child):
                        self.remove(child)
                    else:
                        self._sftp.remove(child)
                self._sftp.rmdir(p)
            else:
                self._sftp.remove(p)
        except IOError as e:
            raise FSError(str(e))

    def mkdir(self, path: str, parents: bool = False, exist_ok: bool = True) -> None:
        p = self._norm(path)
        if not parents:
            try:
                self._sftp.mkdir(p)
            except IOError as e:
                if exist_ok and self.exists(p):
                    return
                raise FSError(str(e))
            return
        # parents=True
        parts = pathlib.PurePosixPath(p).parts
        cur = ''
        for part in parts:
            cur = f"{cur}/{part}" if cur else part
            try:
                self._sftp.stat(cur)
            except IOError:
                try:
                    self._sftp.mkdir(cur)
                except IOError as e:
                    if exist_ok and self.exists(cur):
                        continue
                    raise FSError(str(e))

    def rename(self, src: str, dst: str, overwrite: bool = False) -> None:
        s = self._norm(src)
        d = self._norm(dst)
        if overwrite and self.exists(d):
            try:
                self.remove(d)
            except Exception:
                pass
        self._sftp.rename(s, d)

    def close(self) -> None:
        try:
            self._sftp.close()
        finally:
            self._transport.close()


# -----------------------------
# SMB filesystem (pysmb)
# -----------------------------
class SMBFileSystem(PosixFileSystem):
    supports_stream_open = False  # pysmb streams via retrieveFile/storeFile but doesn't expose a seekable file

    def __init__(self,
                 server: str,
                 share: str,
                 username: str,
                 password: str,
                 domain: str = '',
                 port: int = 139,
                 use_ntlm_v2: bool = True,
                 is_direct_tcp: bool = False,
                 client_name: Optional[str] = None,
                 server_name: Optional[str] = None):
        """
        Args:
            server: SMB server hostname or IP
            share: top-level share name (e.g. "public")
            domain: Windows domain (often empty)
            port: 139 (NetBIOS) or 445 (Direct TCP)
            is_direct_tcp: set True when using port 445
            client_name: NetBIOS name for client (optional)
            server_name: NetBIOS name for server (optional)
        """
        if SMBConnection is None:
            raise ImportError("pysmb is required for SMBFileSystem. Install with `pip install pysmb`.")
        self.server = server
        self.share = share
        self.conn = SMBConnection(username, password,
                                   my_name=client_name or os.uname().nodename,
                                   remote_name=server_name or server,
                                   domain=domain,
                                   use_ntlm_v2=use_ntlm_v2,
                                   is_direct_tcp=is_direct_tcp)
        if not self.conn.connect(server, port):
            raise FSError(f"Failed to connect to SMB server {server}:{port}")

    def _split(self, path: str) -> Tuple[str, str]:
        """Return (share, path_in_share). We are already bound to one share,
        but accept paths like '/dir/file' and normalize leading slashes.
        """
        p = str(pathlib.PurePosixPath(path)).lstrip('/')
        return self.share, p

    # SMB doesn't expose a generic open; we leave it unsupported
    def open(self, path: str, mode: str):  # pragma: no cover - not supported
        raise NotImplementedError("SMBFileSystem does not support direct open(); use read/write/copy via manager.")

    def read_bytes(self, path: str) -> bytes:
        share, rel = self._split(path)
        bio = io.BytesIO()
        self.conn.retrieveFile(share, f"/{rel}", bio)
        return bio.getvalue()

    def write_bytes(self, path: str, data: bytes) -> None:
        share, rel = self._split(path)
        self.ensure_parent_dir(path)
        bio = io.BytesIO(data)
        # Overwrite by default
        self.conn.storeFile(share, f"/{rel}", bio)

    def listdir(self, path: str) -> List[str]:
        share, rel = self._split(path)
        rel = '/' + rel if rel else '/'
        files = self.conn.listPath(share, rel)
        # Exclude current/parent pseudo-entries
        names = [f.filename for f in files if getattr(f, 'filename', '') not in ('.', '..')]
        return names

    def exists(self, path: str) -> bool:
        share, rel = self._split(path)
        try:
            self.conn.getAttributes(share, f"/{rel}")
            return True
        except Exception:
            return False

    def is_dir(self, path: str) -> bool:
        share, rel = self._split(path)
        try:
            attrs = self.conn.getAttributes(share, f"/{rel}")
            # In pysmb, file_attributes has DIRECTORY flag 0x10
            return bool(attrs.isDirectory)
        except Exception:
            return False

    def stat(self, path: str) -> FSStat:
        share, rel = self._split(path)
        attrs = self.conn.getAttributes(share, f"/{rel}")
        mode = 0o777
        return FSStat(size=attrs.file_size, mode=mode, mtime=attrs.last_write_time, atime=attrs.last_access_time, ctime=attrs.create_time)

    def remove(self, path: str) -> None:
        share, rel = self._split(path)
        if self.is_dir(path):
            # Recursive delete
            for name in self.listdir(path):
                child = str(pathlib.PurePosixPath(path) / name)
                self.remove(child)
            self.conn.deleteDirectory(share, f"/{rel}")
        else:
            self.conn.deleteFiles(share, f"/{rel}")

    def mkdir(self, path: str, parents: bool = False, exist_ok: bool = True) -> None:
        share, rel = self._split(path)
        parts = pathlib.PurePosixPath(rel).parts
        if not parents:
            try:
                self.conn.createDirectory(share, f"/{rel}")
            except Exception:
                if not (exist_ok and self.exists(path)):
                    raise
            return
        cur = ''
        for part in parts:
            cur = f"{cur}/{part}" if cur else f"/{part}"
            try:
                self.conn.createDirectory(share, cur)
            except Exception:
                # ignore if exists
                pass

    def rename(self, src: str, dst: str, overwrite: bool = False) -> None:
        share_s, rel_s = self._split(src)
        share_d, rel_d = self._split(dst)
        if share_s != share_d:
            raise FSError("Cross-share rename not supported; use manager.copy+remove")
        if overwrite and self.exists(dst):
            self.remove(dst)
        # pysmb lacks a direct rename; emulate with copy+delete using server-side?
        # Fallback: download temp and re-upload (manager recommended instead)
        data = self.read_bytes(src)
        self.write_bytes(dst, data)
        self.remove(src)

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


# -----------------------------
# URL factory
# -----------------------------
from urllib.parse import urlparse, unquote

def fs_from_url(url: str) -> Tuple[PosixFileSystem, str]:
    """Create FS instance from URL and return (fs, path). Supported schemes:
        file:///abs/path
        sftp://user:pass@host:port/abs/path
        smb://user:pass@server/share/path
    The returned `path` is POSIX-style within the FS.
    """
    u = urlparse(url)
    scheme = u.scheme.lower()
    if scheme == 'file':
        fs = LocalFileSystem()
        path = unquote(u.path)
        return fs, path
    elif scheme == 'sftp':
        username = unquote(u.username or '')
        password = unquote(u.password or '') if u.password else None
        host = u.hostname or 'localhost'
        port = u.port or 22
        fs = SFTPFileSystem(host=host, username=username, password=password, port=port)
        path = unquote(u.path)
        return fs, path
    elif scheme == 'smb':
        # Expect smb://user:pass@server/share/path
        username = unquote(u.username or '')
        password = unquote(u.password or '') if u.password else ''
        server = u.hostname or ''
        # first path segment is the share
        parts = pathlib.PurePosixPath(unquote(u.path)).parts
        if not parts:
            raise FSError("SMB URL must include /<share>/path")
        share = parts[0]
        rel = '/'.join(parts[1:])
        fs = SMBFileSystem(server=server, share=share, username=username, password=password, port=445, is_direct_tcp=True)
        return fs, '/' + rel if rel else '/'
    else:
        raise FSError(f"Unsupported scheme: {scheme}")


# -----------------------------
# Transfer manager with concurrency
# -----------------------------
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class CopyTask:
    src_path: str
    dst_path: str
    src_fs: PosixFileSystem
    dst_fs: PosixFileSystem
    overwrite: bool = True


ProgressCallback = Callable[[str, int, Optional[int]], None]
# callback(file_key, bytes_done, total_bytes or None)


class FileTransferManager:
    def __init__(self, max_workers: int = 4, chunk_size: int = 1024 * 1024):
        self.max_workers = max_workers
        self.chunk_size = chunk_size

    # ---- Public API ----
    def copy(self,
             src_path: str,
             dst_path: str,
             src_fs: PosixFileSystem,
             dst_fs: PosixFileSystem,
             overwrite: bool = True,
             progress: Optional[ProgressCallback] = None) -> None:
        self._copy_one(CopyTask(src_path, dst_path, src_fs, dst_fs, overwrite), progress)

    def copy_many(self,
                  tasks: Iterable[CopyTask],
                  progress: Optional[ProgressCallback] = None) -> List[Tuple[CopyTask, Optional[Exception]]]:
        """Run tasks concurrently; returns a list of (task, exception) where exception is None on success."""
        results: List[Tuple[CopyTask, Optional[Exception]]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            fut_map = {ex.submit(self._copy_one, t, progress): t for t in tasks}
            for fut in as_completed(fut_map):
                t = fut_map[fut]
                try:
                    fut.result()
                    results.append((t, None))
                except Exception as e:
                    results.append((t, e))
        return results

    # ---- Internal helpers ----
    def _copy_one(self, task: CopyTask, progress: Optional[ProgressCallback]) -> None:
        src, dst = task.src_fs, task.dst_fs
        src_path, dst_path = task.src_path, task.dst_path
        key = f"{src.__class__.__name__}:{src_path} -> {dst.__class__.__name__}:{dst_path}"

        # Determine total size if available
        total: Optional[int] = None
        try:
            st = src.stat(src_path)
            total = st.size
        except Exception:
            pass

        if not task.overwrite and dst.exists(dst_path):
            raise FSError(f"Destination exists: {dst_path}")

        dst.ensure_parent_dir(dst_path)

        # Fast path: both support stream open
        if getattr(src, 'supports_stream_open', False) and getattr(dst, 'supports_stream_open', False):
            with src.open(src_path, 'rb') as rf, dst.open(dst_path, 'wb') as wf:
                self._pump(rf, wf, key, total, progress)
            return

        # If destination supports stream open (Local/SFTP), stream upload from a temp file
        # Else, write_bytes directly
        with self._download_to_temp(src, src_path, total, key, progress) as tmp_path:
            if getattr(dst, 'supports_stream_open', False):
                with open(tmp_path, 'rb') as rf, dst.open(dst_path, 'wb') as wf:
                    self._pump(rf, wf, key, total, progress, already_counted=True)
            else:
                # e.g., SMB: let backend read from bytes or its own streaming API
                with open(tmp_path, 'rb') as rf:
                    data = rf.read()
                dst.write_bytes(dst_path, data)
                if progress is not None:
                    progress(key, total or 0, total)

    def _pump(self, rf, wf, key: str, total: Optional[int], progress: Optional[ProgressCallback], already_counted: bool = False):
        done = 0 if not already_counted else (total or 0)
        if progress and already_counted and total is not None:
            progress(key, done, total)
        while True:
            buf = rf.read(self.chunk_size)
            if not buf:
                break
            wf.write(buf)
            done += len(buf)
            if progress is not None:
                progress(key, done, total)

    # context manager yielding a temp file path populated by download
    from contextlib import contextmanager

    @contextmanager
    def _download_to_temp(self, src_fs: PosixFileSystem, src_path: str, total: Optional[int], key: str,
                          progress: Optional[ProgressCallback]):
        # We'll stream to temp if src supports stream open; else fall back to read_bytes (may be memory-heavy for huge files)
        fd, tmp = tempfile.mkstemp(prefix='uft_', suffix='.part')
        os.close(fd)
        try:
            if getattr(src_fs, 'supports_stream_open', False):
                with src_fs.open(src_path, 'rb') as rf, open(tmp, 'wb') as wf:
                    self._pump(rf, wf, key, total, progress)
            else:
                # e.g., SMB read; use backend's own streaming API into local file if available
                try:
                    data = src_fs.read_bytes(src_path)
                    with open(tmp, 'wb') as wf:
                        wf.write(data)
                    if progress is not None:
                        progress(key, total or len(data), total)
                except Exception as e:
                    os.remove(tmp)
                    raise e
            yield tmp
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass


# -----------------------------
# Utility: mirror directories (shallow) & recursive helpers
# -----------------------------
def iter_files(fs: PosixFileSystem, root: str) -> Iterator[str]:
    """Yield file paths (POSIX) under root recursively."""
    stack = [root]
    while stack:
        cur = stack.pop()
        for name in fs.listdir(cur):
            p = str(pathlib.PurePosixPath(cur) / name)
            if fs.is_dir(p):
                stack.append(p)
            else:
                yield p


def relative_to(path: str, root: str) -> str:
    return str(pathlib.PurePosixPath(path).relative_to(pathlib.PurePosixPath(root)))


def mirror_dir(src_fs: PosixFileSystem, dst_fs: PosixFileSystem, src_root: str, dst_root: str,
               manager: Optional[FileTransferManager] = None,
               overwrite: bool = True,
               progress: Optional[ProgressCallback] = None) -> List[Tuple[CopyTask, Optional[Exception]]]:
    """Recursively copy files from src_root to dst_root, creating directories as needed.
    Returns list of (task, exception)."""
    if manager is None:
        manager = FileTransferManager()
    tasks: List[CopyTask] = []
    for fpath in iter_files(src_fs, src_root):
        rel = relative_to(fpath, src_root)
        dst_path = str(pathlib.PurePosixPath(dst_root) / rel)
        tasks.append(CopyTask(fpath, dst_path, src_fs, dst_fs, overwrite=overwrite))
    return manager.copy_many(tasks, progress=progress)


# -----------------------------
# Example usage (as a script)
# -----------------------------
if __name__ == '__main__':
    import argparse

    def progress_print(key: str, done: int, total: Optional[int]):
        if total:
            pct = (done / total) * 100
            print(f"[{key}] {done}/{total} bytes ({pct:.1f}%)")
        else:
            print(f"[{key}] {done} bytes")

    parser = argparse.ArgumentParser(description='Unified File Transfer Tool')
    parser.add_argument('src', help='Source URL (file://, sftp://, smb://) or path for file://')
    parser.add_argument('dst', help='Destination URL or path')
    parser.add_argument('--workers', type=int, default=4, help='Max workers for bulk ops')
    parser.add_argument('--mirror', action='store_true', help='Treat src and dst as directories and mirror recursively')
    parser.add_argument('--no-overwrite', action='store_true', help="Fail if destination exists")

    args = parser.parse_args()

    # If URLs omit scheme, assume file://
    def normalize(u: str) -> str:
        if '://' not in u:
            # treat as local absolute/relative path
            p = pathlib.Path(u).expanduser().resolve()
            return 'file://' + p.as_posix()
        return u

    src_url = normalize(args.src)
    dst_url = normalize(args.dst)

    src_fs, src_path = fs_from_url(src_url)
    dst_fs, dst_path = fs_from_url(dst_url)

    mgr = FileTransferManager(max_workers=args.workers)

    if args.mirror:
        # Ensure destination root exists
        dst_fs.mkdir(dst_path, parents=True, exist_ok=True)
        results = mirror_dir(src_fs, dst_fs, src_path, dst_path, manager=mgr,
                             overwrite=not args.no_overwrite, progress=progress_print)
        errors = [e for _, e in results if e]
        if errors:
            print(f"Completed with {len(errors)} error(s):")
            for t, e in results:
                if e:
                    print(f"- {t.src_path} -> {t.dst_path}: {e}")
        else:
            print("Mirror completed successfully.")
    else:
        mgr.copy(src_path, dst_path, src_fs, dst_fs, overwrite=not args.no_overwrite, progress=progress_print)
        print("Copy completed.")