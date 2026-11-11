"""Impacchettamento ultra-rapido dei dati V4 in blocchi di byte a forma fissa + formato sparse lossless uint64.
Mantiene piena compatibilita' con layout/pack/views e aggiunge FastPacker + pack/unpack sparse bit-exact."""
import ctypes
from dataclasses import fields, is_dataclass
import hashlib

import numpy as np
import torch

ALIGN = 8


def _leaves(value, path=''):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif is_dataclass(value):
        for f in fields(value):
            yield from _leaves(getattr(value, f.name), path + '.' + f.name)


def layout(value):
    """[(percorso, dtype, forma senza la riga, offset, byte)], byte totali per riga, impronta"""
    out = []; off = 0
    for path, t in _leaves(value):
        if t.shape[0] != 1:
            raise ValueError(f'riga attesa 1, trovata {t.shape[0]} in {path}')
        nb = t[0].numel() * t.element_size()
        out.append((path, str(t.dtype), tuple(t.shape[1:]), off, nb))
        off += (nb + ALIGN - 1) // ALIGN * ALIGN
    total = (off + ALIGN - 1) // ALIGN * ALIGN
    sig = hashlib.sha1(repr([(p, d, s) for p, d, s, _, _ in out]).encode()).hexdigest()[:16]
    return out, total, sig


def pack(value, lay=None):
    """-> (impronta, bytes) per una riga (compatibilita')"""
    lay = lay or layout(value)
    items, total, sig = lay
    buf = np.zeros(total, np.uint8)
    ptr = buf.ctypes.data
    for (_, _, _, off, nb), (_, t) in zip(items, _leaves(value)):
        if nb:
            tc = t.detach()
            if not tc.is_contiguous():
                tc = tc.contiguous()
            if tc.device.type != 'cpu':
                tc = tc.cpu()
            ctypes.memmove(ptr + off, tc.data_ptr(), nb)
    return sig, buf.tobytes()


class FastPacker:
    """Impacchettatore compilato zero-overhead per un singolo thread/processo."""
    def __init__(self, template):
        self.lay = layout(template)
        self.items, self.total, self.sig = self.lay
        if self.total % 8 != 0 or (self.total // 8) > 65535:
            raise ValueError(f'dimensione riga {self.total} non compatibile con indici uint16x8')
        spans = []
        exprs = []
        for path, _, _, off, nb in self.items:
            if nb > 0:
                spans.append((off, nb))
                exprs.append('v' + path)
        self._spans = tuple(spans)
        ns = {}
        code = 'def _extract(v):\n    return (' + ', '.join(exprs) + (',' if len(exprs) == 1 else '') + ')\n'
        exec(code, ns)
        self._extract = ns['_extract']
        self._buf = np.zeros(self.total, dtype=np.uint8)
        self._ptr = self._buf.ctypes.data
        self._u64 = self._buf.view(np.uint64)

    def pack_sparse(self, value):
        """Restituisce (sig, idx_u16_bytes, val_u64_bytes) bit-exact lossless (~8 KB invece di 110-186 KB)."""
        self._u64.fill(0)
        ptr = self._ptr
        memmove = ctypes.memmove
        for (off, nb), t in zip(self._spans, self._extract(value)):
            if not t.is_contiguous():
                t = t.contiguous()
            memmove(ptr + off, t.data_ptr(), nb)
        nz = np.flatnonzero(self._u64)
        if nz.size == 0:
            return self.sig, b'', b''
        return self.sig, nz.astype(np.uint16).tobytes(), self._u64[nz].tobytes()


def unpack_sparse_into(packets, dst_u8):
    """Ricostruisce len(packets) righe direttamente dentro dst_u8 [>=N, total] (es. buffer Pinned Memory)."""
    n = len(packets)
    u64 = dst_u8[:n].view(np.uint64)
    u64.fill(0)
    frombuf = np.frombuffer
    u16_dt = np.uint16
    u64_dt = np.uint64
    for i, (ib, vb) in enumerate(packets):
        if ib:
            u64[i, frombuf(ib, u16_dt)] = frombuf(vb, u64_dt)


def pack_episode_sparse(sparse_rows):
    """Converte una lista di (ib, vb) in 3 array piatti (offsets_i32, idx_u16, val_u64) per salvataggio istantaneo."""
    counts = np.fromiter((len(ib) >> 1 for ib, _ in sparse_rows), dtype=np.int32, count=len(sparse_rows))
    offsets = np.empty(len(sparse_rows) + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    idx_u16 = np.frombuffer(b''.join(ib for ib, _ in sparse_rows), dtype=np.uint16)
    val_u64 = np.frombuffer(b''.join(vb for _, vb in sparse_rows), dtype=np.uint64)
    return offsets, idx_u16, val_u64


def unpack_episode_sparse(offsets, idx_u16, val_u64, total, out_u8=None):
    """Ricostruisce [L, total] uint8 in un singolo passaggio vettorizzato C."""
    L = len(offsets) - 1
    if out_u8 is None:
        out_u8 = np.zeros((L, total), dtype=np.uint8)
    else:
        out_u8[:L].fill(0)
    if idx_u16.size > 0:
        counts = np.diff(offsets)
        row_ids = np.repeat(np.arange(L, dtype=np.intp), counts)
        out_u8[:L].view(np.uint64)[row_ids, idx_u16] = val_u64
    return out_u8


_DT = {str(d): d for d in (torch.float32, torch.float16, torch.bfloat16, torch.float64, torch.int64, torch.int32, torch.int16,
                           torch.int8, torch.uint8, torch.bool)}


def views(template, items, buf):
    """ricostruisce il record (stessa struttura di template) come viste dentro buf [righe, byte]"""
    rows = buf.shape[0]; it = iter(items)

    def build(value):
        if isinstance(value, torch.Tensor):
            path, dtype, shape, off, nb = next(it)
            dt = _DT[dtype]
            if nb == 0:
                return torch.zeros((rows, *shape), dtype=dt, device=buf.device)
            return buf[:, off:off + nb].view(dt).view(rows, *shape)
        if is_dataclass(value):
            return type(value)(**{f.name: build(getattr(value, f.name)) for f in fields(value)})
        return value
    return build(template)
