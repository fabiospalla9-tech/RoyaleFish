"""Converte le righe impacchettate di teacher_pack (uint8 [N, total]) nel formato 'row-packed' dell'allenamento
originale di Everlight (ReceivedRowPackedTensorRecord): un buffer per tipo di dato. Cosi' distributed_ppo usa la
via veloce (PackedRolloutDeviceCacheV4) invece di ricomporre i dati mossa per mossa."""
from dataclasses import fields, is_dataclass

import torch

from native_runner.training.v4.remote_worker_transport import ReceivedRowPackedTensorRecord

_DT = {str(d): d for d in (torch.float32, torch.float16, torch.bfloat16, torch.float64, torch.int64, torch.int32, torch.int16,
                           torch.int8, torch.uint8, torch.bool)}


def build_row_packed(template, items, raw):
    """template: record V4 con batch 1; items: layout di teacher_pack; raw: torch.uint8 [N, total] (N = T*B, ordine tempo-major)."""
    rows = raw.shape[0]
    it = iter(items)
    cols = {}          # dtype -> lista di (offset byte, byte)
    width = {}         # dtype -> elementi gia' assegnati

    def walk(value):
        if isinstance(value, torch.Tensor):
            path, dtype, shape, off, nb = next(it)
            dt = _DT[dtype]
            size = torch.empty((), dtype=dt).element_size()
            n = nb // size
            start = width.get(dt, 0); width[dt] = start + n
            cols.setdefault(dt, []).append((off, nb))
            return ('tensor', dt, tuple(shape), start, n)
        if is_dataclass(value):
            return ('dataclass', type(value), tuple((f.name, walk(getattr(value, f.name))) for f in fields(value)))
        return ('value', value)

    schema = walk(template)
    buffers = {}
    for dt, parts in cols.items():
        nz = [raw[:, off:off + nb] for off, nb in parts if nb]
        if not nz:
            buffers[dt] = torch.zeros((rows, 0), dtype=dt)
            continue
        buffers[dt] = torch.cat(nz, dim=1).contiguous().view(dt)
    return ReceivedRowPackedTensorRecord(schema, rows, tuple(buffers), {d: d for d in buffers}, buffers)
