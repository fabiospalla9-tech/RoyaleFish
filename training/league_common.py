"""Shared by the league learner, workers and decision server: policy + value on NetB, batching and sampling."""
import math, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from royalefish12m.model import NetB, D

T = 0.5
WAIT_LOG = math.log(4.0)      # the imitation data kept 25% of the waits
OLD = 2                       # id of the frozen old 2.6
POOL = os.environ.get('RF_POOL', 'pool')


def grow(sd, units, rows):
    """Units new in 16.635 get their own embedding row, starting from the unit they derive from. Every other weight is untouched."""
    sd = dict(sd)
    for k in ('unit.weight', 'net.unit.weight'):
        if k in sd and sd[k].shape[0] < units:
            w = sd[k]; sd[k] = torch.cat([w, w[torch.as_tensor(rows[w.shape[0] - units:] if len(rows) != units - w.shape[0] else rows)].clone()], 0)
    return sd


class RLB(nn.Module):
    def __init__(self, vocab, units, g_w=25, tau=0.34):
        super().__init__()
        self.net = NetB(vocab, units, g_w); self.value = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 1))
        self.register_buffer('tau', torch.tensor(float(tau)))

    def forward(self, E, N, G, H, M, R, A):
        h, lg = self.net(E, N, G, H, M, R)
        lg = lg.float().clone(); lg[:, 1:5] = lg[:, 1:5].masked_fill(~A, -1e4)
        act = lg[:, 1:]; z = torch.logsumexp(act, -1) - (lg[:, 0] + WAIT_LOG); t = self.tau.float()
        x = (z - torch.log(t / (1 - t))) / T
        lp = torch.cat([F.logsigmoid(-x)[:, None], F.logsigmoid(x)[:, None] + F.log_softmax(act / T, -1)], 1)
        return h, lp, self.value(h[:, 0].detach()).squeeze(-1).float()

    def cell(self, h, slot, mask):
        return (self.net.pos_logits(h, slot).float() / T).masked_fill(~mask, -1e9)


def own_mask(device):
    m = torch.zeros(576, dtype=torch.bool, device=device); m[:15 * 18] = True
    return m


def cat_pad(items, key, width_axis=1):
    L = max(x[key].shape[1] for x in items); n = sum(x[key].shape[0] for x in items); first = items[0][key]
    out = np.zeros((n, L) + first.shape[2:], first.dtype); q = 0
    for x in items:
        a = x[key]; out[q:q + len(a), :a.shape[1]] = a; q += len(a)
    return out


@torch.no_grad()
def _act(model, b, j, dev, dtype):
    ne = max(1, int(b['N'][j].max())); hl = max(1, int(b['M'][j, :, 6].sum(1).max())); f = lambda a, ty: torch.as_tensor(a, device=dev).to(ty)
    E = f(b['E'][j, :ne].astype(np.float32), dtype); M = f(b['M'][j, :hl].astype(np.float32), dtype)
    h, lp, v = model(E, f(b['N'][j].astype(np.int64), torch.long), f(b['G'][j], dtype), f(b['H'][j].astype(np.int64), torch.long), M, f(b['R'][j].astype(np.int64), torch.long), f(b['A'][j], torch.bool))
    if 'X' in b and b['X'][j].any():      # first card forced at full elixir: the model still chooses which card and where
        x = f(b['X'][j], torch.bool); lp = lp.clone(); lp[x, 0] = -1e4; lp[x, 5:] = -1e4
    d = torch.distributions.Categorical(logits=lp, validate_args=False); c = d.sample(); logp = d.log_prob(c); cell = torch.zeros_like(c); play = (c >= 1) & (c <= 4)
    if play.any():
        i = play.nonzero().squeeze(1); sl = c[i] - 1; S = f(b['S'][j], torch.bool); mask = own_mask(dev)[None].expand(len(i), -1).clone(); mask[S[i, sl]] = True
        dc = torch.distributions.Categorical(logits=model.cell(h[i], sl, mask), validate_args=False); cs = dc.sample(); cell[i] = cs; logp[i] = logp[i] + dc.log_prob(cs)
    return (c - 1).cpu().numpy(), cell.cpu().numpy(), logp.float().cpu().numpy(), v.float().cpu().numpy()


def act(model, b, dev, dtype=torch.float16, group=1024):
    n = len(b['N']); tok = b['N'].astype(np.int64) + b['M'][:, :, 6].sum(1).astype(np.int64); order = np.argsort(tok, kind='stable')
    c = np.zeros(n, np.int64); cell = np.zeros(n, np.int64); lp = np.zeros(n, np.float32); v = np.zeros(n, np.float32)
    for q in range(0, n, group):
        j = order[q:q + group]
        try:
            c[j], cell[j], lp[j], v[j] = _act(model, b, j, dev, dtype)
        except torch.cuda.OutOfMemoryError:      # too big for this GPU slice: answer it in smaller pieces
            torch.cuda.empty_cache()
            for z in range(0, len(j), 128):
                y = j[z:z + 128]; c[y], cell[y], lp[y], v[y] = _act(model, b, y, dev, dtype)
    return c, cell, lp, v


def evaluate(model, E, N, G, H, M, R, A, S, ch, cell):
    h, lp, v = model(E, N, G, H, M, R, A); logp = lp.gather(1, ch[:, None]).squeeze(1); play = (ch >= 1) & (ch <= 4); cl = None
    if play.any():
        i = play.nonzero().squeeze(1); sl = ch[i] - 1; mask = own_mask(E.device)[None].expand(len(i), -1).clone(); mask[S[i, sl]] = True
        cl = model.cell(h[i], sl, mask); logp = logp.clone(); logp[i] = logp[i] + F.log_softmax(cl, -1).gather(1, cell[i, None]).squeeze(1)
    return logp, lp, cl, v, play


def anchor_kl(l0, l1, c0, c1, play, W):
    """KL(frozen imitation model || current) on: play or wait, which action, which cell."""
    p0 = l0[:, 0].exp().clamp(1e-6, 1 - 1e-6); q0 = l1[:, 0].exp().clamp(1e-6, 1 - 1e-6)
    gate = ((p0 * (p0.log() - q0.log()) + (1 - p0) * ((1 - p0).log() - (1 - q0).log())) * W).sum() / W.sum()
    a = F.log_softmax(l0[:, 1:].clamp(min=-30), -1); b = F.log_softmax(l1[:, 1:].clamp(min=-30), -1); card = ((a.exp() * (a - b)).sum(-1) * W).sum() / W.sum(); cell = torch.zeros((), device=l0.device)
    if c0 is not None:
        x = F.log_softmax(c0.clamp(min=-1e4), -1); y = F.log_softmax(c1.clamp(min=-1e4), -1); cell = (torch.where(c0 > -1e3, x.exp() * (x - y), torch.zeros_like(x))).sum(-1).mean()
    return gate, card, cell
