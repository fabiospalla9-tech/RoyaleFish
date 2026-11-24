"""Model B: the same ~12M transformer as A, with the real unit type, real forms, spells in flight, Evo progress and ability buttons."""
import torch, torch.nn as nn

D, LAYERS, HEADS, FF = 384, 6, 8, 1536
NUM = 17            # entity numeric columns after card, unit type and form


class NetB(nn.Module):
    def __init__(self, vocab, units, g_w=25):
        super().__init__()
        self.name = nn.Embedding(vocab, 96); self.unit = nn.Embedding(units, 64); self.form = nn.Embedding(3, 16)
        self.ent = nn.Linear(96 + 64 + 16 + NUM, D)
        self.glob = nn.Linear(g_w, D)
        self.hand_name = nn.Embedding(vocab, 96); self.hand_form = nn.Embedding(3, 16)
        self.hand = nn.Linear(96 + 16 + 1, D)
        self.slot = nn.Embedding(8, D)          # hand 0..3, next card 4, ability buttons 5..7
        self.cls = nn.Parameter(torch.zeros(1, 1, D))
        self.mem_name = nn.Embedding(vocab, 96); self.mem_form = nn.Embedding(3, 16)
        self.mem = nn.Linear(96 + 16 + 4, D)
        self.mem_slot = nn.Embedding(160, D)
        layer = nn.TransformerEncoderLayer(D, HEADS, FF, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(D)
        self.card_head = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 5))
        self.pos_head = nn.Sequential(nn.Linear(2 * D, 1024), nn.GELU(), nn.Linear(1024, 18 * 32))
        self.ability_head = nn.Sequential(nn.Linear(2 * D, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, E, N, G, H, M, ready):
        """Returns tokens and 8 logits: wait, 4 hand slots, 3 ability buttons (buttons not ready are masked)."""
        B = E.shape[0]
        ent = self.ent(torch.cat([self.name(E[..., 0].long()), self.unit(E[..., 1].long()), self.form(E[..., 2].long().clamp(0, 2)), E[..., 3:]], -1))
        hand = self.hand(torch.cat([self.hand_name(H[..., 0].long()), self.hand_form(H[..., 1].long().clamp(0, 2)), H[..., 2:3].to(ent.dtype) / 100], -1)) + self.slot.weight[None]
        mem = self.mem(torch.cat([self.mem_name(M[..., 0].long()), self.mem_form(M[..., 1].long().clamp(0, 2)), M[..., 2:6]], -1))
        mem = mem + self.mem_slot.weight[None, :M.shape[1]]
        tok = torch.cat([self.cls.expand(B, 1, D) + self.glob(G)[:, None], hand, mem, ent], 1)
        ent_mask = torch.arange(E.shape[1], device=E.device)[None] >= N[:, None]
        mask = torch.cat([torch.zeros(B, 9, dtype=torch.bool, device=E.device), M[..., 6] < 0.5, ent_mask], 1)
        h = self.norm(self.enc(tok, src_key_padding_mask=mask))
        card = self.card_head(h[:, 0]).float()
        ab = self.ability_head(torch.cat([h[:, 0, None].expand(-1, 3, -1), h[:, 6:9]], -1)).squeeze(-1).float()
        rm = torch.stack([(ready >> k) & 1 for k in range(3)], 1).bool()
        return h, torch.cat([card, ab.masked_fill(~rm, -1e4)], 1)

    def pos_logits(self, h, slot):
        hs = h[torch.arange(h.shape[0], device=h.device), 1 + slot.clamp(0, 3)]
        return self.pos_head(torch.cat([h[:, 0], hs], -1))
