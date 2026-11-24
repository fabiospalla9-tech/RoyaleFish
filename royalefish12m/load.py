"""Loads a 12.67M RoyaleFish checkpoint. Usage: python -m royalefish12m.load models/H-minion-giant-28-v342.pt"""
import json
import sys
from pathlib import Path

import torch

from .model import NetB


def load_model(path, device="cpu"):
    """Returns the policy network in eval mode and its decision threshold tau."""
    meta = json.loads((Path(__file__).with_name("meta.json")).read_text())
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    net = NetB(len(meta["vocab"]), meta["units"], meta["g_w"]).to(device)
    net.load_state_dict({k[4:]: v.float() for k, v in checkpoint.items() if k.startswith("net.")})
    return net.eval(), float(checkpoint["tau"])


if __name__ == "__main__":
    model, tau = load_model(sys.argv[1])
    print(f"{sys.argv[1]}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, tau {tau:.4f}")
