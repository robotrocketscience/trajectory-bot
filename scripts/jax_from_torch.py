#!/usr/bin/env python3
"""Convert a torch Policy state_dict (13-128-128-4 tanh) to JAX params npz.
torch Linear weight is (out,in) and does x@W.T+b; JAX policy does x@w+b, so w = W.T."""
import sys, numpy as np, torch
ckpt = sys.argv[1] if len(sys.argv) > 1 else "models/circularize3d_dagger.pt"
out = sys.argv[2] if len(sys.argv) > 2 else "models/dagger_jax.npz"
sd = torch.load(ckpt, map_location="cpu")
print("state_dict keys:", list(sd.keys()))
def g(k): return sd[k].detach().numpy()
np.savez(out,
         w0=g("net.0.weight").T, b0=g("net.0.bias"),
         w1=g("net.2.weight").T, b1=g("net.2.bias"),
         w2=g("net.4.weight").T, b2=g("net.4.bias"))
print(f"wrote {out}: w0{g('net.0.weight').T.shape} w1{g('net.2.weight').T.shape} "
      f"w2{g('net.4.weight').T.shape}")
