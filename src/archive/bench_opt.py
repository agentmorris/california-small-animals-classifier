"""
Find a fast config for eva02_large@448 on a 4090: autocast vs pure-bf16,
grad-checkpointing, forced SDPA backend, torch.compile.
"""

#%% Imports and constants

import time
from contextlib import nullcontext
import torch
import timm
from torch.nn.attention import sdpa_kernel, SDPBackend

M = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
torch.set_float32_matmul_precision("high")
FE = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]


#%% Support functions

def run(tag, bs, grad_ckpt, mode, backends=None, compile=False, iters=12, warm=5):

    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        model = timm.create_model(M, pretrained=False, num_classes=30).cuda()
        if mode == "bf16":
            model = model.to(torch.bfloat16)
        model.train()
        if grad_ckpt:
            model.set_grad_checkpointing(True)
        if compile:
            model = torch.compile(model)
            iters, warm = 30, 22          # allow compile warmup
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        xdt = torch.bfloat16 if mode == "bf16" else torch.float32
        x = torch.randn(bs, 3, 448, 448, device="cuda", dtype=xdt)
        y = torch.randint(0, 30, (bs,), device="cuda")
        lf = torch.nn.CrossEntropyLoss()
        ctx = sdpa_kernel(backends) if backends else nullcontext()
        with ctx:
            for it in range(iters):
                if it == warm:
                    torch.cuda.synchronize(); t0 = time.time()
                opt.zero_grad(set_to_none=True)
                if mode == "autocast":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = lf(model(x), y)
                else:
                    loss = lf(model(x), y)
                loss.backward(); opt.step()
            torch.cuda.synchronize(); dt = (time.time() - t0) / (iters - warm)
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"{tag:42}: {bs/dt:5.0f} img/s ({1/dt:.2f} it/s)  peak {mem:.1f} GB")
    except RuntimeError as e:
        print(f"{tag:42}: OOM/ERR {str(e)[:45]}")
    finally:
        for v in ("model", "opt"):
            if v in dict(locals()):
                pass
        torch.cuda.empty_cache()


#%% Test execution

run("autocast + grad_ckpt           bs16", 16, True, "autocast")
run("autocast + memeff + nockpt     bs16", 16, False, "autocast", FE)
run("autocast + memeff + nockpt     bs32", 32, False, "autocast", FE)
run("bf16 + flash + nockpt          bs16", 16, False, "bf16", FE)
run("bf16 + flash + nockpt          bs32", 32, False, "bf16", FE)
run("bf16 + flash + grad_ckpt       bs48", 48, True, "bf16", FE)
run("bf16 + flash + compile + nockpt bs32", 32, False, "bf16", FE, compile=True)
