"""Does torch.compile speed up the accuracy-safe (autocast/fp32-master) config?"""
import time
import torch
import timm

M = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
torch.set_float32_matmul_precision("high")


def run(tag, bs, compile=False, mode="autocast", grad_ckpt=True):
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
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        xdt = torch.bfloat16 if mode == "bf16" else torch.float32
        x = torch.randn(bs, 3, 448, 448, device="cuda", dtype=xdt)
        y = torch.randint(0, 30, (bs,), device="cuda")
        lf = torch.nn.CrossEntropyLoss()
        iters, warm = (34, 24) if compile else (12, 5)
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
        print(f"{tag:46}: {bs/dt:5.0f} img/s ({1/dt:.2f} it/s)  peak {mem:.1f} GB")
    except RuntimeError as e:
        print(f"{tag:46}: OOM/ERR {str(e)[:45]}")
    finally:
        torch.cuda.empty_cache()


run("autocast+ckpt           bs16 (baseline)", 16)
run("autocast+ckpt+compile   bs16", 16, compile=True)
run("autocast+ckpt+compile   bs24", 24, compile=True)
run("bf16+ckpt+compile       bs32", 32, compile=True, mode="bf16")
run("bf16+ckpt+compile       bs48", 48, compile=True, mode="bf16")
