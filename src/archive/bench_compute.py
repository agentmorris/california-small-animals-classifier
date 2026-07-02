"""
Isolate pure GPU fwd+bwd throughput for eva02_large@448 (no dataloader).
"""

#%% Imports and constants

import time, torch, timm

M = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
torch.set_float32_matmul_precision("high")


#%% Test execution

for grad_ckpt in (True, False):

    for bs in (16, 24):
        try:
            model = timm.create_model(M, pretrained=False, num_classes=30).cuda().train()
            if grad_ckpt:
                model.set_grad_checkpointing(True)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            x = torch.randn(bs, 3, 448, 448, device="cuda")
            y = torch.randint(0, 30, (bs,), device="cuda")
            lossf = torch.nn.CrossEntropyLoss()
            for it in range(12):
                if it == 4:
                    torch.cuda.synchronize(); t0 = time.time()
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = lossf(model(x), y)
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            dt = (time.time() - t0) / 8
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"grad_ckpt={grad_ckpt} bs={bs}: {1/dt:.2f} it/s  {bs/dt:.0f} img/s  "
                  f"peak {mem:.1f} GB")
        except RuntimeError as e:
            print(f"grad_ckpt={grad_ckpt} bs={bs}: {str(e)[:60]}")
        finally:
            del model, opt
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
