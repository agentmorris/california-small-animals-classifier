"""Which SDPA backend works on this Windows torch, and does torch.compile help?"""
import time, torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

q = torch.randn(16, 16, 1024, 64, device="cuda", dtype=torch.bfloat16)  # B,H,N,D
for name, backend in [("FLASH", SDPBackend.FLASH_ATTENTION),
                      ("MEM_EFFICIENT", SDPBackend.EFFICIENT_ATTENTION),
                      ("MATH", SDPBackend.MATH)]:
    try:
        with sdpa_kernel(backend):
            for it in range(10):
                if it == 3:
                    torch.cuda.synchronize(); t0 = time.time()
                o = F.scaled_dot_product_attention(q, q, q)
            torch.cuda.synchronize()
            print(f"SDPA {name:14}: OK  {(time.time()-t0)/7*1000:.1f} ms/call")
    except Exception as e:
        print(f"SDPA {name:14}: FAIL  {str(e)[:70]}")

# does torch.compile work here?
print("\ntorch.compile test:")
try:
    import timm
    m = timm.create_model("eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
                          pretrained=False, num_classes=30).cuda().train()
    m.set_grad_checkpointing(True)
    mc = torch.compile(m)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    x = torch.randn(16, 3, 448, 448, device="cuda")
    y = torch.randint(0, 30, (16,), device="cuda")
    lf = torch.nn.CrossEntropyLoss()
    for it in range(9):
        if it == 4:
            torch.cuda.synchronize(); t0 = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = lf(mc(x), y)
        loss.backward(); opt.step()
    torch.cuda.synchronize()
    dt = (time.time()-t0)/5
    print(f"compiled fwd+bwd bs16: {1/dt:.2f} it/s  {16/dt:.0f} img/s")
except Exception as e:
    print("torch.compile FAILED:", str(e)[:200])
