import json
import torch

assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()
a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
(a @ a.T).float().square().mean().backward()
assert torch.isfinite(a.grad).all()
print(json.dumps({"cuda": torch.version.cuda, "torch": torch.__version__,
                  "gpu": torch.cuda.get_device_name(), "count": torch.cuda.device_count(),
                  "bf16_forward_backward": True}), flush=True)
