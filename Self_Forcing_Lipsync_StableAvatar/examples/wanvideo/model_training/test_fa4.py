import torch
import torch.nn.functional as F

print(f"PyTorch version: {torch.__version__}")

# Check available backends
from torch.nn.attention import SDPBackend
print(f"\nAvailable SDPBackend options: {[b for b in dir(SDPBackend) if not b.startswith('_')]}")

# Check if FA4 is registered (PyTorch 2.10+)
try:
    from torch.nn.attention._registry import _IMPL_REGISTRY
    print(f"\nRegistered SDPA implementations: {list(_IMPL_REGISTRY.keys())}")
except ImportError:
    print("\n_registry not available in this PyTorch version")

# Check if FA4 module loads
try:
    from torch.nn.attention import _fa4
    print("\n_fa4 module loaded successfully")
except ImportError as e:
    print(f"\n_fa4 module not available: {e}")

# Direct test of FA4 cute interface
try:
    from flash_attn.cute.interface import flash_attn_func as fa4_func
    print("flash_attn.cute.interface imported successfully")
    
    # Test FA4 directly
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        
        B, S, H, D = 2, 512, 8, 64
        q = torch.randn(B, S, H, D, device=device, dtype=dtype)
        k = torch.randn(B, S, H, D, device=device, dtype=dtype)
        v = torch.randn(B, S, H, D, device=device, dtype=dtype)
        
        print(f"\nRunning FA4 cute interface directly...")
        out = fa4_func(q, k, v, causal=True)
        print(f"FA4 Output shape: {out.shape}")
        print("FA4 cute interface works!")
        
except Exception as e:
    print(f"FA4 cute test failed: {e}")

# Standard SDPA test
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
    
    B, H, S, D = 2, 8, 512, 64
    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)
    
    print(f"\nRunning standard SDPA...")
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"SDPA Output shape: {out.shape}")