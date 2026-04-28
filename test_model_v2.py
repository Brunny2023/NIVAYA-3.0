import torch
from modules.multimodal import Nivaya3Model
from modules.sandbox import NivayaComputeFabric
from modules.vr_spatial import NivayaVRModule

def test_nivaya_v2():
    print("=== Testing Nivaya 3.0 (Enhanced) ===")
    
    # 1. Model Initialization
    vocab_size = 1000
    embed_dim = 128
    model = Nivaya3Model(vocab_size, embed_dim, 2, 4, 2, 5)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")

    # 2. Multi-Modal + VR Test
    print("\n[Test] Multi-Modal + VR Forward Pass")
    batch_size = 2
    input_ids = torch.randint(0, vocab_size, (batch_size, 8))
    image_feats = torch.randn(batch_size, 4, 768)
    vr_feats = torch.randn(batch_size, 1024, 3) # Point cloud data
    
    output = model(input_ids=input_ids, image_feats=image_feats, vr_feats=vr_feats)
    print(f"Output Logits Shape: {output['logits'].shape}")
    print(f"Agent Tool Logits Shape: {output['tool_logits'].shape}")

    # 3. VR Module Test
    print("\n[Test] VR-Specific Module")
    vr_module = NivayaVRModule(embed_dim)
    vr_out = vr_module(vr_feats)
    print(f"VR Spatial Latent Shape: {vr_out['spatial_latent'].shape}")
    print(f"VR SDF Grid Shape: {vr_out['sdf_grid'].shape}")

    # 4. Sandbox Orchestration Test
    print("\n[Test] Firecracker Sandbox Orchestration")
    fabric = NivayaComputeFabric(max_vms=2)
    worker = fabric.spawn_worker("subagent_alpha")
    if worker:
        res = worker.execute_command("python3 train_submodel.py")
        print(f"Sandbox Output: {res['stdout']}")
    fabric.shutdown_all()

    print("\n=== All Nivaya 3.0 Enhanced Tests Passed! ===")

if __name__ == "__main__":
    test_nivaya_v2()
