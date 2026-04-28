import torch
from modules.multimodal import Nivaya3Model

def test_nivaya():
    print("Initializing Nivaya 3.0...")
    # Model parameters
    vocab_size = 1000
    embed_dim = 256
    num_layers = 4
    num_heads = 8
    num_experts = 4
    num_tools = 5
    
    model = Nivaya3Model(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_experts=num_experts,
        num_tools=num_tools
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters.")
    
    # Test text-only input
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print("Testing text-only forward pass...")
    output = model(input_ids=input_ids)
    
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Tool logits shape: {output['tool_logits'].shape}")
    
    # Test multi-modal input
    print("Testing multi-modal forward pass...")
    image_feats = torch.randn(batch_size, 10, 768) # 10 image patches
    audio_feats = torch.randn(batch_size, 5, 128)  # 5 audio segments
    
    output_mm = model(input_ids=input_ids, image_feats=image_feats, audio_feats=audio_feats)
    
    print(f"Multi-modal output shape: {output_mm['logits'].shape}")
    print("Nivaya 3.0 Architecture Test Passed!")

if __name__ == "__main__":
    test_nivaya()
