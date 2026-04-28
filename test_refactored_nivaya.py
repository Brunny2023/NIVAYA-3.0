import torch
from modules.multimodal import Nivaya3Model

def test_refactored_nivaya():
    print("=== Testing Refactored Nivaya 3.0 Cognitive Engine ===")
    
    # 1. Model Initialization
    vocab_size = 1000
    embed_dim = 128
    num_layers = 2
    num_heads = 4
    num_experts = 2
    
    model = Nivaya3Model(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_experts=num_experts
    )
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")

    # 2. Multi-Modal Input Test
    print("\n[Test] Multi-Modal Input and Tool Call Generation")
    batch_size = 1
    input_ids = torch.randint(0, vocab_size, (batch_size, 8))
    image_feats = torch.randn(batch_size, 4, 768) # Simulate 4 image patches
    audio_feats = torch.randn(batch_size, 5, 128) # Simulate 5 audio segments
    
    output = model(input_ids=input_ids, image_feats=image_feats, audio_feats=audio_feats)
    
    print(f"Output Logits Shape: {output['logits'].shape}")
    tool_call_output = output['tool_call_output']
    print(f"Tool Logits Shape: {tool_call_output['tool_logits'].shape}")
    print(f"Action Logits Shape: {tool_call_output['action_logits'].shape}")
    print(f"Parameter Tokens Shape: {tool_call_output['param_tokens'].shape}")

    # Simulate a tool call decision
    predicted_tool_idx = torch.argmax(tool_call_output['tool_logits'], dim=-1).item()
    predicted_action_idx = torch.argmax(tool_call_output['action_logits'], dim=-1).item()
    
    # In a real scenario, these indices would map to a tool registry
    # and the param_tokens would be decoded into a JSON object.
    print(f"\nSimulated Tool Selection: Tool Index {predicted_tool_idx}, Action Index {predicted_action_idx}")

    print("\n=== All Refactored Nivaya 3.0 Tests Passed! ===")

if __name__ == "__main__":
    test_refactored_nivaya()
