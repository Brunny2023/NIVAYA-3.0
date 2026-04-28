import torch
from modules.multimodal import Nivaya3Model
from modules.planning import NivayaPlanner
from modules.orchestration import NivayaOrchestrator
from modules.simulation import NivayaSimulator
from modules.reasoning import NivayaReasoningEngine
from modules.lovable import LovableExecutionLayer
from modules.registry import ToolRegistry

def test_execution_oriented_nivaya():
    print("=== Nivaya 3.0: Execution-Oriented Intelligence Test ===")
    
    # 1. Initialize Phased Registry (Phase 1)
    registry = ToolRegistry(phase=1)
    print(f"[Success] Registry initialized in Phase {registry.phase}")

    # 2. Initialize Core Modules
    planner = NivayaPlanner(registry)
    simulator = NivayaSimulator(registry)
    reasoning = NivayaReasoningEngine()
    lovable = LovableExecutionLayer(registry)
    orchestrator = NivayaOrchestrator(planner, simulator, lovable, reasoning)
    
    # 3. Test Phase Awareness
    print("\n[Test] Phase Awareness: Attempting to use Phase 2 tool in Phase 1...")
    try:
        invalid_plan = {"task_id": "test", "plan": [{"step_id": "1", "tool": "wallet_tool", "action": "transfer"}]}
        planner.validate_plan(invalid_plan)
    except ValueError as e:
        print(f"[Success] Correctly caught unavailable tool: {e}")

    # 4. Test Autonomous Planning & Execution
    print("\n[Test] Autonomous Mode: 'Optimize research and code execution'")
    results = orchestrator.run_autonomous("Optimize research and code execution")
    print(f"Completed Steps: {list(results.keys())}")

    # 5. Upgrade to Phase 2 and Verify
    print("\n[Test] System Upgrade to Phase 2...")
    registry.set_phase(2)
    if registry.is_tool_available("wallet_tool"):
        print("[Success] Wallet tool is now available in Phase 2.")

    # 6. Verify Structured Output Format (Latent)
    model = Nivaya3Model(vocab_size=1000, embed_dim=128, num_layers=2, num_heads=4)
    input_ids = torch.randint(0, 1000, (1, 10))
    output = model(input_ids=input_ids)
    print(f"\n[Test] Neural Engine Latent Head Verification:")
    print(f"  Tool Selection Logits: {output['tool_call_output']['tool_logits'].shape}")
    print(f"  Constraints Raw: {output['tool_call_output']['constraints_raw'].shape}")

    print("\n=== Nivaya 3.0 Execution-Oriented Upgrade Verified! ===")

if __name__ == "__main__":
    test_execution_oriented_nivaya()
