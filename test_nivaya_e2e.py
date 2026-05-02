import torch
import json
import traceback
from modules.multimodal import Nivaya3Model
from modules.planning import NivayaPlanner
from modules.orchestration import NivayaOrchestrator
from modules.simulation import NivayaSimulator
from modules.reasoning import NivayaReasoningEngine
from modules.nivaya_agent import NivayaAgent
from modules.registry import ToolRegistry

class NivayaE2ETester:
    def __init__(self):
        print("--- Initializing Nivaya 3.0 E2E Test Suite ---")
        self.registry = ToolRegistry(phase=1)
        self.planner = NivayaPlanner(self.registry)
        self.simulator = NivayaSimulator(self.registry)
        self.reasoning = NivayaReasoningEngine()
        self.nivaya_agent = NivayaAgent(self.registry)
        self.orchestrator = NivayaOrchestrator(self.planner, self.simulator, self.nivaya_agent, self.reasoning)
        self.model = Nivaya3Model(vocab_size=1000, embed_dim=128, num_layers=2, num_heads=4)

    def test_neural_head(self):
        print("\n[Test 1] Neural Engine Multi-Modal & Tool Head")
        try:
            input_ids = torch.randint(0, 1000, (1, 10))
            image_feats = torch.randn(1, 4, 768)
            output = self.model(input_ids=input_ids, image_feats=image_feats)
            
            assert "tool_call_output" in output, "Missing tool_call_output"
            assert output["tool_call_output"]["tool_logits"].shape == (1, 20), "Incorrect tool logits shape"
            print(">> Neural Head: PASSED")
        except Exception as e:
            print(f">> Neural Head: FAILED - {e}")
            traceback.print_exc()

    def test_phased_registry(self):
        print("\n[Test 2] Phased Tool Registry & Awareness")
        try:
            # Phase 1
            assert self.registry.is_tool_available("web_tool"), "web_tool should be available in Phase 1"
            assert not self.registry.is_tool_available("wallet_tool"), "wallet_tool should NOT be available in Phase 1"
            
            # Upgrade to Phase 2
            self.registry.set_phase(2)
            assert self.registry.is_tool_available("wallet_tool"), "wallet_tool should be available in Phase 2"
            print(">> Phased Registry: PASSED")
            self.registry.set_phase(1) # Reset for further tests
        except Exception as e:
            print(f">> Phased Registry: FAILED - {e}")

    def test_planning_and_simulation(self):
        print("\n[Test 3] Structured Planning & Simulation-First Logic")
        try:
            objective = "Analyze market trends and save to memory"
            plan_data = self.planner.decompose_task(objective)
            
            # Validate Plan Structure
            assert "task_id" in plan_data, "Missing task_id"
            assert len(plan_data["plan"]) > 0, "Plan is empty"
            
            # Validate Dependency Mapping
            for step in plan_data["plan"]:
                if step["step_id"] == "step_002":
                    assert "step_001" in step["dependencies"], "Dependency mapping failed for step_002"
            
            # Validate Simulation
            sim_result = self.simulator.simulate_plan(plan_data)
            assert "is_viable" in sim_result, "Simulation result missing viability check"
            print(">> Planning & Simulation: PASSED")
        except Exception as e:
            print(f">> Planning & Simulation: FAILED - {e}")

    def test_autonomous_orchestration_and_feedback(self):
        print("\n[Test 4] Autonomous Orchestration & Feedback Loop")
        try:
            # We need to simulate a failure to test adaptation
            # Let's mock the execution layer to fail on a specific tool
            class FailingNivayaAgent(NivayaAgent):
                def execute(self, tool_call):
                    if tool_call["tool"] == "web_tool":
                        return self._format_response(tool_call["task_id"], tool_call["step_id"], "failed", 
                                                   error={"code": "TIMEOUT", "message": "Connection timed out"})
                    return super().execute(tool_call)

            failing_agent = FailingNivayaAgent(self.registry)
            failing_orchestrator = NivayaOrchestrator(self.planner, self.simulator, failing_agent, self.reasoning)
            
            print(">> Attempting execution with simulated failure...")
            results = failing_orchestrator.run_autonomous("Test adaptation")
            
            # Check if feedback analysis was triggered
            # The current orchestrator prints failure adaptation, we verify the state
            assert len(results) < 3, "Orchestrator should have stopped or pivoted on failure"
            print(">> Orchestration & Feedback: PASSED (Logic verified)")
        except Exception as e:
            print(f">> Orchestration & Feedback: FAILED - {e}")

    def run_all(self):
        self.test_neural_head()
        self.test_phased_registry()
        self.test_planning_and_simulation()
        self.test_autonomous_orchestration_and_feedback()
        print("\n--- E2E Testing Completed ---")

if __name__ == "__main__":
    tester = NivayaE2ETester()
    tester.run_all()
