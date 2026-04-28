import json

class NivayaOrchestrator:
    """
    State-Aware Orchestrator with Feedback-Driven Adaptation.
    Controls the Lovable execution layer.
    """
    def __init__(self, planner, simulator, execution_layer, reasoning_engine):
        self.planner = planner
        self.simulator = simulator
        self.execution_layer = execution_layer
        self.reasoning = reasoning_engine
        self.state = {
            "completed_steps": {},
            "active_plan": None,
            "task_history": []
        }

    def run_autonomous(self, objective):
        print(f"[NivayaOrchestrator] Starting Autonomous Mode for: {objective}")
        
        # 1. Plan
        plan_data = self.planner.decompose_task(objective)
        self.planner.validate_plan(plan_data)
        
        # 2. Simulate
        sim_result = self.simulator.simulate_plan(plan_data)
        if not sim_result["is_viable"]:
            print("[NivayaOrchestrator] Plan simulation failed. Adjusting...")
            # In a real scenario, this would trigger re-planning
            pass
            
        self.state["active_plan"] = plan_data
        
        # 3. Execute with Feedback Loop
        for step in plan_data["plan"]:
            if self._dependencies_met(step):
                response = self._execute_step_with_retry(step)
                
                if response["status"] == "success":
                    self.state["completed_steps"][step["step_id"]] = response["result"]
                else:
                    print(f"[NivayaOrchestrator] Step {step['step_id']} failed. Adapting...")
                    self._handle_failure(step, response)
                    break # Stop or pivot
            
        return self.state["completed_steps"]

    def _dependencies_met(self, step):
        return all(dep in self.state["completed_steps"] for dep in step["dependencies"])

    def _execute_step_with_retry(self, step):
        # Build NTIS call
        tool_call = {
            "task_id": self.state["active_plan"]["task_id"],
            "step_id": step["step_id"],
            "tool": step["tool"],
            "action": step["action"],
            "parameters": step["parameters"],
            "constraints": {"max_cost": 0.5, "timeout": 30, "requires_approval": False}
        }
        
        print(f"[NivayaOrchestrator] Dispatching {step['tool']}:{step['action']}...")
        return self.execution_layer.execute(tool_call)

    def _handle_failure(self, step, response):
        # Feedback-driven adaptation
        analysis = self.reasoning.analyze_feedback({"step": step, "response": response})
        if analysis["action"] == "retry":
            print(f"[NivayaOrchestrator] Strategy adjustment: {analysis['adjustment']}")
            # Logic to modify the active plan and retry
        else:
            print("[NivayaOrchestrator] Critical failure. Escalating.")
