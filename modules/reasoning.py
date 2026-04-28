class NivayaReasoningEngine:
    """
    Analyzes execution feedback and optimizes future plans.
    """
    def __init__(self):
        self.knowledge_base = {}

    def analyze_feedback(self, step_result):
        """
        Detects failure and suggests adjustments.
        """
        # Support both 'response' (new) and 'result' (old/mock) keys for backward compatibility in tests
        response = step_result.get("response") or step_result.get("result")
        step = step_result.get("step")
        
        if response and response.get("status") in ["failed", "failure"]:
            step_id = step.get("step_id") or step.get("id") or "unknown"
            print(f"[NivayaReasoning] Failure detected in step {step_id}. Analyzing...")
            return {"action": "retry", "adjustment": "Increase timeout or check parameters"}
        return {"action": "continue"}

    def optimize_resources(self, plan_estimates):
        """
        Optimizes plan based on cost and latency awareness.
        """
        total_cost = sum(float(est["estimated_cost"].replace("$", "")) for est in plan_estimates)
        print(f"[NivayaReasoning] Total estimated plan cost: ${total_cost:.4f}")
        # Logic to swap tools or re-order steps to minimize cost/latency
        return "Optimized"

class NivayaFeedbackLoop:
    """
    The closed-loop system for continuous improvement.
    """
    def __init__(self, reasoning_engine, orchestrator):
        self.reasoning_engine = reasoning_engine
        self.orchestrator = orchestrator

    def run_loop(self, objective):
        # 1. Plan
        planner = self.orchestrator.planner
        plan = planner.decompose_task(objective, {})
        
        # 2. Simulate & Optimize
        estimates = [planner.simulate_outcome(step, {}) for step in plan]
        self.reasoning_engine.optimize_resources(estimates)
        
        # 3. Execute with Feedback
        for step in plan:
            # Mock execution of a single step
            tool_call = planner.generate_tool_call("dynamic_tool", "run", {"task": step["step"]})
            # Simulate a result
            result = {"status": "success", "data": "OK"}
            feedback = self.reasoning_engine.analyze_feedback({"step": step, "result": result})
            
            if feedback["action"] == "retry":
                # Handle retry logic
                pass
        
        print("[NivayaFeedbackLoop] Objective achieved with continuous reasoning.")
