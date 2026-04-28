class NivayaSimulator:
    """
    Simulation-First Logic for Nivaya 3.0.
    Predicts failures and optimizes plans before execution.
    """
    def __init__(self, tool_registry):
        self.tool_registry = tool_registry

    def simulate_plan(self, plan_data):
        """
        Evaluates a plan for potential risks and failures.
        """
        print(f"[NivayaSimulator] Simulating plan for Task {plan_data['task_id']}...")
        risks = []
        for step in plan_data["plan"]:
            risk = self._evaluate_step_risk(step)
            if risk["level"] == "high":
                risks.append(risk)
        
        return {
            "is_viable": len(risks) == 0,
            "risks": risks,
            "suggested_adjustments": self._generate_adjustments(risks)
        }

    def _evaluate_step_risk(self, step):
        # Simulated risk evaluation logic
        if step["tool"] == "web_tool" and "latest" in str(step["parameters"]):
            return {"step_id": step["step_id"], "level": "medium", "type": "STALE_DATA_RISK"}
        return {"step_id": step["step_id"], "level": "low", "type": "NONE"}

    def _generate_adjustments(self, risks):
        adjustments = []
        for risk in risks:
            if risk["type"] == "STALE_DATA_RISK":
                adjustments.append({"step_id": risk["step_id"], "action": "add_verification_step"})
        return adjustments
