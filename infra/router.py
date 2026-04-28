import os

class NivayaRouter:
    """
    Model Routing Layer for Nivaya 3.0.
    Optimizes cost by preferring local execution over cloud fallback.
    """
    def __init__(self):
        self.local_available = self._check_local_runtime()

    def _check_local_runtime(self):
        """
        Checks if a local runtime (like Ollama or local PyTorch) is available.
        """
        # Simulated check for local environment variables or service ports
        return os.getenv("NIVAYA_LOCAL_ENABLED", "true").lower() == "true"

    def route_execution(self, task_input):
        """
        Determines whether to run the task locally or fallback to cloud.
        """
        if self.local_available:
            print("[Router] Routing to LOCAL execution (Cost: $0)")
            return "local"
        else:
            print("[Router] Routing to CLOUD fallback (Cost: $$$)")
            return "cloud"

    def execute(self, task_input, model_instance):
        """
        Executes the task based on the routing decision.
        """
        mode = self.route_execution(task_input)
        
        if mode == "local":
            # Use the local persistent model instance
            return model_instance.run_autonomous(task_input)
        else:
            # Simulate cloud API fallback
            return {"status": "success", "source": "cloud_api", "data": "Cloud-processed result"}
