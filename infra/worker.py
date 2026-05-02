import time
import json
import torch
from queue import NivayaQueue
from modules.multimodal import Nivaya3Model
from modules.planning import NivayaPlanner
from modules.orchestration import NivayaOrchestrator
from modules.simulation import NivayaSimulator
from modules.reasoning import NivayaReasoningEngine
from modules.nivaya_agent import NivayaAgent
from modules.registry import ToolRegistry

class NivayaWorker:
    """
    Persistent Worker for Nivaya 3.0.
    Loads the model once at startup and processes tasks from the queue.
    """
    def __init__(self):
        print("[Worker] Initializing Nivaya 3.0 Core...")
        # Load Model and Components ONCE
        self.registry = ToolRegistry(phase=1)
        self.model = Nivaya3Model(vocab_size=1000, embed_dim=128, num_layers=2, num_heads=4)
        self.planner = NivayaPlanner(self.registry)
        self.simulator = NivayaSimulator(self.registry)
        self.reasoning = NivayaReasoningEngine()
        self.nivaya_agent = NivayaAgent(self.registry)
        self.orchestrator = NivayaOrchestrator(self.planner, self.simulator, self.nivaya_agent, self.reasoning)
        
        self.queue = NivayaQueue()
        print("[Worker] Nivaya Core Loaded. Waiting for tasks...")

    def run(self):
        while True:
            task = self.queue.fetch_task()
            if task:
                self.process_task(task)
            else:
                time.sleep(1)

    def process_task(self, task):
        task_id = task.get("task_id")
        objective = task.get("input")
        
        print(f"[Worker] Processing Task: {task_id}")
        # Update Status to 'running' (Mock Supabase Update)
        self._update_supabase_status(task_id, "running")
        
        try:
            # Execute using the persistent orchestrator
            results = self.orchestrator.run_autonomous(objective)
            
            # Store Result (Mock Supabase Update)
            self._update_supabase_result(task_id, "completed", results)
            print(f"[Worker] Task {task_id} Completed.")
            
        except Exception as e:
            print(f"[Worker] Task {task_id} Failed: {str(e)}")
            self._update_supabase_result(task_id, "failed", {"error": str(e)})

    def _update_supabase_status(self, task_id, status):
        # In a real implementation, this would use the Supabase Python SDK
        print(f"[Mock Supabase] Task {task_id} status -> {status}")

    def _update_supabase_result(self, task_id, status, result):
        # In a real implementation, this would use the Supabase Python SDK
        print(f"[Mock Supabase] Task {task_id} status -> {status}, result stored.")

if __name__ == "__main__":
    worker = NivayaWorker()
    worker.run()
