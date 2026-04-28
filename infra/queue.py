import redis
import json
import os

class NivayaQueue:
    """
    Redis-based Task Queue for Nivaya 3.0.
    Ensures async execution and cost optimization.
    """
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis = redis.Redis(host=host, port=port, db=db)
        self.queue_name = "nivaya_task_queue"

    def enqueue_task(self, task_payload):
        """
        Adds a task to the queue.
        task_payload should contain task_id and input.
        """
        self.redis.rpush(self.queue_name, json.dumps(task_payload))
        print(f"[Queue] Enqueued task: {task_payload.get('task_id')}")

    def fetch_task(self):
        """
        Retrieves a task from the queue.
        """
        task_data = self.redis.blpop(self.queue_name, timeout=5)
        if task_data:
            return json.loads(task_data[1])
        return None
