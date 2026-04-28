from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from queue import NivayaQueue
import uuid

app = FastAPI(title="Nivaya 3.0 API Layer")
queue = NivayaQueue()

class TaskRequest(BaseModel):
    project_id: str
    user_id: str
    input: str

@app.post("/execute-task")
async def execute_task(request: TaskRequest):
    """
    Enqueues a task for async execution by Nivaya workers.
    """
    task_id = str(uuid.uuid4())
    task_payload = {
        "task_id": task_id,
        "project_id": request.project_id,
        "user_id": request.user_id,
        "input": request.input
    }
    
    try:
        queue.enqueue_task(task_payload)
        return {"task_id": task_id, "status": "queued"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-result/{task_id}")
async def get_result(task_id: str):
    """
    Fetches task result from Supabase (Mocked).
    """
    # In real implementation, query Supabase
    return {"task_id": task_id, "status": "completed", "result": "..."}

@app.get("/agent-status")
async def agent_status():
    """
    Returns the status of the persistent Nivaya worker.
    """
    return {"worker": "active", "queue_depth": 0}

@app.get("/workspace-sync")
async def workspace_sync(workspace_id: str):
    """
    Syncs workspace data across devices (via Supabase Realtime).
    """
    return {"workspace_id": workspace_id, "sync_status": "synced"}
