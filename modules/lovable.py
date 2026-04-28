import time
import uuid

class LovableExecutionLayer:
    """
    Simulates the 'Lovable' execution layer for Nivaya 3.0.
    Adheres to the Nivaya Tool Interface Standard (NTIS) for responses.
    """
    def __init__(self, tool_registry):
        self.tool_registry = tool_registry

    def execute(self, tool_call):
        """
        Processes a tool call and returns an NTIS-compliant response.
        """
        task_id = tool_call.get("task_id", str(uuid.uuid4()))
        step_id = tool_call.get("step_id", "0")
        tool_name = tool_call.get("tool")
        action = tool_call.get("action")
        
        start_time = time.time()
        
        # Validate tool existence
        if tool_name not in self.tool_registry.tools:
            return self._format_response(task_id, step_id, "failed", 
                                       error={"code": "TOOL_NOT_FOUND", "message": f"Tool '{tool_name}' not registered."})

        # Simulate execution
        try:
            # In a real scenario, this would call the actual tool implementation
            result = {"message": f"Successfully executed {action} on {tool_name}"}
            status = "success"
            error = None
        except Exception as e:
            status = "failed"
            result = {}
            error = {"code": "EXECUTION_ERROR", "message": str(e)}

        execution_time = (time.time() - start_time) * 1000 # ms
        cost = 0.005 # Mock cost

        return self._format_response(
            task_id, step_id, status, 
            result=result, error=error, 
            execution_time=execution_time, cost=cost
        )

    def _format_response(self, task_id, step_id, status, result=None, error=None, execution_time=0, cost=0):
        response = {
            "task_id": task_id,
            "step_id": step_id,
            "status": status,
            "result": result or {},
            "error": error or {"code": "", "message": ""},
            "metrics": {
                "execution_time": execution_time,
                "cost": cost
            }
        }
        return response
