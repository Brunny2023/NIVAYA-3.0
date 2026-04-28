class ToolRegistry:
    """
    NTIS-Compliant Phased Tool Registry for Nivaya 3.0.
    """
    def __init__(self, phase=1):
        self.phase = phase
        self.tools = {}
        self._initialize_phased_tools()

    def _initialize_phased_tools(self):
        # Phase 1: Core Tools
        self.register_tool("web_tool", ["search", "browse", "extract"], {"query": "string"}, phase=1)
        self.register_tool("code_tool", ["execute_python", "lint", "format"], {"code": "string"}, phase=1)
        self.register_tool("memory_tool", ["store", "retrieve", "delete"], {"key": "string", "value": "any"}, phase=1)

        # Phase 2: Advanced Tools
        self.register_tool("device_tool", ["control", "status"], {"device_id": "string", "command": "string"}, phase=2)
        self.register_tool("wallet_tool", ["transfer", "balance"], {"amount": "number", "recipient": "string"}, phase=2)
        self.register_tool("execution_sandbox", ["create", "destroy", "run"], {"image": "string"}, phase=2)

    def register_tool(self, name, actions, schema, phase=1):
        self.tools[name] = {
            "name": name,
            "actions": actions,
            "schema": schema,
            "phase_requirement": phase
        }

    def get_available_tools(self):
        """
        Returns tools available in the current system phase.
        """
        return {k: v for k, v in self.tools.items() if v["phase_requirement"] <= self.phase}

    def is_tool_available(self, tool_name):
        tool = self.tools.get(tool_name)
        return tool is not None and tool["phase_requirement"] <= self.phase

    def set_phase(self, phase):
        self.phase = phase
        print(f"[ToolRegistry] System upgraded to Phase {phase}")
