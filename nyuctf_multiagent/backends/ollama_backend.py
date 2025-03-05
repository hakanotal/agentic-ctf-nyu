import json
import requests
from typing import List, Dict, Any, Optional
from .backend import Backend, BackendResponse
import ollama

from ..conversation import MessageRole
from ..tools import ToolCall, ToolResult

class OllamaBackend(Backend):
    
    NAME = "ollama"
    MODELS = {
        "gemma2:2b-instruct-q4_0": {
            "max_context": 2048,
            "cost_per_input_token": 0,
            "cost_per_output_token": 0
        },
        "llama3": {
            "max_context": 8192,
            "cost_per_input_token": 0,
            "cost_per_output_token": 0
        },
        "llama3:8b": {
            "max_context": 8192,
            "cost_per_input_token": 0,
            "cost_per_output_token": 0
        },
        "llama3:70b": {
            "max_context": 8192,
            "cost_per_input_token": 0,
            "cost_per_output_token": 0
        },
        "mistral": {
            "max_context": 8192,
            "cost_per_input_token": 0,
            "cost_per_output_token": 0
        },
        "mistral:7b-instruct-v0.2": {
            "max_context": 8192,
            "cost_per_input_token": 0,
            "cost_per_output_token": 0
        },
        "codellama": {
            "max_context": 16384,
            "cost_per_input_token": 0,
            "cost_per_output_token": 0
        }
    }
    
    def __init__(self, role: str, model: str, tools: Dict[str, Any], api_key: str, config: Dict[str, Any]):
        super().__init__(role, model, tools, config)
        self.model = model
        self.tool_schemas = [{"function_declarations": [self.get_tool_schema(tool) for tool in tools.values()]}]
    
    @staticmethod
    def get_tool_schema(tool: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": tool.NAME,
            "description": tool.DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {n: {"type": p[0], "description": p[1]} for n, p in tool.PARAMETERS.items()},
                "required": list(tool.REQUIRED_PARAMETERS),
            }
        }
    
    def _call_model(self, messages):
        formatted_messages = []
        for m in messages:
            if m.role == MessageRole.OBSERVATION:
                # Format observation as a tool response
                formatted_messages.append({
                    "role": "tool",
                    "content": json.dumps(m.tool_data.result),
                    "name": m.tool_data.name
                })
            elif m.role == MessageRole.ASSISTANT:
                msg = {"role": "assistant"}
                if m.content is not None:
                    msg["content"] = m.content
                if m.tool_data is not None:
                    # Include tool call information in the message
                    msg["tool_calls"] = [{
                        "name": m.tool_data.name,
                        "arguments": m.tool_data.arguments
                    }]
                formatted_messages.append(msg)
            else:
                formatted_messages.append({"role": m.role.value, "content": m.content})
        
        try:
            # Call Ollama API with the formatted messages and tool schemas
            response = ollama.chat(
                model=self.model,
                messages=formatted_messages,
                options={
                    "temperature": self.get_param(self.role, "temperature"),
                    "num_predict": self.get_param(self.role, "max_tokens")
                },
                tools=self.tool_schemas
            )
            return response
        except Exception as e:
            raise Exception(f"Ollama API error: {str(e)}")
    
    def calculate_cost(self, response):
        # Ollama is free to use locally, so cost is 0
        return 0
    
    def send(self, messages):
        try:
            response = self._call_model(messages)
            cost = self.calculate_cost(response)
            
            # Extract content and tool calls from the response
            content = response.get("message", {}).get("content")
            tool_call = None
            
            # Check if there's a tool call in the response
            if "tool_calls" in response.get("message", {}):
                tool_data = response["message"]["tool_calls"][0]
                tool_call = ToolCall(
                    name=tool_data["name"],
                    id=tool_data.get("id", "tool-call-id"),
                    arguments=tool_data["arguments"]
                )
            
            return BackendResponse(content=content, tool_call=tool_call, cost=cost)
        except Exception as e:
            return BackendResponse(error=f"Backend Error: {str(e)}")
    
    