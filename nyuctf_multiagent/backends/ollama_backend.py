import json
import requests
from typing import List, Dict, Any, Optional
from .backend import Backend, BackendResponse
import uuid
from ollama import Client

from ..conversation import MessageRole
from ..tools import ToolCall, ToolResult

class OllamaBackend(Backend):
    
    NAME = "ollama"
    MODELS = {
        "qwq:latest": {
            "max_context": 8192,
            "cost_per_input_token": 0,
            "cost_per_output_token": 0
        },
        "hermes3:latest": {
            "max_context": 8192,
            "cost_per_input_token": 0,
            "cost_per_output_token": 0
        },
        "llama3-groq-tool-use:latest": {
            "max_context": 8192,
            "cost_per_input_token": 0,
            "cost_per_output_token": 0
        },
        "qwen2.5-coder:14b": {
            "max_context": 8192,
            "cost_per_input_token": 0,
            "cost_per_output_token": 0
        },
        "qwen2.5:14b": {
            "max_context": 8192,
            "cost_per_input_token": 0,
            "cost_per_output_token": 0
        },
        "MFDoom/deepseek-r1-tool-calling:8b": {
            "max_context": 8192,
            "cost_per_input_token": 0,
            "cost_per_output_token": 0
        }
    }
    
    def __init__(self, role: str, model: str, tools: Dict[str, Any], api_key: str, config: Dict[str, Any]):
        super().__init__(role, model, tools, config)
        self.model = model
        self.client = Client(host='http://169.226.53.98:11434')
        self.tool_schemas = [self.get_tool_schema(tool) for tool in tools.values()]
    
    @staticmethod
    def get_tool_schema(tool: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.NAME,
                "description": tool.DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {n: {"type": p[0], "description": p[1]} for n, p in tool.PARAMETERS.items()},
                    "required": list(tool.REQUIRED_PARAMETERS),
                }
            }
        }
    
    def _call_model(self, messages):
        formatted_messages = []
        for m in messages:
            if m.role == MessageRole.OBSERVATION:
                # Format observation as a tool response
                formatted_messages.append({
                    "role": "tool",
                    "name": m.tool_data.name,
                    "content": json.dumps(m.tool_data.result)
                })
            elif m.role == MessageRole.ASSISTANT:
                msg = {
                    "role": "assistant",
                    "content": ""
                }
                if m.content is not None:
                    msg["content"] = m.content
                if m.tool_data is not None:
                    # Include tool call information in the message
                    msg["tool_calls"] = [{
                        "type": "function",
                        "function": {
                            "name": m.tool_data.name,
                            "arguments": m.tool_data.arguments,
                        }
                    }]
                formatted_messages.append(msg)
            else:
                formatted_messages.append({"role": m.role.value, "content": m.content})
        
        try:
            # Call Ollama API with the formatted messages and tool schemas
            response = self.client.chat(
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
        return 0.01
    
    def send(self, messages):
        try:
            response = self._call_model(messages)
            cost = self.calculate_cost(response)
            
            # Extract content and tool calls from the response
            content = response.get("message", {}).get("content")
            tool_call = None
            
            # Check if there's a tool call in the response
            if "tool_calls" in response.get("message", {}):
                # print(response["message"]["tool_calls"])
                tool_data = response["message"]["tool_calls"][0]
                tool_call = ToolCall(
                    id=str(uuid.uuid4()),
                    name=tool_data.function.name,
                    arguments=tool_data.function.arguments
                )
            
            return BackendResponse(content=content, tool_call=tool_call, cost=cost)
        except Exception as e:
            return BackendResponse(error=f"Backend Error: {str(e)}")
    
    