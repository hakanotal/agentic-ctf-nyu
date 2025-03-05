from .openai_backend import OpenAIBackend
from .anthropic_backend import AnthropicBackend
from .together_backend import TogetherBackend
from .gemini_backend import GeminiBackend
from .ollama_backend import OllamaBackend
from .backend import Role

BACKENDS = [OllamaBackend, OpenAIBackend, AnthropicBackend, TogetherBackend, GeminiBackend]
MODELS = {m: b for b in BACKENDS for m in b.MODELS}
