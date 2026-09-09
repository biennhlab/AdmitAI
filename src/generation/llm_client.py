import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

class LLMClient:
    """Wrapper for NVIDIA NIM (OpenAI-compatible) API"""
    
    def __init__(self, api_key: str, model: str):
        self.model = model
        # Using NVIDIA's OpenAI compatible endpoint
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )

    def generate(self, system_prompt: str, messages: List[Dict[str, str]], temperature: float = 0.3) -> str:
        """Generate a response using the provided system prompt and message history."""
        try:
            formatted_messages = [{"role": "system", "content": system_prompt}]
            formatted_messages.extend(messages)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling LLM provider: {str(e)}")
            raise RuntimeError(f"LLM Provider Error: {str(e)}")

    def generate_with_tools(self, system_prompt: str, messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> Any:
        """Not implemented for Phase 1. Will be used in Agentic Router later."""
        raise NotImplementedError("Tool calling not yet implemented for Phase 1")
