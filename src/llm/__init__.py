"""LLM provider integrations with protocol-based adapter pattern."""

from .protocols import LLMProvider, Message, MessageRole
from .adapters import AnthropicAdapter, OpenRouterAdapter
from .completion import complete, complete_with_messages

__all__ = [
    # Protocols
    "LLMProvider",
    "Message",
    "MessageRole",
    # Adapters
    "AnthropicAdapter",
    "OpenRouterAdapter",
    # Convenience functions
    "complete",
    "complete_with_messages",
]
