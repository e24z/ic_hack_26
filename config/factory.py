"""Factory functions to create backends from configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.base import ModelBackend
    from models.validators.types import NLIBackend, Validator
    from .loader import ProfileConfig, SummarizerConfig, NLIConfig, LettuceConfig


def create_summarizer(config: SummarizerConfig) -> ModelBackend:
    """Create a summarizer backend from configuration.

    Args:
        config: Summarizer configuration

    Returns:
        ModelBackend instance (OpenRouterBackend or MockBackend)

    Raises:
        ValueError: If backend type is not supported
    """
    if config.backend == "openrouter":
        from models.backends.openrouter import OpenRouterBackend

        if not config.api_key:
            raise ValueError("OpenRouter backend requires api_key")
        if not config.model:
            raise ValueError("OpenRouter backend requires model")

        return OpenRouterBackend(
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url or "https://openrouter.ai/api/v1",
        )

    elif config.backend == "mock":
        from models.backends.mock import MockBackend

        return MockBackend()

    else:
        raise ValueError(f"Unsupported summarizer backend: {config.backend}")


def create_nli_backend(config: NLIConfig) -> NLIBackend:
    """Create an NLI backend from configuration.

    Args:
        config: NLI configuration

    Returns:
        NLIBackend instance (HTTP, direct, or mock)

    Raises:
        ValueError: If backend type is not supported or required fields are missing
    """
    if config.backend == "http":
        from models.validators.nli_http import NLIHttpBackend

        if not config.url:
            raise ValueError("HTTP NLI backend requires url")

        return NLIHttpBackend(url=config.url)

    elif config.backend == "direct":
        from models.validators.nli_direct import NLIDirectBackend

        if not config.api_key:
            raise ValueError("Direct NLI backend requires api_key")
        if not config.model:
            raise ValueError("Direct NLI backend requires model")

        return NLIDirectBackend(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url or "https://openrouter.ai/api/v1/chat/completions",
        )

    elif config.backend == "mock":
        from models.validators.nli_mock import NLIMockBackend

        return NLIMockBackend()

    else:
        raise ValueError(f"Unsupported NLI backend: {config.backend}")


def create_validator(
    nli_backend: NLIBackend,
    config: LettuceConfig,
) -> Validator:
    """Create a validator from configuration.

    Args:
        nli_backend: NLI backend to use for validation
        config: Lettuce configuration

    Returns:
        Validator instance (LettuceDetectValidator with HTTP, direct, or mock backend)

    Raises:
        ValueError: If backend type is not supported or required fields are missing
    """
    from models.validators.lettucedetect import LettuceDetectValidator

    if config.backend == "http":
        # HTTP mode - use existing HTTP backend
        if not config.url:
            raise ValueError("HTTP Lettuce backend requires url")

        return LettuceDetectValidator(
            nli_backend=nli_backend,
            base_url=config.url,
        )

    elif config.backend == "direct":
        # Direct mode - load model in-process
        from models.validators.lettuce_direct import LettuceDirectBackend

        if not config.model_path:
            raise ValueError("Direct Lettuce backend requires model_path")

        lettuce_backend = LettuceDirectBackend(model_path=config.model_path)
        return LettuceDetectValidator(
            nli_backend=nli_backend,
            lettuce_backend=lettuce_backend,
        )

    elif config.backend == "mock":
        # Mock mode - no actual detection
        from models.validators.lettuce_mock import LettuceMockBackend

        lettuce_backend = LettuceMockBackend()
        return LettuceDetectValidator(
            nli_backend=nli_backend,
            lettuce_backend=lettuce_backend,
        )

    else:
        raise ValueError(f"Unsupported Lettuce backend: {config.backend}")


def create_from_profile(
    profile: ProfileConfig,
) -> tuple[ModelBackend, NLIBackend, Validator]:
    """Create all backends from a profile configuration.

    This is the main factory function that creates a complete set of backends
    from a configuration profile.

    Args:
        profile: Profile configuration containing all backend configs

    Returns:
        Tuple of (summarizer, nli_backend, validator)

    Raises:
        ValueError: If any backend configuration is invalid
    """
    summarizer = create_summarizer(profile.summarizer)
    nli_backend = create_nli_backend(profile.nli)
    validator = create_validator(nli_backend, profile.lettuce)

    return summarizer, nli_backend, validator
