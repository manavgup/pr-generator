"""
Configuration for LLM providers and models.
"""
import os
import logging
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class LLMProvider(str, Enum):
    """Enum for supported LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"

class LLMConfig(BaseModel):
    """Configuration for LLM providers and models."""
    provider: LLMProvider = Field(..., description="LLM provider (ollama or openai)")
    model: str = Field(..., description="Model name to use")
    api_key: Optional[str] = Field(None, description="API key for the provider")
    base_url: Optional[str] = Field(None, description="Base URL for the provider API")
    temperature: float = Field(0.7, description="Temperature for generation")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """
        Create LLMConfig from environment variables.
        
        Environment variables:
        - LLM_PROVIDER: 'ollama' or 'openai'
        - LLM_MODEL: Model name (e.g., 'gpt-4o-mini' for OpenAI, 'llama3' for Ollama)
        - LLM_API_KEY: API key (required for OpenAI)
        - LLM_BASE_URL: Base URL (optional)
        - LLM_TEMPERATURE: Temperature (default: 0.7)
        - LLM_MAX_TOKENS: Maximum tokens (optional)
        """
        provider_str = os.getenv("LLM_PROVIDER", "openai").lower()
        provider = LLMProvider.OPENAI if provider_str == "openai" else LLMProvider.OLLAMA
        
        # Default model based on provider
        default_model = "gpt-4o-mini" if provider == LLMProvider.OPENAI else "llama3"
        
        return cls(
            provider=provider,
            model=os.getenv("LLM_MODEL", default_model),
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS")) if os.getenv("LLM_MAX_TOKENS") else None
        )

    @classmethod
    def create(
        cls,
        llm_provider: str,
        llm_model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> "LLMConfig":
        """
        Create an LLM configuration object with specified parameters.
        
        Args:
            llm_provider: LLM provider (openai or ollama)
            llm_model: Model name to use
            api_key: API key for the provider
            base_url: Base URL for the provider API
            temperature: Temperature for the LLM (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLMConfig instance
        """
        # Determine provider
        provider = LLMProvider.OPENAI if llm_provider.lower() == "openai" else LLMProvider.OLLAMA
        
        # Determine model based on provider
        if not llm_model:
            model = "gpt-4o-mini" if provider == LLMProvider.OPENAI else "llama3"
        else:
            model = llm_model
        
        # Get API key from environment if not provided
        if provider == LLMProvider.OPENAI and not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OpenAI API key not provided and not found in environment")
        
        # Use default base URL if not provided
        if provider == LLMProvider.OLLAMA and not base_url:
            base_url = "http://localhost:11434"
        
        # Create configuration
        logger.info(f"Creating LLM config for {provider.value} with model {model}")
        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens
        )


def create_llm_config(
    llm_provider: str,
    llm_model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> LLMConfig:
    """
    Create an LLM configuration object.
    
    Args:
        llm_provider: LLM provider (openai or ollama)
        llm_model: Model name to use
        api_key: API key for the provider
        base_url: Base URL for the provider API
        temperature: Temperature for the LLM (0.0 to 1.0)
        max_tokens: Maximum tokens to generate
        
    Returns:
        LLM configuration
    """
    return LLMConfig.create(
        llm_provider=llm_provider,
        llm_model=llm_model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens
    )