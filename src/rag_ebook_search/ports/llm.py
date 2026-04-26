"""LLM port interface."""

from abc import ABC, abstractmethod


class LLMPort(ABC):
    """Port for language model operations.

    This abstract base class defines the interface that all LLM
    implementations must follow. It allows the application to swap
    LLM providers without changing business logic.
    """

    @abstractmethod
    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate a response to a prompt.

        Args:
            prompt: The prompt to generate a response for.
            temperature: Sampling temperature for generation.

        Returns:
            Generated text response.
        """
        pass

    @abstractmethod
    async def generate_with_context(self, context: str, question: str) -> str:
        """Generate an answer based on context and question.

        Args:
            context: Context information to use for generation.
            question: Question to answer.

        Returns:
            Generated answer.
        """
        pass
