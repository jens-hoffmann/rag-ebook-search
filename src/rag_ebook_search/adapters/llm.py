"""LangChain LLM adapter."""

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from rag_ebook_search.config import Settings
from rag_ebook_search.ports.llm import LLMPort


class LangChainLLMAdapter(LLMPort):
    """LangChain implementation of the LLMPort.

    Uses OpenAI-compatible API via LMStudio or other providers.
    """

    def __init__(self, config: Settings):
        """Initialize the LLM adapter.

        Args:
            config: Application settings.
        """
        self._config = config
        self._llm = ChatOpenAI(
            model=config.llm_model,
            base_url=config.lmstudio_base_url,
            api_key=SecretStr(config.lmstudio_api_key),
            temperature=0.7,
        )

    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate a response to a prompt.

        Args:
            prompt: The prompt to generate a response for.
            temperature: Sampling temperature for generation.

        Returns:
            Generated text response.
        """
        # Create a new instance with the specified temperature
        llm = ChatOpenAI(
            model=self._config.llm_model,
            base_url=self._config.lmstudio_base_url,
            api_key=SecretStr(self._config.lmstudio_api_key),
            temperature=temperature,
        )
        response = await llm.ainvoke(prompt)
        return str(response.content)

    async def generate_with_context(self, context: str, question: str) -> str:
        """Generate an answer based on context and question.

        Args:
            context: Context information to use for generation.
            question: Question to answer.

        Returns:
            Generated answer.
        """
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        response = await self._llm.ainvoke(prompt)
        return str(response.content)
