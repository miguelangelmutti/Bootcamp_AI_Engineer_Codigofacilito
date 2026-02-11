"""Hello AI System demo entrypoint."""

from core.config import get_settings
from core.llm_client import LLMClient
from core.logger import get_logger, setup_logger


logger = get_logger(__name__)


def main() -> None:
    settings = get_settings()
    setup_logger(settings.log_level)

    user_prompt = input("Ingresa tu mensaje: ")
    try:
        client = LLMClient(
            provider=settings.llm_provider,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
        )

        result = client.chat(
            [
                {"role": "system", "content": "You are a concise and helpful assistant."},
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ]
        )

        metadata = result["metadata"]
        usage = metadata["usage"]

        print("\n=== LLM Response ===")
        print(result["response"])

        print("\n=== Metrics ===")
        print(f"provider: {metadata['provider']}")
        print(f"model: {metadata['model']}")
        print(f"prompt_tokens: {usage['prompt_tokens']}")
        print(f"completion_tokens: {usage['completion_tokens']}")
        print(f"total_tokens: {usage['total_tokens']}")
        print(f"latency_ms: {metadata['latency_ms']}")
        print(f"estimated_cost_usd: {metadata['estimated_cost_usd']}")
    except Exception as exc:
        logger.exception("Hello AI System execution failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
