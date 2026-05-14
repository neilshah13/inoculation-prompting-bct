"""OpenRouter sampling backend.

Mirrors `ip.external.openai_driver.services` for sampling, but routes
through OpenRouter (`https://openrouter.ai/api/v1`) with the provider
pinned to Azure by default (see https://openrouter.ai/docs/guides/routing/provider-selection).

Single-key, OpenAI-compatible. No fine-tuning path — sampling only.
"""
import os
from pathlib import Path

import dotenv
import openai
from tqdm.asyncio import tqdm

from ip.llm.data_models import Chat, LLMResponse, SampleCfg
from ip.utils import fn_utils


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_PROVIDER = "azure"


def _default_provider_for_model(model_id: str) -> str | None:
    """Pick a reasonable OpenRouter provider per model family.

    Returns None to let OpenRouter route freely. Gemini family models aren't
    hosted by Azure on OpenRouter, so we deliberately fall through to None
    (OpenRouter will route to google-ai-studio or google-vertex automatically).
    Keep `azure` pinning for OpenAI models so Neil's gpt-4.1 judge path
    behaves the same as before.
    """
    if model_id.startswith("openai/"):
        return DEFAULT_PROVIDER  # azure
    return None

_client: openai.AsyncOpenAI | None = None


def _load_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    # Fall back to repo-root .env, matching the rest of the codebase.
    root = Path(__file__).resolve().parents[3]
    dotenv.load_dotenv(root / ".env")
    key = os.environ.get("OPENROUTER_API_KEY") or dotenv.dotenv_values(root / ".env").get(
        "OPENROUTER_API_KEY"
    )
    if not key:
        raise KeyError("OPENROUTER_API_KEY not set in env or .env")
    return key


def get_client() -> openai.AsyncOpenAI:
    global _client
    if _client is None:
        _client = openai.AsyncOpenAI(api_key=_load_api_key(), base_url=OPENROUTER_BASE_URL)
    return _client


@fn_utils.auto_retry_async([Exception], max_retry_attempts=5)
@fn_utils.timeout_async(timeout=120)
@fn_utils.max_concurrency_async(max_size=5)
async def sample(
    model_id: str,
    input_chat: Chat,
    sample_cfg: SampleCfg,
    provider: str | None = None,
) -> LLMResponse:
    if provider is None:
        provider = _default_provider_for_model(model_id)
    client = get_client()
    kwargs = sample_cfg.model_dump()
    extra_body = {"provider": {"only": [provider]}} if provider else None
    api_response = await client.chat.completions.create(
        messages=[m.model_dump() for m in input_chat.messages],
        model=model_id,
        extra_body=extra_body,
        **kwargs,
    )
    choice = api_response.choices[0]
    if choice.message.content is None or choice.finish_reason is None:
        raise RuntimeError(f"No content or finish reason for {model_id} via OpenRouter")

    if sample_cfg.logprobs and choice.logprobs is not None:
        logprobs = [
            {l.token: l.logprob for l in c.top_logprobs} for c in choice.logprobs.content
        ]
    else:
        logprobs = None

    return LLMResponse(
        model_id=model_id,
        completion=choice.message.content,
        stop_reason=choice.finish_reason,
        logprobs=logprobs,
    )


async def batch_sample(
    model_id: str,
    input_chats: list[Chat],
    sample_cfgs: list[SampleCfg],
    description: str | None = None,
    provider: str | None = None,
) -> list[LLMResponse]:
    return await tqdm.gather(
        *[sample(model_id, c, s, provider=provider) for c, s in zip(input_chats, sample_cfgs)],
        disable=description is None,
        desc=description,
        total=len(input_chats),
    )
