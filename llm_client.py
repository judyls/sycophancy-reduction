"""
Unified LLM client supporting Claude, OpenAI, and Llama (via Groq).

Provider setup:
  claude: ANTHROPIC_API_KEY in .env.local or shell env
  openai: OPENAI_API_KEY in .env.local or shell env
  llama:  GROQ_API_KEY in .env.local or shell env   (free tier at console.groq.com)
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env.local")

PROVIDER_MODELS = {
    "claude": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
    "llama":  "llama-3.3-70b-versatile",   # served via Groq
}

PROVIDER_ENV_VARS = {
    "claude": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "llama":  "GROQ_API_KEY",
}


def check_env(provider: str):
    var = PROVIDER_ENV_VARS[provider]
    if not os.environ.get(var):
        raise EnvironmentError(f"{var} not set (required for provider='{provider}')")


class LLMClient:
    def __init__(self, provider: str):
        if provider not in PROVIDER_MODELS:
            raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(PROVIDER_MODELS)}")
        self.provider = provider
        self.model = PROVIDER_MODELS[provider]
        check_env(provider)
        self._client = self._build_client()

    def _build_client(self):
        if self.provider == "claude":
            import anthropic
            return anthropic.Anthropic()
        elif self.provider == "openai":
            import openai
            return openai.OpenAI()
        elif self.provider == "llama":
            import openai as _openai
            return _openai.OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=os.environ.get("GROQ_API_KEY", ""),
            )

    def complete(self, prompt: str, max_tokens: int = 1000, retries: int = 3) -> str:
        for attempt in range(retries):
            try:
                return self._complete_once(prompt, max_tokens)
            except Exception as e:
                if attempt == retries - 1:
                    raise
                wait = 2 ** attempt * 5
                print(f"  [{self.provider}] error (attempt {attempt + 1}): {e}. Waiting {wait}s...")
                time.sleep(wait)

    def _complete_once(self, prompt: str, max_tokens: int) -> str:
        if self.provider == "claude":
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        else:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
