#!/usr/bin/env python3
"""Термінальний чат через OpenAI API, але тільки через requests."""

import os
import sys
from typing import Any

import requests


MODEL = "gpt-4o-mini-2024-07-18"
API_URL = "https://api.openai.com/v1/chat/completions"
EXIT_WORDS = {"exit", "quit", "q", "вихід", "вийти"}

TEMPERATURE = 0.7
TOP_P = 1.0
N_COMPLETIONS = 1
MAX_TOKENS: int | None = 1024
PRESENCE_PENALTY = 0.0
FREQUENCY_PENALTY = 0.0
STOP_SEQUENCES: list[str] | None = None
SEED: int | None = None
LOGIT_BIAS: dict[str, int] = {}
USER_STABLE_ID: str | None = None
LOGPROBS = False
TOP_LOGPROBS = 0
REQUEST_TIMEOUT_SEC = 60


def _request_payload(messages: list[dict[str, str]]) -> dict[str, Any]:
    """Збирає JSON для /chat/completions; опускає None та порожні опції."""
    data: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "n": N_COMPLETIONS,
        "presence_penalty": PRESENCE_PENALTY,
        "frequency_penalty": FREQUENCY_PENALTY,
    }
    if MAX_TOKENS is not None:
        data["max_tokens"] = MAX_TOKENS
    if STOP_SEQUENCES:
        data["stop"] = STOP_SEQUENCES
    if SEED is not None:
        data["seed"] = SEED
    if LOGIT_BIAS:
        data["logit_bias"] = LOGIT_BIAS
    if USER_STABLE_ID:
        data["user"] = USER_STABLE_ID
    if LOGPROBS:
        data["logprobs"] = True
        if TOP_LOGPROBS > 0:
            data["top_logprobs"] = TOP_LOGPROBS
    return data


def _call_chat_completions(
    api_key: str, messages: list[dict[str, str]]
) -> tuple[str | None, str | None]:
    """Повертає (reply, error). У випадку помилки reply=None."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = _request_payload(messages)

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT_SEC,
        )
    except requests.RequestException as e:
        return None, f"Помилка мережі/API: {e}"

    if response.status_code >= 400:
        try:
            err_json = response.json()
            err_message = err_json.get("error", {}).get("message", response.text)
        except ValueError:
            err_message = response.text
        return None, f"HTTP {response.status_code}: {err_message}"

    try:
        data = response.json()
    except ValueError:
        return None, "Відповідь API не є валідним JSON."

    try:
        reply = (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError):
        return None, "Несподіваний формат відповіді API."
    return reply, None


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print(
            "Помилка: не знайдено OPENAI_API_KEY.\n"
            "Встановіть змінну середовища, наприклад:\n"
            "export OPENAI_API_KEY='ваш_ключ'",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Бот готовий (модель: {MODEL}). Питайте що завгодно; для виходу: exit / quit / вихід.\n"
    )
    history: list[dict[str, str]] = []

    while True:
        try:
            line = input("Ви> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо побачення.")
            break

        if not line:
            continue
        if line.lower() in EXIT_WORDS:
            print("До побачення.")
            break

        history.append({"role": "user", "content": line})
        reply, err = _call_chat_completions(api_key, history)
        if err:
            history.pop()
            print(f"Помилка API: {err}", file=sys.stderr)
            continue

        history.append({"role": "assistant", "content": reply or ""})
        print(f"Бот> {reply}\n")


if __name__ == "__main__":
    main()
