#!/usr/bin/env python3
"""Простий термінальний бот: питання з клавіатури → відповідь від OpenAI."""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError


MODEL = "gpt-4o-mini-2024-07-18"
EXIT_WORDS = {"exit", "quit", "q", "вихід", "вийти"}


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print(
            "Помилка: не знайдено OPENAI_API_KEY.\n"
            "Скопіюйте .env.example у .env і додайте ключ API OpenAI.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    print(f"Бот готовий (модель: {MODEL}). Питайте що завгодно; для виходу: exit / quit / вихід.\n")

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
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=history,
            )
        except OpenAIError as e:
            history.pop()
            print(f"Помилка API: {e}", file=sys.stderr)
            continue

        reply = (completion.choices[0].message.content or "").strip()
        history.append({"role": "assistant", "content": reply})
        print(f"Бот> {reply}\n")


if __name__ == "__main__":
    main()
