#!/usr/bin/env python3
"""Графічний чат через OpenAI API (requests), на основі bot2.py — tkinter."""

from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from typing import Any

import requests
from dotenv import load_dotenv

from bot2 import API_URL, MODEL, REQUEST_TIMEOUT_SEC, _request_payload

# Усі кольори задаємо на класичних tk-віджетах (tk.Label/tk.Frame), без ttk — так воно
# однаково малюється на macOS/Windows і не «з’їдає» фон/текст системною темою.

_SURFACE = "#e8eaed"  # тло вікна й рамок
_TEXT_BG = "#ffffff"
_TEXT_FG = "#0d0d0d"
_ACCENT_USER = "#0b57d0"
_ACCENT_BOT = "#1a1d24"
_ACCENT_ERR = "#b00020"

# Панель токенів: темний фон + білий текст (максимальний контраст).
_TOKEN_BAR_BG = "#0c4a6e"
_TOKEN_BAR_FG = "#ffffff"

_BTN_BG = "#dbeafe"
_BTN_FG = "#0c4a6e"
_BTN_ACTIVE = "#bfdbfe"


def _call_chat_completions(
    api_key: str, messages: list[dict[str, str]]
) -> tuple[str | None, str | None, dict[str, int] | None]:
    """Повертає (reply, error, usage). usage — поле usage з відповіді API або None."""
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
        return None, f"Помилка мережі/API: {e}", None

    if response.status_code >= 400:
        try:
            err_json = response.json()
            err_message = err_json.get("error", {}).get("message", response.text)
        except ValueError:
            err_message = response.text
        return None, f"HTTP {response.status_code}: {err_message}", None

    try:
        data: dict[str, Any] = response.json()
    except ValueError:
        return None, "Відповідь API не є валідним JSON.", None

    try:
        reply = (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError):
        return None, "Несподіваний формат відповіді API.", None

    raw_usage = data.get("usage")
    usage: dict[str, int] | None = None
    if isinstance(raw_usage, dict):
        usage = {k: int(v) for k, v in raw_usage.items() if isinstance(v, (int, float))}
    return reply, None, usage


class ChatApp(tk.Tk):
    def __init__(self, api_key: str) -> None:
        super().__init__()
        self.title(f"Чат ({MODEL})")
        self.minsize(520, 420)
        try:
            self.configure(background=_SURFACE)
        except tk.TclError:
            pass
        self._api_key = api_key
        self._history: list[dict[str, str]] = []
        self._session_tokens = 0
        self._busy = False

        _token_font = tkfont.Font(self, size=13, weight="bold")

        # Закріплена зверху панель токенів — тільки tk (не ttk), щоб фон і текст гарантовано малювались.
        top = tk.Frame(self, bg=_TOKEN_BAR_BG, height=48, highlightthickness=0)
        top.pack(side=tk.TOP, fill=tk.X)
        top.pack_propagate(False)
        self._token_var = tk.StringVar(value="Токени за сеанс: 0")
        tk.Label(
            top,
            textvariable=self._token_var,
            bg=_TOKEN_BAR_BG,
            fg=_TOKEN_BAR_FG,
            font=_token_font,
            anchor="w",
            padx=14,
            pady=12,
            highlightthickness=0,
        ).pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sep = tk.Frame(self, height=2, bg="#64748b", highlightthickness=0)
        sep.pack(side=tk.TOP, fill=tk.X)

        chat_frame = tk.Frame(self, bg=_SURFACE, highlightthickness=0)
        chat_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=6)

        self._chat = tk.Text(
            chat_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("TkTextFont", 11),
            padx=8,
            pady=8,
            bg=_TEXT_BG,
            fg=_TEXT_FG,
            insertbackground=_TEXT_FG,
            highlightthickness=1,
            highlightbackground="#c9ccd4",
            highlightcolor="#6b8cce",
            selectbackground="#c8d8f0",
            selectforeground=_TEXT_FG,
        )
        scroll = tk.Scrollbar(
            chat_frame,
            command=self._chat.yview,
            bg=_SURFACE,
            troughcolor="#cbd5e1",
            activebackground="#94a3b8",
            highlightthickness=0,
        )
        self._chat.configure(yscrollcommand=scroll.set)
        self._chat.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._tag_user = "user"
        self._tag_bot = "bot"
        self._tag_err = "err"
        self._tag_plain = "plain"
        self._chat.tag_configure(self._tag_user, foreground=_ACCENT_USER, background=_TEXT_BG)
        self._chat.tag_configure(self._tag_bot, foreground=_ACCENT_BOT, background=_TEXT_BG)
        self._chat.tag_configure(self._tag_err, foreground=_ACCENT_ERR, background=_TEXT_BG)
        self._chat.tag_configure(self._tag_plain, foreground=_TEXT_FG, background=_TEXT_BG)

        bottom = tk.Frame(self, bg=_SURFACE, highlightthickness=0)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 10))

        tk.Label(
            bottom,
            text="Ваше повідомлення:",
            bg=_SURFACE,
            fg=_TEXT_FG,
            font=tkfont.Font(self, size=11, weight="bold"),
            anchor="w",
            highlightthickness=0,
        ).pack(anchor=tk.W, pady=(0, 4))
        input_row = tk.Frame(bottom, bg=_SURFACE, highlightthickness=0)
        input_row.pack(fill=tk.BOTH, expand=True)
        self._input = tk.Text(
            input_row,
            height=3,
            wrap=tk.WORD,
            font=("TkTextFont", 11),
            bg=_TEXT_BG,
            fg=_TEXT_FG,
            insertbackground=_TEXT_FG,
            highlightthickness=1,
            highlightbackground="#c9ccd4",
            highlightcolor="#6b8cce",
            selectbackground="#c8d8f0",
            selectforeground=_TEXT_FG,
        )
        self._input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        self._send_btn = tk.Button(
            input_row,
            text="Надіслати",
            command=self._on_send,
            bg=_BTN_BG,
            fg=_BTN_FG,
            activebackground=_BTN_ACTIVE,
            activeforeground=_BTN_FG,
            font=tkfont.Font(self, size=11, weight="bold"),
            relief=tk.RAISED,
            padx=14,
            pady=6,
            highlightthickness=0,
        )
        self._send_btn.pack(side=tk.RIGHT, anchor=tk.N)

        self._input.bind("<Control-Return>", lambda e: (self._on_send(), "break"))
        self._input.bind("<Meta-Return>", lambda e: (self._on_send(), "break"))

        self.protocol("WM_DELETE_WINDOW", self._on_quit)

    def _append_chat(self, who: str, text: str, tag: str | None) -> None:
        self._chat.configure(state=tk.NORMAL)
        who_tag = tag if tag in (self._tag_user, self._tag_bot, self._tag_err) else self._tag_plain
        self._chat.insert(tk.END, who, who_tag)
        body_tag = tag if tag else self._tag_plain
        self._chat.insert(tk.END, text + "\n\n", body_tag)
        self._chat.configure(state=tk.DISABLED)
        self._chat.see(tk.END)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._send_btn.configure(state=tk.DISABLED if busy else tk.NORMAL)

    def _on_send(self) -> None:
        if self._busy:
            return
        text = self._input.get("1.0", tk.END).strip()
        if not text:
            return

        self._input.delete("1.0", tk.END)
        self._append_chat("Ви: ", text, self._tag_user)

        self._history.append({"role": "user", "content": text})
        self._set_busy(True)

        def work() -> None:
            reply, err, usage = _call_chat_completions(self._api_key, self._history)

            def finish() -> None:
                self._set_busy(False)
                if err:
                    self._history.pop()
                    self._append_chat("Помилка: ", err, self._tag_err)
                    return
                assert reply is not None
                self._history.append({"role": "assistant", "content": reply})
                if usage:
                    total = usage.get("total_tokens")
                    if total is None:
                        total = usage.get("prompt_tokens", 0) + usage.get(
                            "completion_tokens", 0
                        )
                    self._session_tokens += int(total)
                self._token_var.set(f"Токени за сеанс: {self._session_tokens}")
                self._append_chat("Бот: ", reply, self._tag_bot)

            self.after(0, finish)

        threading.Thread(target=work, daemon=True).start()

    def _on_quit(self) -> None:
        self.destroy()


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

    app = ChatApp(api_key)
    app.mainloop()


if __name__ == "__main__":
    main()
