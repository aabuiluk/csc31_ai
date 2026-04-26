#!/usr/bin/env python3
"""RAG-чат: відповіді лише з наданого PDF (витяг тексту → ембеддинги → пошук → LLM)."""

from __future__ import annotations

import argparse
import io
import math
import os
import re
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog
from typing import Any

import requests
from dotenv import load_dotenv
from pypdf import PdfReader

from bot2 import API_URL, EXIT_WORDS, MAX_TOKENS, MODEL, REQUEST_TIMEOUT_SEC

load_dotenv()

# ---------------------------------------------------------------------------
# RAG / ембеддинги
# ---------------------------------------------------------------------------

EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"


def _embedding_model() -> str:
    """Модель з .env: OPENAI_EMBEDDING_MODEL (якщо немає доступу до v3 — спробуйте text-embedding-ada-002)."""
    m = (os.environ.get("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small").strip()
    return m or "text-embedding-3-small"


def _embeddings_error_hint(http_status: int, api_message: str) -> str:
    low = api_message.lower()
    hints: list[str] = []
    if http_status in (401, 403) or "permission" in low or "does not have access" in low:
        hints.append(
            "У кабінеті OpenAI перевірте: правильний API-ключ, проєкт/організацію, "
            "ліміти та чи дозволені Embeddings для цього ключа (не лише Chat)."
        )
        hints.append(
            "У .env задайте іншу модель, наприклад: OPENAI_EMBEDDING_MODEL=text-embedding-ada-002"
        )
    if "verified" in low and "organization" in low:
        hints.append("Частина моделей вимагає верифікації організації на platform.openai.com.")
    if not hints:
        return ""
    return "\n\nПідказка:\n" + "\n".join(f"• {h}" for h in hints)

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 6
RAG_TEMPERATURE = 0.2
EMBED_BATCH = 64

# --- GUI (tk, без ttk — передбачувані кольори) ---
_SURFACE = "#e8eaed"
_TEXT_BG = "#ffffff"
_TEXT_FG = "#0d0d0d"
_ACCENT_USER = "#0b57d0"
_ACCENT_BOT = "#1a1d24"
_ACCENT_ERR = "#b00020"
_ACCENT_SYS = "#047857"
_BAR_BG = "#0c4a6e"
_BAR_FG = "#ffffff"
_BTN_BG = "#dbeafe"
_BTN_FG = "#0c4a6e"
_BTN_ACTIVE = "#bfdbfe"


def _extract_with_pypdf(path: str) -> tuple[str | None, str | None]:
    """Повертає (текст або None, фатальна помилка файлу або None)."""
    try:
        reader = PdfReader(path)
    except OSError as e:
        return None, f"Не вдалося відкрити файл: {e}"
    except Exception as e:  # noqa: BLE001
        return None, f"Помилка читання PDF (pypdf): {e}"

    parts: list[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception as e:  # noqa: BLE001
            return None, f"Помилка витягу тексту зі сторінки (pypdf): {e}"
        t = t.strip()
        if t:
            parts.append(t)
    full = "\n\n".join(parts).strip()
    return (full if full else None), None


def _extract_with_pymupdf(path: str) -> tuple[str | None, str | None]:
    """PyMuPDF часто краще «бачить» текст, ніж pypdf."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return None, None
    try:
        doc = fitz.open(path)
    except Exception as e:  # noqa: BLE001
        return None, f"PyMuPDF: не вдалося відкрити PDF: {e}"
    try:
        parts: list[str] = []
        for page in doc:
            t = (page.get_text() or "").strip()
            if t:
                parts.append(t)
        full = "\n\n".join(parts).strip()
        return (full if full else None), None
    finally:
        doc.close()


def _extract_with_ocr(path: str) -> tuple[str | None, str | None]:
    """Розпізнавання сканів (потрібні: pip install pymupdf pytesseract pillow і програма Tesseract на ПК)."""
    try:
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image
    except ImportError:
        return None, None
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return (
            None,
            "OCR: не знайдено Tesseract. macOS: brew install tesseract tesseract-lang; "
            "Windows: встановіть з https://github.com/UB-Mannheim/tesseract/wiki",
        )

    try:
        doc = fitz.open(path)
    except Exception as e:  # noqa: BLE001
        return None, f"OCR: не вдалося відкрити PDF: {e}"

    parts: list[str] = []
    try:
        for page in doc:
            plain = (page.get_text() or "").strip()
            if plain:
                parts.append(plain)
                continue
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                try:
                    ocr = pytesseract.image_to_string(img, lang="ukr+eng")
                except Exception:  # noqa: BLE001 — якщо немає ukr traineddata
                    ocr = pytesseract.image_to_string(img, lang="eng")
            except Exception as e:  # noqa: BLE001
                return None, f"OCR помилка на сторінці: {e}"
            ocr = ocr.strip()
            if ocr:
                parts.append(ocr)
    finally:
        doc.close()

    full = "\n\n".join(parts).strip()
    return (full if full else None), None


def _extract_pdf_text(path: str) -> tuple[str | None, str | None]:
    """Повертає (текст, помилка). Порядок: pypdf → PyMuPDF → OCR (якщо текстового шару немає)."""
    t1, err_fatal = _extract_with_pypdf(path)
    if err_fatal:
        return None, err_fatal

    t2, err_fitz = _extract_with_pymupdf(path)
    if err_fitz and not t1 and not t2:
        return None, err_fitz

    candidates = [x for x in (t1, t2) if x]
    if candidates:
        best = max(candidates, key=len)
        if best.strip():
            return best.strip(), None

    t3, err_ocr = _extract_with_ocr(path)
    if t3 and t3.strip():
        return t3.strip(), None

    tail = ""
    if err_ocr and "Tesseract" in err_ocr:
        tail = f" {err_ocr}"
    elif err_ocr:
        tail = f" OCR: {err_ocr}"

    return (
        None,
        "У PDF не знайдено тексту для індексації (скан без OCR або порожній файл)."
        f"{tail} Для сканів: встановіть Tesseract і пакети pymupdf pytesseract pillow.",
    )


def _chunk_text(text: str, max_len: int, overlap: int) -> list[str]:
    if max_len <= 0:
        return []
    step = max_len - overlap
    if step <= 0:
        step = max_len
    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        chunks.append(text[i : i + max_len])
        i += step
    return chunks


def _norm_words(text: str) -> set[str]:
    words = re.findall(r"[A-Za-zА-Яа-яІіЇїЄєҐґ0-9_]+", text.lower())
    return {w for w in words if len(w) >= 3}


def _query_hints(query: str) -> set[str]:
    """Нормалізовані підказки для технічних термінів у документах (CSS/HTML/JS тощо)."""
    q = query.lower()
    hints: set[str] = set()
    if any(k in q for k in ("колон", "стовпц", "columns", "column")):
        hints.update({"grid", "template", "columns", "grid-template-columns", "repeat", "1fr"})
    if any(k in q for k in ("рядк", "row", "rows")):
        hints.update({"grid", "template", "rows", "grid-template-rows", "repeat", "1fr"})
    if "grid" in q:
        hints.update({"grid-template-columns", "repeat", "gap"})
    if any(k in q for k in ("gap", "відступ", "проміж")):
        hints.update({"gap", "grid", "margin", "padding"})
    return hints


def _keyword_overlap_score(query: str, chunk: str) -> float:
    q = _norm_words(query)
    q.update(_query_hints(query))
    if not q:
        return 0.0
    c = _norm_words(chunk)
    if not c:
        return 0.0
    inter = len(q & c)
    return inter / (len(q) + 0.25 * len(c))


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = 0.0
    sa = 0.0
    sb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        sa += x * x
        sb += y * y
    if sa <= 0.0 or sb <= 0.0:
        return 0.0
    return dot / (math.sqrt(sa) * math.sqrt(sb))


def _embed_batch(api_key: str, texts: list[str]) -> tuple[list[list[float]] | None, str | None]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {"model": _embedding_model(), "input": texts}
    try:
        response = requests.post(
            EMBEDDINGS_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT_SEC,
        )
    except requests.RequestException as e:
        return None, f"Помилка мережі (embeddings): {e}"

    if response.status_code >= 400:
        try:
            err_json = response.json()
            err_message = err_json.get("error", {}).get("message", response.text)
        except ValueError:
            err_message = response.text
        return None, (
            f"HTTP {response.status_code} (embeddings): {err_message}"
            + _embeddings_error_hint(response.status_code, err_message)
        )

    try:
        data = response.json()
    except ValueError:
        return None, "Відповідь embeddings API не є JSON."

    rows = data.get("data")
    if not isinstance(rows, list) or not rows:
        return None, "Несподіваний формат відповіді embeddings."

    by_index: dict[int, list[float]] = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        emb = item.get("embedding")
        if isinstance(idx, int) and isinstance(emb, list) and emb and isinstance(emb[0], (int, float)):
            by_index[idx] = [float(x) for x in emb]

    ordered: list[list[float]] = []
    for i in range(len(texts)):
        vec = by_index.get(i)
        if vec is None:
            return None, "Неповні ембеддинги в відповіді API."
        ordered.append(vec)
    return ordered, None


def _embed_all_chunks(api_key: str, chunks: list[str]) -> tuple[list[list[float]] | None, str | None]:
    all_vecs: list[list[float]] = []
    for start in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[start : start + EMBED_BATCH]
        vecs, err = _embed_batch(api_key, batch)
        if err:
            return None, err
        assert vecs is not None
        all_vecs.extend(vecs)
    return all_vecs, None


def _retrieve_indices(
    query: str,
    query_emb: list[float],
    chunks: list[str],
    chunk_embs: list[list[float]],
    k: int,
) -> tuple[list[int], str]:
    k = min(k, len(chunk_embs))
    scored: list[tuple[int, float, float, float]] = []
    for i, emb in enumerate(chunk_embs):
        sem = _cosine_sim(query_emb, emb)
        lex = _keyword_overlap_score(query, chunks[i])
        # Якщо є сильний лексичний збіг (CSS/коди), підсилюємо його.
        lex_boost = 0.35 if lex > 0.09 else 0.0
        total = 0.70 * sem + 0.30 * lex + lex_boost
        scored.append((i, total, sem, lex))
    scored.sort(key=lambda t: t[1], reverse=True)

    # Беремо трохи ширше вікно кандидатів та додаємо сусідні чанки для контексту.
    head = scored[: max(k + 4, k)]
    idx_set: set[int] = set()
    for i, *_ in head:
        idx_set.add(i)
        if i - 1 >= 0:
            idx_set.add(i - 1)
        if i + 1 < len(chunks):
            idx_set.add(i + 1)

    ordered = sorted(idx_set, key=lambda idx: next((s[1] for s in scored if s[0] == idx), -1.0), reverse=True)
    final = ordered[: max(k + 2, k)]
    debug_rows = []
    for idx in final[:5]:
        row = next((s for s in scored if s[0] == idx), None)
        if row is None:
            continue
        debug_rows.append(f"#{idx} total={row[1]:.3f} sem={row[2]:.3f} lex={row[3]:.3f}")
    return final, "; ".join(debug_rows)


def _build_system_prompt(context_blocks: list[str]) -> str:
    joined = "\n\n---\n\n".join(context_blocks)
    return (
        "Ти асистент з відповідями суворо за наданим КОНТЕКСТОМ з PDF-документа.\n"
        "Правила:\n"
        "- Використовуй лише факти та формулювання, які прямо або логічно випливають з КОНТЕКСТУ.\n"
        "- Не додавай знання ззовні документа, не вигадуй деталей.\n"
        "- Якщо у контексті є приклад/шаблон коду, дозволено робити прямі адаптації цього шаблону "
        "(наприклад, змінити 12 колонок на 5), але без нових не підтверджених технологій.\n"
        "- Перш ніж казати, що нічого не знайдено, перевір чи є в контексті близький приклад коду/патерн, який можна адаптувати.\n"
        "- Якщо в контексті справді немає достатньої інформації для відповіді — напиши чітко, що в документі цього не знайдено.\n"
        "- Відповідай мовою запитання користувача (якщо запитання українською — відповідай українською).\n\n"
        f"КОНТЕКСТ:\n{joined}"
    )


def _call_chat_rag(api_key: str, messages: list[dict[str, str]]) -> tuple[str | None, str | None]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "temperature": RAG_TEMPERATURE,
        "top_p": 1.0,
        "n": 1,
    }
    if MAX_TOKENS is not None:
        payload["max_tokens"] = MAX_TOKENS

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


class RagPdfApp(tk.Tk):
    """Вікно: обрати PDF у діалозі → індексація → чат лише за документом."""

    def __init__(
        self,
        api_key: str,
        *,
        initial_pdf: str | None = None,
        top_k: int = TOP_K,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        super().__init__()
        self.title(f"RAG по PDF ({MODEL})")
        self.minsize(560, 480)
        try:
            self.configure(background=_SURFACE)
        except tk.TclError:
            pass

        self._api_key = api_key
        self._top_k = top_k
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        self._chunks: list[str] = []
        self._chunk_embs: list[list[float]] = []
        self._rag_history: list[dict[str, str]] = []
        self._busy = False

        self._tag_user = "user"
        self._tag_bot = "bot"
        self._tag_err = "err"
        self._tag_sys = "sys"
        self._tag_plain = "plain"

        bar_font = tkfont.Font(self, size=12, weight="bold")
        btn_font = tkfont.Font(self, size=11, weight="bold")

        top = tk.Frame(self, bg=_BAR_BG, height=52, highlightthickness=0)
        top.pack(side=tk.TOP, fill=tk.X)
        top.pack_propagate(False)

        tk.Label(
            top,
            text="Чат лише за PDF — оберіть файл:",
            bg=_BAR_BG,
            fg=_BAR_FG,
            font=bar_font,
            anchor="w",
            padx=12,
            highlightthickness=0,
        ).pack(side=tk.LEFT, fill=tk.Y)

        self._load_btn = tk.Button(
            top,
            text="Обрати PDF…",
            command=self._on_pick_pdf,
            bg=_BTN_BG,
            fg=_BTN_FG,
            activebackground=_BTN_ACTIVE,
            activeforeground=_BTN_FG,
            font=btn_font,
            relief=tk.RAISED,
            padx=12,
            pady=6,
            highlightthickness=0,
        )
        self._load_btn.pack(side=tk.RIGHT, padx=12, pady=8)

        self._status_var = tk.StringVar(value="Документ не завантажено. Натисніть «Обрати PDF…».")
        tk.Label(
            self,
            textvariable=self._status_var,
            bg=_SURFACE,
            fg=_TEXT_FG,
            font=tkfont.Font(self, size=10),
            anchor="w",
            padx=12,
            pady=4,
            highlightthickness=0,
        ).pack(side=tk.TOP, fill=tk.X)

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
        self._chat.tag_configure(self._tag_user, foreground=_ACCENT_USER, background=_TEXT_BG)
        self._chat.tag_configure(self._tag_bot, foreground=_ACCENT_BOT, background=_TEXT_BG)
        self._chat.tag_configure(self._tag_err, foreground=_ACCENT_ERR, background=_TEXT_BG)
        self._chat.tag_configure(self._tag_sys, foreground=_ACCENT_SYS, background=_TEXT_BG)
        self._chat.tag_configure(self._tag_plain, foreground=_TEXT_FG, background=_TEXT_BG)

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

        bottom = tk.Frame(self, bg=_SURFACE, highlightthickness=0)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 10))

        tk.Label(
            bottom,
            text="Питання (лише за завантаженим PDF):",
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
            font=btn_font,
            relief=tk.RAISED,
            padx=14,
            pady=6,
            highlightthickness=0,
        )
        self._send_btn.pack(side=tk.RIGHT, anchor=tk.N)

        self._input.bind("<Control-Return>", lambda e: (self._on_send(), "break"))
        self._input.bind("<Meta-Return>", lambda e: (self._on_send(), "break"))

        self._send_btn.configure(state=tk.DISABLED)
        self.protocol("WM_DELETE_WINDOW", self.destroy)

        if initial_pdf and os.path.isfile(initial_pdf):
            self.after(100, lambda p=initial_pdf: self._start_index_pdf(p))

    def _append_chat(self, who: str, text: str, tag: str | None) -> None:
        self._chat.configure(state=tk.NORMAL)
        if tag in (self._tag_user, self._tag_bot, self._tag_err, self._tag_sys):
            who_tag = tag
        else:
            who_tag = self._tag_plain
        self._chat.insert(tk.END, who, who_tag)
        body_tag = tag if tag else self._tag_plain
        self._chat.insert(tk.END, text + "\n\n", body_tag)
        self._chat.configure(state=tk.DISABLED)
        self._chat.see(tk.END)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        st = tk.DISABLED if busy else tk.NORMAL
        self._load_btn.configure(state=st)
        can_send = (not busy) and bool(self._chunks)
        self._send_btn.configure(state=tk.NORMAL if can_send else tk.DISABLED)

    def _on_pick_pdf(self) -> None:
        if self._busy:
            return
        path = filedialog.askopenfilename(
            parent=self,
            title="Оберіть PDF-документ",
            filetypes=[("PDF", "*.pdf"), ("Усі файли", "*.*")],
        )
        if not path:
            return
        self._start_index_pdf(path)

    def _start_index_pdf(self, path: str) -> None:
        if self._busy:
            return
        self._set_busy(True)
        self._status_var.set(
            f"Індексація… {os.path.basename(path)} · ембеддинги: {_embedding_model()}"
        )
        self.update_idletasks()

        def work() -> None:
            raw, err = _extract_pdf_text(path)
            if err:
                self.after(0, lambda e=err: self._finish_index(False, error=e))
                return
            chunks = _chunk_text(raw, self._chunk_size, self._chunk_overlap)
            if not chunks:
                self.after(
                    0,
                    lambda: self._finish_index(
                        False, error="Не вдалося розбити документ на фрагменти."
                    ),
                )
                return
            embs, err2 = _embed_all_chunks(self._api_key, chunks)
            if err2:
                self.after(0, lambda e=err2: self._finish_index(False, error=e))
                return

            base = os.path.basename(path)
            self.after(
                0,
                lambda: self._finish_index(
                    True,
                    basename=base,
                    chunks=chunks,
                    embs=embs,
                    raw_len=len(raw),
                ),
            )

        threading.Thread(target=work, daemon=True).start()

    def _finish_index(
        self,
        success: bool,
        *,
        basename: str = "",
        chunks: list[str] | None = None,
        embs: list[list[float]] | None = None,
        raw_len: int = 0,
        error: str | None = None,
    ) -> None:
        self._busy = False
        if not success or error:
            self._chunks = []
            self._chunk_embs = []
            self._rag_history.clear()
            self._status_var.set("Документ не завантажено.")
            self._append_chat("Помилка: ", error or "Невідома помилка.", self._tag_err)
            self._set_busy(False)
            return

        assert chunks is not None and embs is not None
        self._chunks = chunks
        self._chunk_embs = embs
        self._rag_history.clear()
        self._status_var.set(
            f"{basename} · {len(chunks)} фрагментів · символів у тексті: {raw_len}"
        )
        preview = " ".join(chunks[0].split())[:260]
        self._append_chat(
            "Система: ",
            f"Завантажено «{basename}». Можна ставити питання лише за цим PDF.\n"
            f"Прев'ю тексту: {preview}...",
            self._tag_sys,
        )
        self._set_busy(False)

    def _on_send(self) -> None:
        if self._busy or not self._chunks:
            return
        text = self._input.get("1.0", tk.END).strip()
        if not text:
            return
        self._input.delete("1.0", tk.END)
        self._append_chat("Ви: ", text, self._tag_user)
        self._set_busy(True)

        def work() -> None:
            q_vecs, err = _embed_batch(self._api_key, [text])
            if err:
                self.after(0, lambda e=err: self._finish_chat(None, e))
                return
            assert q_vecs is not None
            query_emb = q_vecs[0]
            idxs, debug = _retrieve_indices(
                text, query_emb, self._chunks, self._chunk_embs, self._top_k
            )
            context_blocks = [self._chunks[i] for i in idxs]
            system_content = _build_system_prompt(context_blocks)
            messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
            messages.extend(self._rag_history)
            messages.append({"role": "user", "content": text})

            reply, err2 = _call_chat_rag(self._api_key, messages)
            if err2:
                self.after(0, lambda e=err2: self._finish_chat(None, e, user_line=text))
                return
            self.after(
                0, lambda r=reply, u=text, d=debug: self._finish_chat(r, None, user_line=u, debug=d)
            )

        threading.Thread(target=work, daemon=True).start()

    def _finish_chat(
        self,
        reply: str | None,
        error: str | None,
        *,
        user_line: str = "",
        debug: str = "",
    ) -> None:
        self._busy = False
        if error:
            self._append_chat("Помилка: ", error, self._tag_err)
            self._set_busy(False)
            return
        assert reply is not None
        self._rag_history.append({"role": "user", "content": user_line})
        self._rag_history.append({"role": "assistant", "content": reply})
        self._append_chat("Бот: ", reply, self._tag_bot)
        if debug:
            self._append_chat("Debug retrieval: ", debug, self._tag_sys)
        self._set_busy(False)


def _run_cli(
    api_key: str,
    pdf_path: str,
    *,
    top_k: int,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    raw_text, err = _extract_pdf_text(pdf_path)
    if err:
        print(f"Помилка: {err}", file=sys.stderr)
        sys.exit(1)

    chunks = _chunk_text(raw_text, chunk_size, chunk_overlap)
    if not chunks:
        print("Помилка: не вдалося розбити документ на фрагменти.", file=sys.stderr)
        sys.exit(1)

    print(
        f"Індексація PDF: {pdf_path}\n"
        f"  символів тексту: {len(raw_text)}, фрагментів: {len(chunks)}\n"
        f"  чат-модель: {MODEL}, ембеддинги: {_embedding_model()}\n"
        f"Питайте про вміст документа; вихід: exit / quit / вихід.\n"
    )
    preview = " ".join(chunks[0].split())[:260]
    print(f"Прев'ю витягнутого тексту: {preview}...\n")

    chunk_embs, err = _embed_all_chunks(api_key, chunks)
    if err:
        print(f"Помилка ембеддингів: {err}", file=sys.stderr)
        sys.exit(1)
    assert chunk_embs is not None

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

        q_vecs, err = _embed_batch(api_key, [line])
        if err:
            print(f"Помилка: {err}", file=sys.stderr)
            continue
        assert q_vecs is not None
        query_emb = q_vecs[0]

        idxs, debug = _retrieve_indices(line, query_emb, chunks, chunk_embs, top_k)
        context_blocks = [chunks[i] for i in idxs]
        system_content = _build_system_prompt(context_blocks)

        messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
        messages.extend(history)
        messages.append({"role": "user", "content": line})

        reply, err = _call_chat_rag(api_key, messages)
        if err:
            print(f"Помилка API: {err}", file=sys.stderr)
            continue

        history.append({"role": "user", "content": line})
        history.append({"role": "assistant", "content": reply or ""})
        print(f"Бот> {reply}\n")
        print(f"Debug retrieval: {debug}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG-чат лише за PDF: за замовчуванням вікно з вибором файлу; --cli — термінал."
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        default=None,
        help="(Необов’язково) шлях до PDF — у графічному режимі відкриється одразу з цим файлом",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Термінальний режим (потрібен шлях до PDF у аргументі).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        metavar="K",
        help=f"Фрагментів у промпті (за замовчуванням {TOP_K}).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Макс. довжина фрагмента в символах (за замовчуванням {CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help=f"Перетин між фрагментами (за замовчуванням {CHUNK_OVERLAP}).",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print(
            "Помилка: не знайдено OPENAI_API_KEY.\n"
            "Додайте ключ у .env (як для bot2).",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.cli:
        if not args.pdf:
            parser.error("У режимі --cli потрібно вказати шлях до PDF.")
        if not os.path.isfile(args.pdf):
            print(f"Помилка: файл не знайдено: {args.pdf}", file=sys.stderr)
            sys.exit(1)
        _run_cli(
            api_key,
            args.pdf,
            top_k=args.top_k,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        return

    if args.pdf and not os.path.isfile(args.pdf):
        print(f"Помилка: файл не знайдено: {args.pdf}", file=sys.stderr)
        sys.exit(1)

    app = RagPdfApp(
        api_key,
        initial_pdf=args.pdf,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    app.mainloop()


if __name__ == "__main__":
    main()
