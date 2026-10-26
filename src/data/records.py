"""Structured data records and JSONL row parsing."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ConversationMessage:
    role: str
    content: str
    answers: tuple[str, ...] = ()

@dataclass(frozen=True)
class ConversationTurn:
    """One assistant target and the messages visible before it."""

    messages: tuple[ConversationMessage, ...]
    answer: str
    answers: tuple[str, ...] = ()
    dataset: str = ""
    context_id: str = ""

    @property
    def question(self) -> str:
        for message in reversed(self.messages):
            if message.role == "user":
                return message.content
        return ""

    @property
    def references(self) -> tuple[str, ...]:
        return self.answers or ((self.answer,) if self.answer else ())

@dataclass(frozen=True)
class ContextRecord:
    context: str
    conversations: tuple[ConversationMessage, ...]
    dataset: str = ""
    context_id: str = ""
    context_token_length: int | None = None
    task: str | None = None
    independent_qa: bool = False

    def __post_init__(self) -> None:
        normalized = []
        for item in self.conversations:
            if isinstance(item, ConversationMessage):
                normalized.append(item)
            elif isinstance(item, dict):
                normalized.append(
                    ConversationMessage(
                        str(item["role"]),
                        str(item["content"]),
                        tuple(str(value) for value in item.get("answers", ()) or ()),
                    )
                )
            else:
                raise TypeError(f"unsupported conversation item: {type(item)!r}")
        object.__setattr__(self, "conversations", tuple(normalized))

    def turns(self) -> tuple[ConversationTurn, ...]:
        turns = []
        history = []
        pending_user = None
        for message in self.conversations:
            if message.role == "user":
                pending_user = message
                if self.independent_qa:
                    history = [message]
                else:
                    history.append(message)
            elif message.role == "assistant" and pending_user is not None:
                turns.append(
                    ConversationTurn(
                        tuple(history),
                        message.content,
                        message.answers,
                        self.dataset,
                        self.context_id,
                    )
                )
                if not self.independent_qa:
                    history.append(message)
                pending_user = None
            elif message.role == "system":
                history.append(message)
        return tuple(turns)

def render_conversation(
    tokenizer,
    messages: tuple[ConversationMessage, ...],
    *,
    enable_thinking: bool = False,
) -> str:
    """Render structured messages using the model's native chat template."""
    if not getattr(tokenizer, "apply_chat_template", None):
        raise ValueError(
            "conversation training requires a tokenizer with a chat template"
        )
    payload = [
        {"role": message.role, "content": message.content} for message in messages
    ]
    return tokenizer.apply_chat_template(
        payload,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            yield row

def _messages_from_row(row: dict[str, Any]) -> tuple[ConversationMessage, ...]:
    qa_pairs = row.get("qa_pairs")
    if isinstance(qa_pairs, list):
        messages = []
        for pair in qa_pairs:
            if not isinstance(pair, dict):
                continue
            question = str(pair.get("question", "")).strip()
            answers = pair.get("answers") or ()
            if isinstance(answers, str):
                answers = (answers,)
            answers = tuple(str(value).strip() for value in answers if str(value).strip())
            answer = str(pair.get("answer") or (answers[0] if answers else "")).strip()
            if not question or not answer:
                continue
            question = format_question(question, row.get("dataset", ""), pair.get("metadata") or {})
            messages.extend(
                (
                    ConversationMessage("user", question),
                    ConversationMessage("assistant", answer, answers),
                )
            )
        if messages or isinstance(qa_pairs, list):
            return tuple(messages)
    raw = row.get("conversations")
    if not isinstance(raw, list):
        raise ValueError(
            "conversation data rows must contain a list named 'conversations'"
        )
    messages = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"conversation item {index} must be an object")
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported conversation role: {role!r}")
        if not content:
            continue
        answers = item.get("answers", ())
        if isinstance(answers, str):
            answers = (answers,)
        messages.append(
            ConversationMessage(
                role,
                content,
                tuple(str(value).strip() for value in answers if str(value).strip()),
            )
        )
    # Preserve row-level answer references for a single-turn conversation.
    if len([message for message in messages if message.role == "assistant"]) == 1:
        references = row.get("answers") or ()
        if isinstance(references, str):
            references = (references,)
        references = tuple(str(value).strip() for value in references if str(value).strip())
        for index, message in enumerate(messages):
            if message.role == "assistant":
                messages[index] = ConversationMessage(
                    message.role, message.content, message.answers or references
                )
                break
    return tuple(messages)
def format_question(question: str, dataset: str, metadata: dict) -> str:
    options = metadata.get("options") if dataset == "race" else None
    if not options:
        return question
    choices = "\n".join(str(option).strip() for option in options)
    return f"{question}\nOptions:\n{choices}"
