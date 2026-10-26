"""Helpers for locating prompt boundaries in rendered chat templates."""
from __future__ import annotations


def _user_prefix_text(text):
    """Return the rendered conversation prefix through its first user header."""
    marker = "<|im_start|>user\n"
    boundary = text.find(marker)
    if boundary < 0:
        return ""
    boundary += len(marker)
    return text[:boundary]

def split_chat_prompt(text):
    """Split memory insertion prefix, complete chat history, and generation header."""
    prefix = _user_prefix_text(text)
    boundary = text.rfind("<|im_end|>")
    if boundary < 0:
        return prefix, text[len(prefix):], ""
    return prefix, text[len(prefix):boundary], text[boundary:]
