"""Shell rendering helpers for generated command plans."""

from __future__ import annotations


def shell_join(parts: list[str]) -> str:
    def quote(part: str) -> str:
        if not part:
            return "''"
        safe = set(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789_+-=.,:/@%$"
        )
        if all(char in safe for char in part):
            return part
        return "'" + part.replace("'", "'\"'\"'") + "'"

    return " ".join(quote(str(part)) for part in parts)
