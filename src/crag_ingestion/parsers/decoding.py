from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..exceptions import ParseError


@dataclass(slots=True)
class DecodedText:
    text: str
    encoding: str
    warnings: list[str] = field(default_factory=list)


def read_text_safely(path: Path) -> DecodedText:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        return DecodedText(raw.decode("utf-8-sig"), "utf-8-sig")
    if raw.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        return DecodedText(raw.decode("utf-32"), "utf-32")
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return DecodedText(raw.decode("utf-16"), "utf-16")
    try:
        return DecodedText(raw.decode("utf-8"), "utf-8")
    except UnicodeDecodeError:
        pass

    # BOM-less UTF-16 normally has a strong alternating-NUL pattern. Only try
    # it when that signal is present, never merely because byte length is even.
    if raw and raw.count(b"\x00") / len(raw) >= 0.2:
        for encoding in ("utf-16-le", "utf-16-be"):
            try:
                text = raw.decode(encoding)
            except UnicodeDecodeError:
                continue
            if text.count("\x00") / max(1, len(text)) < 0.05:
                return DecodedText(
                    text,
                    encoding,
                    [f"Inferred BOM-less {encoding}; verify the source encoding"],
                )

    # Never guess UTF-16/32 without a BOM: short ANSI documents can otherwise
    # decode into plausible but meaningless characters.
    try:
        from charset_normalizer import from_bytes

        candidate = from_bytes(raw).best()
    except ImportError:
        candidate = None
    if candidate is not None:
        encoding = (candidate.encoding or "").replace("_", "-").casefold()
        if (
            not encoding.startswith(("utf-16", "utf-32"))
            and candidate.chaos < 0.2
            and candidate.coherence >= 0.2
        ):
            text = str(candidate)
            if "\x00" not in text:
                return DecodedText(
                    text,
                    encoding,
                    [f"Decoded non-UTF input as {encoding}; verify the source encoding"],
                )

    for encoding in ("cp1258", "cp1252"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if "\x00" not in text:
            return DecodedText(
                text,
                encoding,
                [f"Decoded legacy input as {encoding}; verify the source encoding"],
            )
    raise ParseError(f"Unable to decode text file safely: {path}")
