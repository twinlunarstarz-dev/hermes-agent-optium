"""Deterministic cleanup turning assistant Markdown into a spoken script.

Shared by explicit TTS calls, gateway auto-TTS, voice-mode streaming and the web
dashboard. Non-ASCII characters are written as escapes on purpose so the file
stays free of invisible/look-alike glyphs.
"""

from __future__ import annotations

import html
import re

# Sentinel appended to former heading lines so smooth_whitespace_for_tts folds the
# heading into the sentence after it ("Weather, it will be sunny") instead of a bare
# "Weather." label.
_HEAD = "\x00"

_MD_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:[^()]|\([^)]*\))*\)")
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\((?:[^()]|\([^)]*\))*\)")
_MD_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", flags=re.DOTALL)
_MD_UNDERSCORE_BOLD_RE = re.compile(r"__(.+?)__", flags=re.DOTALL)
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", flags=re.DOTALL)
_MD_UNDERSCORE_ITALIC_RE = re.compile(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", flags=re.DOTALL)
_MD_STRIKE_RE = re.compile(r"~~(.+?)~~", flags=re.DOTALL)
_MD_HEADING_LINE_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", flags=re.MULTILINE)
_MD_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?", flags=re.MULTILINE)
_MD_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", flags=re.MULTILINE)
_MD_HR_RE = re.compile(r"^\s*[-*_]{3,}\s*$", flags=re.MULTILINE)
_MD_TABLE_PIPE_RE = re.compile(r"\s*\|\s*")
# URLs are READ aloud ("github dot com slash user slash repo"), not deleted —
# trailing sentence punctuation is stripped in ``_url_to_spoken``, not by the regex.
_URL_RE = re.compile(r"https?://\S+")
# Local file links ("MEDIA:/Users/me/file.xlsx") are click targets on screen, not
# speech: voices loop on the hyphenated slug ("eeeeee"). The token is silence; the
# assistant's prose already says "the files are below". Trailing sentence
# punctuation is left in place so "see MEDIA:/x.py. Then" keeps its full stop.
_MEDIA_PATH_RE = re.compile(r"MEDIA:\S+?(?=[.,;:!?)\]]*(?:\s|$))")

_DEGREE_UNITS = (("C", "Celsius"), ("F", "Fahrenheit"))
# Unit suffix (regex, after a digit) -> spoken word; km/h variants before the bare "m".
_UNIT_WORDS = (
    (r"km\s*/\s*h", "kilometres per hour"), (r"km/h", "kilometres per hour"),
    (r"mm", "millimetres"), (r"cm", "centimetres"), (r"m", "metres"))
# Capital magnitude suffixes after a number: "$1.5M" -> "1.5 million", "5K" -> "5 thousand".
_MAGNITUDE_SUFFIXES = ((r"M", "million"), (r"K", "thousand"), (r"B", "billion"))
# Currency prefix (regex) -> spoken word; order matters (NZ$/A$/US$ before bare $).
_CURRENCY_WORDS = (
    (r"NZ\$", "New Zealand dollars", re.IGNORECASE), (r"A\$", "Australian dollars", re.IGNORECASE),
    (r"US\$", "US dollars", re.IGNORECASE), ("€", "euros", 0), ("£", "pounds", 0), (r"\$", "dollars", 0),
)

# Common technical abbreviations need explicit letter names for small TTS models.
_SPOKEN_ABBREVIATIONS = (
    (r"\bHTTPS\b", "H T T P S"), (r"\bHTTP\b", "H T T P"),
    (r"\bAPIs\b", "A P I's"), (r"\bAPI\b", "A P I"),
    (r"\bURLs\b", "U R Ls"), (r"\bURL\b", "U R L"),
    (r"\bTTS\b", "T T S"), (r"\bSTT\b", "S T T"),
    (r"\bLLM\b", "L L M"), (r"\bGPU\b", "G P U"), (r"\bCPU\b", "C P U"),
    (r"\bCLI\b", "C L I"), (r"\bMCP\b", "M C P"), (r"\bJSON\b", "J S O N"),
    (r"\bYAML\b", "Y A M L"), (r"\bSSH\b", "S S H"),
    (r"\bFYI\b", "F Y I"), (r"\bETA\b", "E T A"),
    (r"\bASAP\b", "as soon as possible"), (r"\bPR\b", "pull request"),
)


# Broad emoji / pictograph cleanup: most voice providers read emojis as awkward labels.
_EMOJI_RE = re.compile(
    "[\U0001F1E6-\U0001F1FF\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF☀-➿]+",
    flags=re.UNICODE)
_VARIATION_SELECTOR_RE = re.compile("[︎️]")


def _url_to_spoken(url: str) -> str:
    """Turn a URL into words a TTS engine reads naturally: ``https://github.com/u/r.``
    -> ``github dot com slash u slash r`` (trailing sentence punctuation dropped)."""
    body = re.sub(r"^https?://", "", url, flags=re.IGNORECASE).rstrip(".,;:!?)>\"'”’]")
    parts = []
    for i, chunk in enumerate(body.split("/")):
        if i > 0:
            parts.append("slash")
        if i == 0:
            # Dots in the host are spoken: "github.com" -> "github dot com".
            parts.append(re.sub(r"\.", " dot ", chunk.strip(".")))
        elif chunk:
            parts.append(chunk)
    return " ".join(p for p in parts if p).strip()


def strip_markdown_for_tts(text: str) -> str:
    """Strip Markdown/Telegram formatting while preserving readable words."""
    if not text:
        return ""
    text = html.unescape(str(text))
    text = _MD_CODE_BLOCK_RE.sub(" ", text)
    text = _MD_IMAGE_RE.sub(lambda m: f" {m.group(1)} " if m.group(1) else " ", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub(lambda m: " " + _url_to_spoken(m.group(0)) + " ", text)
    text = _MD_INLINE_CODE_RE.sub(r"\1", text)
    text = _MD_BOLD_RE.sub(r"\1", text)
    text = _MD_UNDERSCORE_BOLD_RE.sub(r"\1", text)
    text = _MD_ITALIC_RE.sub(r"\1", text)
    text = _MD_UNDERSCORE_ITALIC_RE.sub(r"\1", text)
    text = _MD_STRIKE_RE.sub(r"\1", text)
    # Speak file paths ("slash home slash user slash doc dot pdf") instead of deleting
    # them; bare filenames with a known extension are also spoken as themselves.
    text = _PATH_TOKEN_RE.sub(lambda m: " " + _path_to_spoken(m.group(0)) + " ", text)
    text = _PATH_INTRO_RE.sub(" ", text)
    # Mark headings (do not just delete the marker): see _HEAD.
    text = _MD_HEADING_LINE_RE.sub(lambda m: m.group(1).rstrip() + _HEAD, text)
    text = _MD_BLOCKQUOTE_RE.sub("", text)
    text = _MD_LIST_ITEM_RE.sub("", text)
    text = _MD_HR_RE.sub("", text)
    # Leftover table pipes become pauses instead of a spoken "vertical bar".
    return _MD_TABLE_PIPE_RE.sub("; ", text)


def _normalize_temperature_ranges(text: str) -> str:
    """``11-17°C`` -> ``11 to 17 degrees Celsius`` (en/em dash or hyphen; unicode minus normalized)."""
    number = r"([-+\u2212]?\d+(?:\.\d+)?)"
    for unit, word in _DEGREE_UNITS:
        text = re.sub(
            r"(?<!\w)" + number + r"\s*[\u2013\u2014-]\s*" + number + r"\s*°\s*" + unit + r"\b",
            lambda m, w=word: (
                f"{m.group(1).replace(chr(0x2212), '-')} to {m.group(2).replace(chr(0x2212), '-')} degrees {w}"
            ),
            text, flags=re.IGNORECASE)
    return text


def normalize_symbols_for_tts(text: str) -> str:
    """Expand common symbols/shorthand into words a TTS engine reads well."""
    if not text:
        return ""
    text = re.sub("[   ]", " ", str(text))  # non-breaking / thin spaces
    text = text.replace("\u2212", "-").replace("…", "...")  # minus sign, ellipsis
    text = _normalize_temperature_ranges(text)
    # Temperatures with a number first, then bare units ("measured in degrees C"),
    # then any remaining degree symbol (angles, stray cases).
    for unit, word in _DEGREE_UNITS:
        text = re.sub(
            r"(?<!\w)([-+]?\d+(?:\.\d+)?)\s*°\s*" + unit + r"\b", r"\1 degrees " + word, text, flags=re.IGNORECASE,
        )
    for unit, word in _DEGREE_UNITS:
        text = re.sub(r"°\s*" + unit + r"\b", "degrees " + word, text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\w)([-+]?\d+(?:\.\d+)?)\s*°", r"\1 degrees", text).replace("°", " degrees")
    # Common weather/travel units.  Case-sensitive: a capital M/K after a number is a
    # magnitude suffix ("$1.5M", "5K rows"), not a metric unit — handled next.
    for pattern, word in _UNIT_WORDS:
        text = re.sub(r"(?<=\d)\s*" + pattern + r"\b", " " + word, text)
    for pattern, word in _MAGNITUDE_SUFFIXES:
        text = re.sub(r"(?<=\d)\s*" + pattern + r"\b", " " + word, text)
    # Numeric rates only ("5/month" -> "5 per month").  Requiring digit-then-letter
    # keeps "and/or", "N/A", "TCP/IP" and dates like "2026/06" intact.
    text = re.sub(r"(?<=\d)\s*/\s*(?=[A-Za-z])", " per ", text)
    # Money and percentages. The integer part must END in a digit so a trailing
    # comma ("A$50, ...") is not swallowed into the spoken amount. Prefixed
    # currencies run first so "$" doesn't eat "NZ$".
    for symbol, word, flags in _CURRENCY_WORDS:
        text = re.sub(symbol + r"\s*([\d,]*\d(?:\.\d+)?)", r"\1 " + word, text, flags=flags)
    text = re.sub(r"(?<=\d)\s*%", " percent", text)
    # Operators and separators that commonly leak from formatted answers.
    text = re.sub("[•◦▪▫]", " ", text.replace("&", " and "))  # bullet glyphs
    for symbol, word in (("→", " to "), ("⇒", " to "), ("≈", " about "), ("~", " about ")):
        text = text.replace(symbol, word)
    for pattern, word in _SPOKEN_ABBREVIATIONS:
        text = re.sub(pattern, word, text)
    return _EMOJI_RE.sub("", _VARIATION_SELECTOR_RE.sub("", text))


def smooth_whitespace_for_tts(text: str) -> str:
    """Collapse visual formatting into calm spoken paragraphs. A _HEAD-marked heading folds into
    the next content line as a lead-in ("Weather, It will be sunny."); a heading with no content
    after it becomes its own sentence."""
    if not text:
        return ""
    raw_lines = text.splitlines()
    add_sentence_pauses = sum(1 for raw_line in raw_lines if raw_line.replace(_HEAD, "").strip()) > 1
    lines: list[str] = []
    pending_heading: str | None = None

    def flush_pending() -> None:
        nonlocal pending_heading
        if pending_heading is not None:
            lines.append(pending_heading.rstrip(".:;,") + ".")
            pending_heading = None
    for raw_line in raw_lines:
        is_heading = raw_line.rstrip().endswith(_HEAD)
        line = raw_line.replace(_HEAD, "").strip()
        if not line:
            # Hold a pending heading across blank lines so it still folds into the next content line.
            if pending_heading is None and lines and lines[-1] != "":
                lines.append("")
            continue
        if is_heading:
            flush_pending()
            pending_heading = line.rstrip(".:;,")
            continue
        if pending_heading is not None:
            line = f"{pending_heading.rstrip('.:;,')}, {line}"
            pending_heading = None
        if add_sentence_pauses:
            if line[-1] in ":;":
                # A trailing colon promises a continuation ("the list:"); the raw
                # token after it (file link, code fence) is already gone from
                # speech, so the voice would hang on the open pause. Close it.
                line = line.rstrip(":;").rstrip() + "."
            elif line[-1] not in ".!?":
                line += "."
        lines.append(line)
    flush_pending()
    text = "\n".join(lines)
    for pattern, repl in ((r"\n{3,}", "\n\n"), (r"[ \t]{2,}", " "), (r"\s+([,.;:!?])", r"\1"),
                          (r"([,.;:!?])([A-Za-z])", r"\1 \2"), (r"\.{4,}", "...")):
        text = re.sub(pattern, repl, text)
    # Single-line text ("Here is the list:") skips the per-line pause pass, so close a
    # final colon here too -- never a digit-preceded one ("final score 3:2" is a ratio).
    text = re.sub(r"([^0-9])(?:[:;]\s*)+$", r"\1.", text)
    return text.strip()


# ``/reasoning show`` emits ``<think>...</think>`` in the final message: users want to
# SEE reasoning, not hear it. An unterminated block (streaming cut-off) is also silenced.
# Reasoning blocks: models with ``/reasoning show`` enabled emit ``<think>...</think>`` blocks in the final
# assistant message. See #34213.
_THINK_BLOCK_RE = re.compile(r"<think[\s>].*?</think>", flags=re.DOTALL | re.IGNORECASE)
_THINK_BLOCK_OPEN_RE = re.compile(r"<think[\s>].*\Z", flags=re.DOTALL | re.IGNORECASE)

# run_agent.py's turn-end file-mutation verifier footer (a ``⚠️ File-mutation verifier:``
# header line plus indented ``•`` bullets) is a UI affordance, not speech.
_VERIFIER_FOOTER_RE = re.compile(r"^\s*⚠️?\s*File-mutation verifier:.*(?:\n[ \t]+•.*)*", flags=re.MULTILINE)

# ``run_turn.py`` may prepend reasoning for visual Discord display.  It is not part of the
# answer and must not leak into voice playback, regardless of the configured display style.
_DISPLAY_REASONING_FENCED_RE = re.compile(
    r"\A[ \t]*(?:[-#>]+[ \t]*)?(?:💭[ \t]*)?\*{0,2}Reasoning:?\*{0,2}[ \t]*\n"
    r"[ \t]*```[\s\S]*?```[ \t]*(?:\n|$)", flags=re.IGNORECASE)
_DISPLAY_REASONING_SUBTEXT_RE = re.compile(
    r"\A[ \t]*-#[ \t]*(?:💭[ \t]*)?\*{0,2}Reasoning:?\*{0,2}[ \t]*\n"
    r"(?:^[ \t]*-#[ \t]?.*(?:\n|$))*", flags=re.MULTILINE | re.IGNORECASE)
_DISPLAY_REASONING_BLOCKQUOTE_RE = re.compile(
    r"\A[ \t]*>[ \t]*(?:💭[ \t]*)?\*{0,2}Reasoning:?\*{0,2}[ \t]*\n"
    r"(?:^[ \t]*>.*(?:\n|$))*", flags=re.MULTILINE | re.IGNORECASE)
_DISPLAY_REASONING_LABEL_RE = re.compile(
    r"\A[ \t]*(?:[-#>]+[ \t]*)?(?:💭[ \t]*)?\*{0,2}Reasoning:?\*{0,2}[ \t]*",
    flags=re.IGNORECASE)


def strip_nonspoken_blocks(text: str) -> str:
    """Remove hidden/displayed reasoning and delivery-verifier metadata from speech."""
    if not text:
        return ""
    for pattern in (
        _THINK_BLOCK_RE, _THINK_BLOCK_OPEN_RE, _VERIFIER_FOOTER_RE,
        _DISPLAY_REASONING_FENCED_RE, _DISPLAY_REASONING_SUBTEXT_RE,
        _DISPLAY_REASONING_BLOCKQUOTE_RE, _DISPLAY_REASONING_LABEL_RE,
    ):
        text = pattern.sub(" ", text)
    return text


def _path_to_spoken(path: str) -> str:
    """Turn a file path into words a TTS engine reads naturally:
    ``/home/user/doc.pdf`` -> ``slash home slash user slash doc dot pdf``."""
    spoken = re.sub(r"\.(?=[A-Za-z0-9])", " dot ", path)  # extensions & dots in names
    spoken = spoken.replace("/", " slash ").replace("\\", " slash ")
    spoken = spoken.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", spoken).strip()


# Paths are READ aloud ("slash home slash user slash doc dot pdf") when presenting a
# file name or location, not deleted.  Keep ordinary rates (``5/month``) and prose
# slashes intact; target absolute/relative paths and repo paths with a file extension.
_PATH_TOKEN_RE = re.compile(
    r"(?<!\w)(?:"
    r"/(?:[A-Za-z0-9._~+@-]+/)+[A-Za-z0-9._~+@-]+"
    r"|~/(?:[A-Za-z0-9._~+@-]+/)+[A-Za-z0-9._~+@-]+"
    r"|(?:\.\.?/)(?:[A-Za-z0-9._~+@-]+/)*[A-Za-z0-9._~+@-]+"
    r"|[A-Za-z]:[\\/](?:[^\\/\s,;:!?)]*[\\/])*[^\\/\s,;:!?)]*"
    r"|(?:[A-Za-z0-9._-]+/)+[A-Za-z0-9._-]+\.(?:py|md|yaml|yml|json|toml|txt|wav|mp3|ogg|sh|js|ts|tsx|jsx)"
    r")(?=$|[\s,;:!?])"
)
_PATH_INTRO_RE = re.compile(
    r"\b(?:the\s+)?(?:file|response|output|change|code|script|path)\s+"
    r"(?:is|was|lives|can be found)\s+(?:at|in)\s*(?=$|[,.;:!?])",
    flags=re.IGNORECASE,
)

def flatten_newlines_for_payload(text: str) -> str:
    """Collapse newlines into sentence breaks for single-line TTS payloads: some OpenAI-compatible
    backends (e.g. Kokoro) truncate at the first newline; smoothing already ends each line with
    punctuation, so this is safe.

    See #9004.
    """
    if not text:
        return ""
    for pattern, repl in ((r"\n{2,}", ". "), (r"(?<=[.!?;:,])\n", " "), (r"\n", ". "), (r"\.\s*\.", "."),
                          (r"[ \t]{2,}", " ")):
        text = re.sub(pattern, repl, text)
    # Single-line text ("Here is the list:") skips the per-line pause pass, so close a
    # final colon here too -- never a digit-preceded one ("final score 3:2" is a ratio).
    text = re.sub(r"([^0-9])(?:[:;]\s*)+$", r"\1.", text)
    return text.strip()


def prepare_spoken_text(text: str, max_chars: int | None = 4000) -> str:
    """Return a TTS-friendly script from assistant text (deterministic cleanup, not a rewrite).
    Pipeline: non-spoken blocks > Markdown > symbols/units > line formatting into sentence
    pauses > single line (for newline-sensitive providers), then ``max_chars``."""
    spoken = text
    for step in (strip_nonspoken_blocks, strip_markdown_for_tts, normalize_symbols_for_tts,
                 smooth_whitespace_for_tts, flatten_newlines_for_payload):
        spoken = step(spoken)
    if max_chars is not None and max_chars > 0 and len(spoken) > max_chars:
        spoken = spoken[:max_chars].rstrip()
    return spoken


def _strip_markdown_for_tts(text: str) -> str:
    """``prepare_spoken_text`` without a length cap (``tts_tool`` compatibility name)."""
    return prepare_spoken_text(text, max_chars=None)
