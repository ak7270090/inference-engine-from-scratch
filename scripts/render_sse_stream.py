"""Render a saved SSE completion stream (``curl -N ... > stream.txt``) as a terminal-style PNG.

    python scripts/render_sse_stream.py stream.txt image.png "$ curl -N localhost:8000/..."

Token text is highlighted so the one-token-per-chunk pattern is easy to see.
"""

import argparse
import html
import json
import re
from pathlib import Path

from render_terminal import build_terminal_html, screenshot_html

SSE_DATA_PREFIX = "data: "
SSE_DONE_MARKER = "[DONE]"
SHORT_ID_LENGTH = 13
KEY_COLOR = "#569cd6"
TOKEN_COLOR = "#ce9178"
FIELD_COLOR = "#c586c0"
DONE_COLOR = "#4ec9b0"


def format_chunk(data: str) -> str:
    """One ``data:`` line, without noisy fields, with the token text highlighted."""
    prefix = f'<span style="color:{KEY_COLOR}">data:</span> '
    if data == SSE_DONE_MARKER:
        return f'{prefix}<span style="color:{DONE_COLOR}">{SSE_DONE_MARKER}</span>'
    chunk = json.loads(data)
    chunk["id"] = chunk["id"][:SHORT_ID_LENGTH] + "…"
    chunk.pop("created", None)
    chunk.pop("object", None)
    escaped = html.escape(json.dumps(chunk, ensure_ascii=False))
    escaped = re.sub(
        r"(&quot;text&quot;: )(&quot;.*?&quot;)",
        rf'\1<span style="color:{TOKEN_COLOR};font-weight:bold">\2</span>',
        escaped,
    )
    escaped = re.sub(
        r"(&quot;(usage|finish_reason)&quot;)",
        rf'<span style="color:{FIELD_COLOR}">\1</span>',
        escaped,
    )
    return prefix + escaped


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an SSE stream as a terminal PNG")
    parser.add_argument("stream_file", type=Path)
    parser.add_argument("image_path", type=Path)
    parser.add_argument("command")
    parser.add_argument("title", nargs="?", default="")
    parsed = parser.parse_args()
    data_lines = [
        line.removeprefix(SSE_DATA_PREFIX)
        for line in parsed.stream_file.read_text(encoding="utf-8").splitlines()
        if line.startswith(SSE_DATA_PREFIX)
    ]
    body_html = "\n".join(format_chunk(data) for data in data_lines)
    screenshot_html(build_terminal_html(body_html, parsed.command, parsed.title), parsed.image_path)


if __name__ == "__main__":
    main()
