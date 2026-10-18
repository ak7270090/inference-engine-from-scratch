"""Render a saved command output as a terminal-style PNG for a README.

    python scripts/render_terminal.py output.txt image.png "$ python -m pytest" "title"

Needs ``playwright`` and Microsoft Edge (preinstalled on Windows).
"""

import argparse
import html
import re
from pathlib import Path

from playwright.sync_api import sync_playwright

TEXT_COLOR = "#d4d4d4"
BACKGROUND_COLOR = "#1e1e1e"
COMMAND_COLOR = "#9cdcfe"
COLOR_RULES = (
    (r"\b(PASSED|passed|All checks passed!|already formatted)", "#4ec9b0"),
    (r"(Required test coverage of \d+% reached[^<]*)", "#4ec9b0"),
    (r"\b(XFAIL|xfailed|XPASS)\b", "#dcdcaa"),
    (r"\b(SKIPPED|skipped|deselected)\b", "#9cdcfe"),
    (r"\b(FAILED|failed|ERROR|error)\b", "#f48771"),
    (r"(TOTAL\s+\d+\s+\d+\s+\d+%)", "#ce9178"),
    (r"(\[\s*\d+%\])", "#808080"),
)
PRE_STYLE = (
    f"margin:0;padding:14px 18px;font:12.5px/1.5 Consolas,monospace;color:{TEXT_COLOR};"
    f"background:{BACKGROUND_COLOR};white-space:pre-wrap;border-radius:0 0 8px 8px"
)
TITLE_STYLE = (
    "background:#2d2d2d;color:#bbb;font:12px Segoe UI,sans-serif;padding:7px 14px;"
    "border-radius:8px 8px 0 0"
)
WINDOW_DOTS = (
    '<span style="color:#ff5f57">&#9679;</span> <span style="color:#febc2e">&#9679;</span> '
    '<span style="color:#28c840">&#9679;</span>'
)


def drop_warnings_block(lines: list[str]) -> list[str]:
    """pytest's warnings summary is noise in a screenshot."""
    kept, is_skipping = [], False
    for line in lines:
        if "warnings summary" in line:
            is_skipping = True
        elif is_skipping and line.startswith("-- Docs"):
            is_skipping = False
        elif not is_skipping:
            kept.append(line)
    return kept


def colorize_line(line: str) -> str:
    escaped = html.escape(line)
    for pattern, color in COLOR_RULES:
        escaped = re.sub(pattern, rf'<span style="color:{color}">\1</span>', escaped)
    if re.match(r"^=+ .* =+$", line):
        escaped = f'<span style="font-weight:bold">{escaped}</span>'
    return escaped


def build_terminal_html(body_html: str, command: str, title: str) -> str:
    title_bar = f'<div style="{TITLE_STYLE}">{WINDOW_DOTS}&nbsp;&nbsp;{html.escape(title)}</div>'
    command_html = f'<span style="color:{COMMAND_COLOR}">{html.escape(command)}</span>'
    return (
        '<html><body style="margin:0;background:#fff">'
        '<div id="terminal" style="display:inline-block;min-width:900px;max-width:1150px">'
        f'{title_bar}<pre style="{PRE_STYLE}">{command_html}\n{body_html}</pre></div></body></html>'
    )


def screenshot_html(page_html: str, image_path: Path) -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge")
        page = browser.new_page(device_scale_factor=2)
        page.set_content(page_html)
        page.locator("#terminal").screenshot(path=str(image_path))
        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render command output as a terminal PNG")
    parser.add_argument("output_file", type=Path)
    parser.add_argument("image_path", type=Path)
    parser.add_argument("command")
    parser.add_argument("title", nargs="?", default="")
    parsed = parser.parse_args()
    text = parsed.output_file.read_text(encoding="utf-8", errors="replace").replace("\r", "")
    lines = drop_warnings_block(text.rstrip("\n").split("\n"))
    body_html = "\n".join(colorize_line(line) for line in lines)
    screenshot_html(build_terminal_html(body_html, parsed.command, parsed.title), parsed.image_path)


if __name__ == "__main__":
    main()
