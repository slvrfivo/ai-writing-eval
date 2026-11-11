"""Generate the README figures from the code, config, and experiment log.

- docs/assets/target_loss_weights.svg
    The real assistant target built by ``src/qlora/targets.py``, coloured by the
    token role each character span maps to, with weights from the v1 config.
- docs/assets/prediction_distribution.svg
    Validation prediction distributions for zero-shot, QLoRA v1 and v1b, parsed
    from ``reports/experiments.md``.

Only the standard library is used. ``tests/test_figures.py`` fails when a
committed SVG differs from what this script generates.

Usage (from the repository root):
    python src/make_figures.py          # rewrite both SVGs
    python src/make_figures.py --check  # exit 1 if a committed SVG is stale
"""

from __future__ import annotations

import argparse
import ast
import math
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

try:
    from .evaluate import DIMENSIONS
    from .qlora.config import QLoRAConfig
    from .qlora.targets import build_assistant_target
except ImportError:  # python src/make_figures.py
    from evaluate import DIMENSIONS
    from qlora.config import QLoRAConfig
    from qlora.targets import build_assistant_target


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
CONFIG_PATH = ROOT / "configs" / "qwen3_4b_qlora_v1.json"
EXPERIMENTS_PATH = ROOT / "reports" / "experiments.md"

TARGET_SVG = ASSETS / "target_loss_weights.svg"
DISTRIBUTION_SVG = ASSETS / "prediction_distribution.svg"

FONT = "Arial, 'Noto Sans KR', 'Malgun Gothic', sans-serif"
MONO = "Consolas, 'D2Coding', 'Noto Sans Mono CJK KR', monospace"
INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"

# Example scores shown in the target figure (same as the README's output example).
EXAMPLE_SCORES = {"content": 4, "organization": 3, "expression": 4}

ROLE_STYLE = {
    # role: (background, text colour, bold)
    "score": ("#C2410C", "#FFFFFF", True),
    "structure": ("#E5E7EB", "#374151", False),
    "rationale": ("#EFF6FF", "#1E40AF", False),
}
ROLE_LABEL = {
    "score": "Score",
    "structure": "JSON structure",
    "rationale": "Rationale",
    "prompt": "Prompt",
}

# section heading in reports/experiments.md -> series label
DISTRIBUTION_SECTIONS = {
    "Qwen3-4B zero-shot baseline": "Zero-shot",
    "Qwen3-4B weighted QLoRA v1": "QLoRA v1 (selected)",
    "Qwen3-4B class-balanced QLoRA v1b": "QLoRA v1b",
}
SERIES_COLOR = {
    "Zero-shot": "#9CA3AF",
    "QLoRA v1 (selected)": "#2563EB",
    "QLoRA v1b": "#F59E0B",
}


# ---------------------------------------------------------------- helpers --
def char_width(ch: str, size: float) -> float:
    """Approximate glyph width. textLength pins the rendered width to this value."""
    if ord(ch) >= 0x1100:  # Hangul and other wide glyphs
        return size * 1.0
    return size * 0.6


def text_width(text: str, size: float) -> float:
    # math.fsum is exact, so the SVG is identical on every Python version
    # (built-in sum() switched to compensated float summation in 3.12).
    return math.fsum(char_width(ch, size) for ch in text)


def fmt(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def svg_text(x, y, content, *, size=13, color=INK, anchor="start", weight="normal", family=FONT,
             halo=False):
    """halo=True draws a white outline behind the glyphs so lines underneath stay readable."""
    outline = ' stroke="#FFFFFF" stroke-width="3" paint-order="stroke"' if halo else ""
    return (
        f'<text x="{fmt(x)}" y="{fmt(y)}" font-family="{family}" font-size="{size}" '
        f'fill="{color}" text-anchor="{anchor}" font-weight="{weight}"{outline}>{escape(str(content))}</text>'
    )


def svg_document(width: int, height: int, title: str, description: str, body: list[str]) -> str:
    head = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">{escape(description)}</desc>',
        f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
    ]
    return "\n".join(head + body + ["</svg>"]) + "\n"


# ------------------------------------------------------ target figure --
def target_lines(target):
    """Split the compact target JSON into display lines of (text, role) chunks.

    A new line starts before every dimension key after the first and before
    every rationale, so each dimension reads as "key + score" then "rationale".
    """
    lines: list[list[tuple[str, str]]] = [[]]
    for span in target.spans:
        text = target.text[span.start:span.end]
        starts_key = span.role == "structure" and any(
            text.startswith(f'"{dimension}"') for dimension in DIMENSIONS[1:]
        )
        if (starts_key or span.role == "rationale") and lines[-1]:
            lines.append([])
        lines[-1].append((text, span.role))
    return lines


def render_target_figure() -> str:
    config = QLoRAConfig.from_json(CONFIG_PATH)
    weights = config.loss_weights
    target = build_assistant_target(EXAMPLE_SCORES)
    lines = target_lines(target)

    width, size, line_h, left, indent = 960, 14, 34, 40, 28
    body: list[str] = []
    body.append(svg_text(left, 36, "Loss weight by token role", size=20, weight="bold"))
    body.append(svg_text(
        left, 60,
        "Training target for scores 4 / 3 / 4, built by src/qlora/targets.py. "
        "Weights from configs/qwen3_4b_qlora_v1.json.",
        size=13, color=MUTED,
    ))

    # Prompt block (not trained)
    y = 84
    body.append(f'<rect x="{left}" y="{y}" width="{width - 2 * left}" height="40" rx="6" '
                f'fill="#FFFFFF" stroke="#D1D5DB" stroke-dasharray="4 3"/>')
    body.append(svg_text(left + 14, y + 25, "System prompt + writing prompt + essay (chat template)",
                         size=13, color=MUTED))
    body.append(svg_text(width - left - 14, y + 25, f"Prompt  weight {weights['prompt']:.2f}",
                         size=13, color=MUTED, anchor="end", weight="bold"))

    body.append(svg_text(left, y + 70, "Assistant target (what the model learns to generate)",
                         size=13, color=INK, weight="bold"))

    y = y + 92
    for line in lines:
        x = left + (indent if line[0][1] == "rationale" else 0)
        for text, role in line:
            background, color, bold = ROLE_STYLE[role]
            w = text_width(text, size)
            pad = 3 if role != "score" else 6
            body.append(f'<rect x="{fmt(x)}" y="{y - 18}" width="{fmt(w + 2 * pad)}" height="25" '
                        f'rx="4" fill="{background}"/>')
            body.append(
                f'<text x="{fmt(x + pad)}" y="{y}" font-family="{MONO}" font-size="{size}" fill="{color}" '
                f'font-weight="{"bold" if bold else "normal"}" textLength="{fmt(w)}" '
                f'lengthAdjust="spacingAndGlyphs" xml:space="preserve">{escape(text)}</text>'
            )
            x += w + 2 * pad + 2
        y += line_h

    # Legend with the configured weights
    y += 14
    x = left
    for role in ("score", "structure", "rationale"):
        background, color, _ = ROLE_STYLE[role]
        label = f"{ROLE_LABEL[role]}  ×{weights[role]:.2f}"
        outline = ' stroke="#9CA3AF"' if role != "score" else ""
        body.append(f'<rect x="{fmt(x)}" y="{y - 13}" width="16" height="16" rx="3" fill="{background}"{outline}/>')
        body.append(svg_text(x + 24, y, label, size=14, weight="bold" if role == "score" else "normal"))
        x += 24 + text_width(label, 14) + 36
    y += 30
    body.append(svg_text(
        left, y,
        "Tokens take the role of the span they overlap (character offsets). Rationales are fixed "
        "rubric templates, one per dimension and score.",
        size=12, color=MUTED,
    ))

    height = y + 24
    description = (
        "The compact JSON training target for scores 4, 3 and 4 with each span coloured by role: "
        f"score tokens weight {weights['score']:.2f}, JSON structure {weights['structure']:.2f}, "
        f"rationale {weights['rationale']:.2f}, prompt {weights['prompt']:.2f}."
    )
    return svg_document(width, height, "Loss weight by token role", description, body)


# ------------------------------------------------ distribution figure --
def parse_distributions(markdown: str):
    """Return ({series: {dimension: {score: count}}}, {dimension: gt_mean}) from the experiment log."""
    sections = re.split(r"^## ", markdown, flags=re.MULTILINE)
    row = re.compile(r"^\| (Content|Organization|Expression) \|(.*)`(\{[^`]*\})` \|\s*$", re.MULTILINE)
    series: dict[str, dict[str, dict[int, int]]] = {}
    gt_means: dict[str, float] = {}
    for heading, label in DISTRIBUTION_SECTIONS.items():
        section = next((s for s in sections if s.startswith(heading)), None)
        if section is None:
            raise ValueError(f"section '## {heading}' not found in {EXPERIMENTS_PATH.name}")
        rows = row.findall(section)
        if [name for name, _, _ in rows] != ["Content", "Organization", "Expression"]:
            raise ValueError(f"expected three distribution rows under '## {heading}'")
        series[label] = {}
        for name, middle, literal in rows:
            dimension = name.lower()
            counts = ast.literal_eval(literal)
            series[label][dimension] = {score: int(counts.get(score, 0)) for score in range(1, 6)}
            cells = [c.strip() for c in middle.split("|") if c.strip()]
            if label == "QLoRA v1 (selected)" and cells:
                gt_means[dimension] = float(cells[0])  # GT mean over all 400 validation essays
    return series, gt_means


def render_distribution_figure() -> str:
    series, gt_means = parse_distributions(EXPERIMENTS_PATH.read_text(encoding="utf-8"))
    labels = list(series)
    totals = {label: sum(series[label]["content"].values()) for label in labels}

    width, height = 960, 456
    left, top, panel_w, panel_gap, plot_h = 56, 138, 276, 26, 230
    y_max = 400
    body: list[str] = []
    body.append(svg_text(left - 16, 36, "Predicted score distribution on validation", size=20, weight="bold"))
    body.append(svg_text(
        left - 16, 60,
        f"Counts of essays per predicted score. Zero-shot has {totals['Zero-shot']} valid outputs, "
        f"QLoRA runs {totals['QLoRA v1 (selected)']}. Dashed line: mean ground-truth score.",
        size=13, color=MUTED,
    ))

    # legend
    x = left - 16
    for label in labels:
        body.append(f'<rect x="{fmt(x)}" y="78" width="14" height="14" rx="2" fill="{SERIES_COLOR[label]}"/>')
        body.append(svg_text(x + 20, 90, label, size=13))
        x += 20 + text_width(label, 13) + 28

    group_w = panel_w / 5
    bar_w, bar_gap = 15, 3
    for p, dimension in enumerate(DIMENSIONS):
        x0 = left + p * (panel_w + panel_gap)
        base = top + plot_h
        # grid and y labels
        for tick in range(0, y_max + 1, 100):
            gy = base - tick / y_max * plot_h
            body.append(f'<line x1="{x0}" y1="{fmt(gy)}" x2="{x0 + panel_w}" y2="{fmt(gy)}" '
                        f'stroke="{GRID if tick else "#9CA3AF"}"/>')
            if p == 0:
                body.append(svg_text(x0 - 8, gy + 4, tick, size=11, color=MUTED, anchor="end"))
        body.append(svg_text(x0 + panel_w / 2, top - 12, dimension.capitalize(), size=14,
                             anchor="middle", weight="bold"))

        # ground-truth mean marker on the 1-5 score axis (drawn behind the bars)
        mean = gt_means[dimension]
        mx = x0 + (mean - 0.5) * group_w
        body.append(f'<line x1="{fmt(mx)}" y1="{top}" x2="{fmt(mx)}" y2="{base}" stroke="{MUTED}" '
                    f'stroke-width="1.5" stroke-dasharray="5 4"/>')
        body.append(svg_text(mx + 5, top + 12, f"GT mean {mean:.2f}", size=11, color=INK, halo=True))

        for score in range(1, 6):
            gx = x0 + (score - 1) * group_w
            start = gx + (group_w - bar_w * len(labels) - bar_gap * (len(labels) - 1)) / 2
            for i, label in enumerate(labels):
                count = series[label][dimension][score]
                bx = start + i * (bar_w + bar_gap)
                bh = count / y_max * plot_h
                if count:
                    body.append(f'<rect x="{fmt(bx)}" y="{fmt(base - bh)}" width="{bar_w}" '
                                f'height="{fmt(bh)}" fill="{SERIES_COLOR[label]}"/>')
                    body.append(svg_text(bx + bar_w / 2, base - bh - 4, count, size=9,
                                         color=MUTED, anchor="middle", halo=True))
            body.append(svg_text(gx + group_w / 2, base + 18, score, size=12, color=INK, anchor="middle"))

    body.append(svg_text(left + (3 * panel_w + 2 * panel_gap) / 2, top + plot_h + 42,
                         "Predicted score", size=13, color=MUTED, anchor="middle"))
    body.append(svg_text(left - 16, height - 14, "Source: reports/experiments.md", size=11, color=MUTED))

    description = "; ".join(
        f"{label}: " + ", ".join(
            f"{d} {dict((s, c) for s, c in series[label][d].items() if c)}" for d in DIMENSIONS
        )
        for label in labels
    )
    return svg_document(width, height, "Predicted score distribution on validation", description, body)


# ---------------------------------------------------------------- main --
FIGURES = {
    TARGET_SVG: render_target_figure,
    DISTRIBUTION_SVG: render_distribution_figure,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="only check that committed SVGs are current")
    args = parser.parse_args(argv)

    stale = []
    for path, render in FIGURES.items():
        svg = render()
        if args.check:
            current = path.read_text(encoding="utf-8").replace("\r\n", "\n") if path.exists() else None
            if current != svg:
                stale.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(svg.encode("utf-8"))
        print(f"wrote {path.relative_to(ROOT).as_posix()}")
    if stale:
        names = ", ".join(p.relative_to(ROOT).as_posix() for p in stale)
        sys.exit(f"stale figures: {names}. Run: python src/make_figures.py")


if __name__ == "__main__":
    main()
