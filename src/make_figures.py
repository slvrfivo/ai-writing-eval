"""Generate the README figures from the code, config, and experiment log.

- docs/assets/target_loss_weights.svg
    The real assistant target built by ``src/qlora/targets.py``, coloured by the
    token role each character span maps to, with weights from the v1 config.
- docs/assets/prediction_distribution.svg
    Validation ground truth (rounded) and prediction distributions for
    zero-shot, QLoRA v1 and v1b, parsed from ``reports/experiments.md``.
- docs/assets/model_comparison.svg
    Mean RMSE and Spearman per model, parsed from ``reports/experiments.md``.
- docs/assets/dimension_correlation.svg
    Pearson correlation between the three dimensions in train and validation,
    parsed from ``reports/experiments.md``.

Only the standard library is used. ``tests/test_figures.py`` fails when a
committed SVG differs from what this script generates.

Usage (from the repository root):
    python src/make_figures.py          # rewrite all SVGs
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
COMPARISON_SVG = ASSETS / "model_comparison.svg"
CORRELATION_SVG = ASSETS / "dimension_correlation.svg"

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
    "Zero-shot": "#14B8A6",
    "QLoRA v1 (selected)": "#2563EB",
    "QLoRA v1b": "#F59E0B",
}
GT_FILL, GT_STROKE = "#E5E7EB", "#9CA3AF"  # ground-truth bars drawn behind the predictions


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


def markdown_section(markdown: str, heading: str, level: int = 2) -> str:
    """Text of the first section whose heading starts with ``heading``."""
    marker = "#" * level + " "
    sections = re.split(rf"^{marker}", markdown, flags=re.MULTILINE)
    section = next((s for s in sections[1:] if s.startswith(heading)), None)
    if section is None:
        raise ValueError(f"section '{marker}{heading}' not found in {EXPERIMENTS_PATH.name}")
    return section


def parse_ground_truth(markdown: str) -> dict[str, dict[int, int]]:
    """Rounded validation ground-truth counts per dimension from the dataset statistics table."""
    section = markdown_section(markdown, "Validation ground truth after rounding", level=3)
    row = re.compile(r"^\| (Content|Organization|Expression) \|((?: *\d+ *\|){5})", re.MULTILINE)
    rows = row.findall(section)
    if [name for name, _ in rows] != ["Content", "Organization", "Expression"]:
        raise ValueError("expected three ground-truth rows")
    return {
        name.lower(): {score: int(v) for score, v in zip(range(1, 6), cells.strip(" |").split("|"))}
        for name, cells in rows
    }


def render_distribution_figure() -> str:
    markdown = EXPERIMENTS_PATH.read_text(encoding="utf-8")
    series, gt_means = parse_distributions(markdown)
    truth = parse_ground_truth(markdown)
    labels = list(series)
    totals = {label: sum(series[label]["content"].values()) for label in labels}

    width, height = 960, 476
    left, top, panel_w, panel_gap, plot_h = 56, 138, 276, 26, 230
    y_max = 400
    body: list[str] = []
    body.append(svg_text(left - 16, 36, "Score distribution on validation: ground truth vs predictions",
                         size=20, weight="bold"))
    body.append(svg_text(
        left - 16, 60,
        f"Essays per score. Gray: ground truth rounded with ROUND_HALF_UP ({sum(truth['content'].values())} "
        f"essays, count under each score). Zero-shot has {totals['Zero-shot']} valid outputs.",
        size=13, color=MUTED,
    ))

    # legend
    x = left - 16
    body.append(f'<rect x="{fmt(x)}" y="78" width="14" height="14" rx="2" fill="{GT_FILL}" stroke="{GT_STROKE}"/>')
    body.append(svg_text(x + 20, 90, "Ground truth", size=13))
    x += 20 + text_width("Ground truth", 13) + 28
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

        for score in range(1, 6):
            gx = x0 + (score - 1) * group_w
            # ground truth: a wide gray bar behind the prediction bars, count printed under the axis
            gt = truth[dimension][score]
            gh = gt / y_max * plot_h
            if gt:
                body.append(f'<rect x="{fmt(gx + 3)}" y="{fmt(base - gh)}" width="{fmt(group_w - 6)}" '
                            f'height="{fmt(gh)}" fill="{GT_FILL}" stroke="{GT_STROKE}"/>')
            body.append(svg_text(gx + group_w / 2, base + 33, f"GT {gt}", size=10, color=MUTED,
                                 anchor="middle"))
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

    body.append(svg_text(left + (3 * panel_w + 2 * panel_gap) / 2, top + plot_h + 58,
                         "Score", size=13, color=MUTED, anchor="middle"))
    body.append(svg_text(left - 16, height - 14, "Source: reports/experiments.md", size=11, color=MUTED))

    description = "; ".join(
        [f"Ground truth: " + ", ".join(
            f"{d} {dict((s, c) for s, c in truth[d].items() if c)} (mean {gt_means[d]:.2f})" for d in DIMENSIONS)]
        + [f"{label}: " + ", ".join(
            f"{d} {dict((s, c) for s, c in series[label][d].items() if c)}" for d in DIMENSIONS)
           for label in labels]
    )
    return svg_document(width, height, "Score distribution on validation", description, body)


# ------------------------------------------------- model comparison --
def parse_model_metrics(markdown: str) -> list[tuple[str, float, float | None, str]]:
    """[(label, mean RMSE, mean Spearman or None, note)] from the experiment log."""
    # global mean baseline: first results table, Mean RMSE / Mean Spearman columns
    baseline = re.search(r"^\| [\d-]+ \| Global Mean Baseline \|(.*)$", markdown, re.MULTILINE)
    if baseline is None:
        raise ValueError("Global Mean Baseline row not found")
    cells = [c.strip() for c in baseline.group(1).split("|")]
    rmse, spearman = float(cells[4]), cells[8]
    rows = [("Mean baseline", rmse, None if spearman == "undefined" else float(spearman), "validation")]

    mean_row = re.compile(r"^\| Mean \| ([\d.]+) \| ([\d.]+) \|", re.MULTILINE)
    for heading, label in DISTRIBUTION_SECTIONS.items():
        match = mean_row.search(markdown_section(markdown, heading))
        if match is None:
            raise ValueError(f"Mean metric row not found under '## {heading}'")
        note = "validation, 399 valid outputs" if label == "Zero-shot" else "validation"
        rows.append((label, float(match.group(1)), float(match.group(2)), note))

    official = markdown_section(markdown, "Official competition results")
    rmse = re.search(r"Official preliminary RMSE: \*\*([\d.]+)\*\*", official)
    spearman = re.search(r"Official preliminary Spearman: \*\*([\d.]+)\*\*", official)
    if rmse is None or spearman is None:
        raise ValueError("official RMSE/Spearman not found")
    rows.append(("QLoRA v1, official test", float(rmse.group(1)), float(spearman.group(1)), "hidden test"))
    return rows


def render_comparison_figure() -> str:
    rows = parse_model_metrics(EXPERIMENTS_PATH.read_text(encoding="utf-8"))
    colours = {
        "Mean baseline": "#9CA3AF",
        "Zero-shot": SERIES_COLOR["Zero-shot"],
        "QLoRA v1 (selected)": SERIES_COLOR["QLoRA v1 (selected)"],
        "QLoRA v1b": SERIES_COLOR["QLoRA v1b"],
        "QLoRA v1, official test": SERIES_COLOR["QLoRA v1 (selected)"],
    }
    width, label_w, panel_w, gap = 960, 190, 320, 60
    top, row_h, bar_h = 104, 34, 20
    height = top + row_h * len(rows) + 56
    body: list[str] = []
    body.append(svg_text(24, 36, "Model comparison (mean of three dimensions)", size=20, weight="bold"))
    body.append(svg_text(
        24, 60,
        "Validation set except the last row, which is the official hidden test. Zero-shot is scored on "
        "its 399 valid outputs.",
        size=13, color=MUTED,
    ))
    panels = [("Mean RMSE (lower is better)", 1, 1.2, 0.2), ("Mean Spearman (higher is better)", 2, 0.6, 0.1)]
    for p, (title, index, x_max, step) in enumerate(panels):
        x0 = 24 + label_w + p * (panel_w + gap)
        body.append(svg_text(x0, top - 16, title, size=14, weight="bold"))
        base = top + row_h * len(rows)
        tick = 0.0
        while tick <= x_max + 1e-9:
            tx = x0 + tick / x_max * panel_w
            body.append(f'<line x1="{fmt(tx)}" y1="{top - 4}" x2="{fmt(tx)}" y2="{base}" '
                        f'stroke="{GRID if tick else "#9CA3AF"}"/>')
            body.append(svg_text(tx, base + 16, f"{tick:.1f}", size=11, color=MUTED, anchor="middle"))
            tick = round(tick + step, 10)
        for r, row in enumerate(rows):
            label, value, note = row[0], row[index], row[3]
            y = top + r * row_h
            if p == 0:
                body.append(svg_text(24, y + 15, label, size=13, weight="bold" if "selected" in label else "normal"))
                body.append(svg_text(24, y + 29, note, size=10, color=MUTED))
            if value is None:
                body.append(svg_text(x0 + 6, y + 18, "undefined (constant predictions)", size=12, color=MUTED))
                continue
            w = value / x_max * panel_w
            colour = colours[label]
            hatched = label.endswith("official test")
            fill = "url(#hatch)" if hatched else colour
            body.append(f'<rect x="{x0}" y="{fmt(y + 4)}" width="{fmt(w)}" height="{bar_h}" fill="{fill}" '
                        f'stroke="{colour}"/>')
            body.append(svg_text(x0 + w + 6, y + 19, f"{value:.4f}", size=12, color=INK))
    defs = ('<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" '
            'patternTransform="rotate(45)"><rect width="6" height="6" fill="#DBEAFE"/>'
            f'<line x1="0" y1="0" x2="0" y2="6" stroke="{SERIES_COLOR["QLoRA v1 (selected)"]}" stroke-width="3"/>'
            '</pattern></defs>')
    body.insert(0, defs)
    body.append(svg_text(24, height - 14, "Source: reports/experiments.md", size=11, color=MUTED))
    description = "; ".join(
        f"{label}: RMSE {rmse:.4f}, Spearman {'undefined' if sp is None else f'{sp:.4f}'} ({note})"
        for label, rmse, sp, note in rows
    )
    return svg_document(width, height, "Model comparison", description, body)


# ------------------------------------------------ correlation figure --
def parse_correlations(markdown: str) -> dict[str, dict[tuple[str, str], float]]:
    section = markdown_section(markdown, "Pearson correlation between dimensions", level=3)
    row = re.compile(r"^\| (Train|Validation) \(([\d,]+)\) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", re.MULTILINE)
    found = row.findall(section)
    if [split for split, *_ in found] != ["Train", "Validation"]:
        raise ValueError("expected Train and Validation correlation rows")
    result = {}
    for split, n, co, ce, oe in found:
        result[f"{split} ({n})"] = {
            ("content", "organization"): float(co),
            ("content", "expression"): float(ce),
            ("organization", "expression"): float(oe),
        }
    return result


def blend(value: float, lo: float = 0.5, hi: float = 1.0) -> tuple[str, bool]:
    """Colour for a correlation cell on a white-to-blue scale; returns (hex, use white text)."""
    t = min(1.0, max(0.0, (value - lo) / (hi - lo)))
    start, end = (239, 246, 255), (30, 64, 175)
    rgb = tuple(round(a + (b - a) * t) for a, b in zip(start, end))
    return "#%02X%02X%02X" % rgb, t > 0.55


def render_correlation_figure() -> str:
    data = parse_correlations(EXPERIMENTS_PATH.read_text(encoding="utf-8"))
    cell, label_w = 74, 104
    width, top = 2 * (label_w + 3 * cell) + 3 * 40, 92
    height = top + 3 * cell + 52
    body: list[str] = []
    body.append(svg_text(24, 34, "Correlation between scoring dimensions", size=20, weight="bold"))
    body.append(svg_text(24, 56, "Pearson r of the unrounded ground-truth scores.", size=13, color=MUTED))
    names = [d.capitalize() for d in DIMENSIONS]
    for p, (split, values) in enumerate(data.items()):
        x0 = 24 + label_w + p * (label_w + 3 * cell + 40)
        body.append(svg_text(x0 + 1.5 * cell, top - 12, split, size=14, anchor="middle", weight="bold"))
        for i, a in enumerate(DIMENSIONS):
            body.append(svg_text(x0 - 8, top + i * cell + cell / 2 + 5, names[i], size=12, anchor="end"))
            for j, b in enumerate(DIMENSIONS):
                r = 1.0 if a == b else values.get((a, b), values.get((b, a)))
                colour, white = blend(r)
                cx, cy = x0 + j * cell, top + i * cell
                body.append(f'<rect x="{fmt(cx)}" y="{fmt(cy)}" width="{cell - 2}" height="{cell - 2}" '
                            f'fill="{colour}"/>')
                body.append(svg_text(cx + cell / 2 - 1, cy + cell / 2 + 4, "1" if a == b else f"{r:.3f}", size=14,
                                     color="#FFFFFF" if white else INK, anchor="middle"))
        for j in range(3):
            body.append(svg_text(x0 + j * cell + cell / 2, top + 3 * cell + 16, names[j], size=12,
                                 anchor="middle"))
    body.append(svg_text(24, height - 12, "Source: reports/experiments.md", size=11, color=MUTED))
    description = "; ".join(
        f"{split}: " + ", ".join(f"{a}-{b} {r:.3f}" for (a, b), r in values.items())
        for split, values in data.items()
    )
    return svg_document(width, height, "Correlation between scoring dimensions", description, body)


# ---------------------------------------------------------------- main --
FIGURES = {
    TARGET_SVG: render_target_figure,
    DISTRIBUTION_SVG: render_distribution_figure,
    COMPARISON_SVG: render_comparison_figure,
    CORRELATION_SVG: render_correlation_figure,
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
