from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "resources" / "design_db.v1.json"


def load_db() -> dict[str, Any]:
    return json.loads(DB_PATH.read_text(encoding="utf-8"))


def slug_score(item: dict[str, Any], product: str, theme: str, style: str | None = None) -> float:
    score = 0.0
    tags = set(item.get("tags", []))
    products = set(item.get("products", []))
    themes = set(item.get("themes", []))
    item_id = item.get("id", "")

    if product in products:
        score += 0.45
    if product in tags:
        score += 0.18
    if theme in themes:
        score += 0.30
    if theme in tags:
        score += 0.12
    if style and (style == item_id or style in tags):
        score += 0.28
    if "premium" in tags:
        score += 0.03
    return min(score, 0.98)


def select_first(items: list[dict[str, Any]], ids: list[str], product: str, theme: str, style: str | None = None) -> dict[str, Any]:
    lookup = {item["id"]: item for item in items}
    ranked = []
    for item_id in ids:
        item = lookup.get(item_id)
        if item:
            ranked.append((slug_score(item, product, theme, style), item))
    ranked.sort(key=lambda row: row[0], reverse=True)
    if ranked:
        return ranked[0][1]
    return items[0]


def select_many(items: list[dict[str, Any]], ids: list[str]) -> list[str]:
    lookup = {item["id"]: item for item in items}
    return [lookup[item_id]["name"] for item_id in ids if item_id in lookup]


def normalize_theme(theme: str) -> str:
    theme = theme.lower().strip()
    return "dark" if theme not in {"light", "dark"} else theme


def get_product(db: dict[str, Any], product: str) -> dict[str, Any]:
    for item in db["products"]:
        if item["id"] == product:
            return item
    valid = ", ".join(sorted(p["id"] for p in db["products"]))
    raise SystemExit(f"Unknown product '{product}'. Valid products: {valid}")


def build_recommendation(db: dict[str, Any], product_id: str, theme: str, style_override: str | None = None) -> dict[str, Any]:
    product = get_product(db, product_id)
    theme = normalize_theme(theme)

    style = select_first(db["styles"], product["recommended_styles"], product_id, theme, style_override)
    palette = select_first(db["palettes"], product["recommended_palettes"], product_id, theme, style_override)
    fonts = select_many(db["fonts"], product["recommended_fonts"])
    charts = select_many(db["charts"], product["recommended_charts"])
    patterns = select_many(db["patterns"], product["recommended_patterns"])

    confidence = round(
        min(
            0.99,
            0.52
            + slug_score(style, product_id, theme, style_override) * 0.24
            + slug_score(palette, product_id, theme, style_override) * 0.24,
        ),
        2,
    )

    return {
        "version": db["version"],
        "product": product["name"],
        "theme": theme.title(),
        "style": style["name"],
        "confidence": confidence,
        "palette": palette["colors"],
        "fonts": fonts,
        "charts": charts,
        "landing_pattern": patterns[0] if patterns else "",
        "supporting_patterns": patterns[1:],
        "ux_notes": product["ux_notes"],
    }


def render_table(rows: list[tuple[str, str]]) -> str:
    width = max(len(left) for left, _ in rows) + 2
    return "\n".join(f" {left.ljust(width)} {right}" for left, right in rows)


def recommendation_to_table(payload: dict[str, Any]) -> str:
    rows = [
        ("Product", str(payload["product"])),
        ("Theme", str(payload["theme"])),
        ("Style", str(payload["style"])),
        ("Confidence", f'{payload["confidence"]:.2f}'),
        ("Fonts", " + ".join(payload["fonts"])),
        ("Charts", ", ".join(payload["charts"])),
        ("Landing Pattern", str(payload["landing_pattern"])),
        ("Primary", payload["palette"]["primary"]),
        ("Secondary", payload["palette"]["secondary"]),
        ("CTA", payload["palette"]["cta"]),
        ("Background", payload["palette"]["background"]),
        ("Text", payload["palette"]["text"]),
    ]
    return render_table(rows)


def search_category(db: dict[str, Any], category: str, tag: str | None) -> list[dict[str, Any]]:
    category_map = {
        "styles": db["styles"],
        "palettes": db["palettes"],
        "fonts": db["fonts"],
        "charts": db["charts"],
        "patterns": db["patterns"],
    }
    if category not in category_map:
        valid = ", ".join(category_map)
        raise SystemExit(f"Unknown category '{category}'. Valid categories: {valid}")
    items = category_map[category]
    if not tag:
        return items
    tag = tag.lower()
    return [item for item in items if tag in {t.lower() for t in item.get("tags", [])}]


def search_to_table(items: list[dict[str, Any]]) -> str:
    rows = []
    for item in items:
        rows.append((item["name"], ", ".join(item.get("tags", []))))
    return render_table(rows or [("Result", "No matches")])


def parse_json_input(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8-sig"))
    return json.loads(path_str)


def srgb_to_linear(value: float) -> float:
    if value <= 0.03928:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    cleaned = hex_color.strip().lstrip("#")
    if len(cleaned) != 6:
        raise ValueError(f"Expected 6-digit hex color, got '{hex_color}'")
    return tuple(int(cleaned[i:i + 2], 16) / 255 for i in range(0, 6, 2))


def contrast_ratio(hex_a: str, hex_b: str) -> float:
    rgb_a = hex_to_rgb(hex_a)
    rgb_b = hex_to_rgb(hex_b)
    lum_a = 0.2126 * srgb_to_linear(rgb_a[0]) + 0.7152 * srgb_to_linear(rgb_a[1]) + 0.0722 * srgb_to_linear(rgb_a[2])
    lum_b = 0.2126 * srgb_to_linear(rgb_b[0]) + 0.7152 * srgb_to_linear(rgb_b[1]) + 0.0722 * srgb_to_linear(rgb_b[2])
    lighter = max(lum_a, lum_b)
    darker = min(lum_a, lum_b)
    return (lighter + 0.05) / (darker + 0.05)


def validate_contrast(payload: dict[str, Any]) -> dict[str, Any]:
    palette = payload.get("palette", payload)
    background = palette["background"]
    surface = palette.get("surface", background)
    text = palette["text"]
    pairs = {
        "text/background": (text, background),
        "primary/background": (palette["primary"], background),
        "secondary/background": (palette["secondary"], background),
        "cta/background": (palette["cta"], background),
        "text/surface": (text, surface),
    }
    results = []
    for label, (fg, bg) in pairs.items():
        ratio = round(contrast_ratio(fg, bg), 2)
        results.append(
            {
                "pair": label,
                "foreground": fg,
                "background": bg,
                "contrast": ratio,
                "passes_aa": ratio >= 4.5,
            }
        )
    return {"mode": "contrast", "results": results}


def to_px(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower().replace("px", "")
    return float(text)


def validate_spacing(payload: dict[str, Any]) -> dict[str, Any]:
    spacing = payload.get("spacing")
    if spacing is None:
        raise SystemExit("Spacing validation expects a JSON object with a 'spacing' array or object.")
    if isinstance(spacing, dict):
        values = [to_px(value) for _, value in sorted(spacing.items())]
    else:
        values = [to_px(value) for value in spacing]

    issues = []
    if values != sorted(values):
        issues.append("Spacing scale is not ascending.")
    if any((value % 4) != 0 for value in values):
        issues.append("Spacing values should stay on a 4px grid.")
    if len(values) > 1:
        ratios = [round(values[i + 1] / values[i], 2) for i in range(len(values) - 1) if values[i] > 0]
    else:
        ratios = []

    return {
        "mode": "spacing",
        "values": values,
        "ratios": ratios,
        "passes": not issues,
        "issues": issues,
    }


def build_prompt(db: dict[str, Any], product: str, theme: str, style_override: str | None = None) -> str:
    rec = build_recommendation(db, product, theme, style_override)
    notes = "; ".join(rec["ux_notes"])
    return (
        f"Design a {rec['theme'].lower()} {rec['product']} interface using {rec['style']}."
        f" Use palette {rec['palette']['primary']}, {rec['palette']['secondary']}, {rec['palette']['cta']},"
        f" background {rec['palette']['background']}, and text {rec['palette']['text']}."
        f" Pair {rec['fonts'][0]} with {rec['fonts'][1] if len(rec['fonts']) > 1 else rec['fonts'][0]}."
        f" Prioritize {rec['landing_pattern']} and charts such as {', '.join(rec['charts'])}."
        f" UX guardrails: {notes}"
    )


def emit(payload: Any, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(payload, indent=2))
    else:
        if isinstance(payload, dict) and "palette" in payload:
            print(recommendation_to_table(payload))
        elif isinstance(payload, dict) and payload.get("mode") == "contrast":
            rows = []
            for item in payload["results"]:
                rows.append((item["pair"], f"{item['contrast']}:1  {'PASS' if item['passes_aa'] else 'FAIL'}"))
            print(render_table(rows))
        elif isinstance(payload, dict) and payload.get("mode") == "spacing":
            rows = [
                ("Values", ", ".join(str(int(v) if v.is_integer() else v) for v in payload["values"])),
                ("Ratios", ", ".join(str(r) for r in payload["ratios"]) or "n/a"),
                ("Passes", "YES" if payload["passes"] else "NO"),
                ("Issues", "; ".join(payload["issues"]) or "None"),
            ]
            print(render_table(rows))
        elif isinstance(payload, list):
            print(search_to_table(payload))
        else:
            print(str(payload))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uipro", description="UI design recommendation engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    recommend = subparsers.add_parser("recommend", help="Recommend a UI direction for a product")
    recommend.add_argument("--product", required=True)
    recommend.add_argument("--theme", required=True)
    recommend.add_argument("--style")
    recommend.add_argument("--format", choices=["json", "table"], default="json")

    validate = subparsers.add_parser("validate", help="Validate contrast or spacing")
    group = validate.add_mutually_exclusive_group(required=True)
    group.add_argument("--contrast")
    group.add_argument("--spacing")
    validate.add_argument("--format", choices=["json", "table"], default="json")

    prompt = subparsers.add_parser("prompt", help="Generate a prompt for an AI design agent")
    prompt.add_argument("--product", required=True)
    prompt.add_argument("--theme", required=True)
    prompt.add_argument("--style")

    db_cmd = subparsers.add_parser("db", help="Search the design database")
    db_sub = db_cmd.add_subparsers(dest="db_command", required=True)
    db_search = db_sub.add_parser("search", help="Search a resource category")
    db_search.add_argument("category", choices=["styles", "palettes", "fonts", "charts", "patterns"])
    db_search.add_argument("--tag")
    db_search.add_argument("--format", choices=["json", "table"], default="table")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    db = load_db()

    if args.command == "recommend":
        payload = build_recommendation(db, args.product, args.theme, args.style)
        emit(payload, args.format)
        return 0

    if args.command == "validate":
        if args.contrast:
            payload = validate_contrast(parse_json_input(args.contrast))
        else:
            payload = validate_spacing(parse_json_input(args.spacing))
        emit(payload, args.format)
        return 0

    if args.command == "prompt":
        print(build_prompt(db, args.product, args.theme, args.style))
        return 0

    if args.command == "db" and args.db_command == "search":
        payload = search_category(db, args.category, args.tag)
        emit(payload, args.format)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
