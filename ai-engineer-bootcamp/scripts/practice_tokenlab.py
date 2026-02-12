#!/usr/bin/env python3
"""TokenLab Practice Script -- run in ~30 min.

Steps:
  1. Compare token counts: Spanish vs English
  2. Measure TTFT (streaming) for 100 / 500 / 2000 input tokens
  3. BudgetChecker demo with simulated pricing
  4. Temperature experiment: same prompt at 0.0, 0.7, 1.5
"""

from __future__ import annotations

import csv
import math
import os
import sys
import textwrap
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.tokenlab import (
    BudgetChecker,
    BudgetExceededError,
    LatencyResult,
    Pricing,
    count_tokens,
    estimate_cost,
    measure_latency,
)

OUTPUTS = ROOT / "outputs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def banner(title: str) -> None:
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  {title}")
    print(f"{sep}\n")


def choose_provider() -> str:
    """Let the user pick Gemini or Groq at startup."""
    print("Selecciona el proveedor LLM:")
    print("  1) Gemini  (default)")
    print("  2) Groq")
    choice = input("Opcion [1]: ").strip()
    if choice == "2":
        return "groq"
    return "gemini"


# ---------------------------------------------------------------------------
# Step 1 -- Token comparison ES vs EN
# ---------------------------------------------------------------------------

def step1_compare_tokens(provider: str) -> None:
    banner("PASO 1: Comparacion de tokens ES vs EN")

    text_es = "La inteligencia artificial esta cambiando el mundo."
    text_en = "Artificial intelligence is changing the world."

    tok_es = count_tokens(text_es, provider=provider)
    tok_en = count_tokens(text_en, provider=provider)
    ratio = tok_es / tok_en if tok_en else float("inf")

    print(f"  Texto ES : {text_es!r}")
    print(f"  Tokens ES: {tok_es}")
    print(f"  Texto EN : {text_en!r}")
    print(f"  Tokens EN: {tok_en}")
    print(f"  Ratio ES/EN: {ratio:.2f}")


# ---------------------------------------------------------------------------
# Step 2 -- TTFT streaming
# ---------------------------------------------------------------------------

def step2_ttft_streaming(provider: str) -> None:
    banner("PASO 2: TTFT Streaming (100 / 500 / 2000 tokens)")

    base = "Responde brevemente: "
    unit = "hola mundo "

    base_tok = count_tokens(base, provider=provider)
    unit_tok = count_tokens(unit, provider=provider)
    print(f"  Base tokens: {base_tok}  |  Unit tokens: {unit_tok}")

    targets = [100, 500, 2000]
    results: list[dict] = []

    for target in targets:
        reps = max(1, math.ceil((target - base_tok) / max(unit_tok, 1)))
        prompt = base + unit * reps
        real_tok = count_tokens(prompt, provider=provider)

        print(f"\n  --- Target ~{target} tokens  (real: {real_tok}) ---")

        lr: LatencyResult = measure_latency(
            prompt,
            stream=True,
            generation_config={"max_output_tokens": 100},
            provider=provider,
        )

        ttft_str = f"{lr.ttft_s:.4f} s" if lr.ttft_s is not None else "N/A"
        tps_str = f"{lr.tps:.1f}" if lr.tps is not None else "N/A"

        print(f"  Input tokens : {lr.input_tokens}")
        print(f"  Output tokens: {lr.output_tokens}")
        print(f"  TTFT         : {ttft_str}")
        print(f"  Total        : {lr.total_s:.4f} s")
        print(f"  TPS          : {tps_str}")

        results.append({
            "target_tokens": target,
            "real_input_tokens": real_tok,
            "ttft_s": lr.ttft_s,
            "total_s": round(lr.total_s, 4),
            "tps": round(lr.tps, 1) if lr.tps else None,
        })

    # -- Save CSV --
    OUTPUTS.mkdir(exist_ok=True)
    csv_path = OUTPUTS / "ttft_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  CSV guardado en: {csv_path}")

    # -- Plot --
    _plot_ttft(results)


def _plot_ttft(results: list[dict]) -> None:
    """Generate TTFT vs size chart (PNG). Falls back gracefully."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [matplotlib no disponible -- grafica omitida, CSV listo]")
        return

    x = [r["real_input_tokens"] for r in results]
    y = [r["ttft_s"] or 0.0 for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, "o-", linewidth=2, markersize=8, color="#2563eb")
    ax.set_xlabel("Input Tokens")
    ax.set_ylabel("TTFT (s)")
    ax.set_title("Time to First Token vs Input Size")
    ax.grid(True, alpha=0.3)

    png_path = OUTPUTS / "ttft_vs_size.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grafica guardada en: {png_path}")


# ---------------------------------------------------------------------------
# Step 3 -- BudgetChecker
# ---------------------------------------------------------------------------

def step3_budget_checker() -> None:
    banner("PASO 3: BudgetChecker ($0.05/request)")

    # Simulated pricing so the budget can actually be exceeded
    pricing = Pricing(input_per_1k=0.01, output_per_1k=0.03)
    checker = BudgetChecker(max_cost_usd=0.05, pricing=pricing, strict=True)

    # -- Small request (should pass) --
    sm_in, sm_out = 100, 50
    res_ok = checker.check(sm_in, sm_out)
    print(f"  Solicitud pequena ({sm_in} in / {sm_out} out):")
    print(f"    {res_ok}")

    # -- Large request (should exceed) --
    lg_in, lg_out = 2000, 2000
    print(f"\n  Solicitud grande ({lg_in} in / {lg_out} out):")
    try:
        checker.check(lg_in, lg_out)
    except BudgetExceededError as exc:
        print(f"    BudgetExceededError: {exc}")

    # -- Non-strict mode --
    soft = BudgetChecker(max_cost_usd=0.05, pricing=pricing, strict=False)
    res_warn = soft.check(lg_in, lg_out)
    print(f"\n  Modo no estricto (misma solicitud grande):")
    print(f"    {res_warn}")


# ---------------------------------------------------------------------------
# Step 4 -- Temperature experiment
# ---------------------------------------------------------------------------

def step4_temperature_experiment(provider: str) -> None:
    banner("PASO 4: Experimento de Temperature (0.0 / 0.7 / 1.5)")

    prompt = "Genera un poema de 250 palabras sobre LLMs y los tokens."
    temps = [0.0, 0.7, 1.5]
    poems: list[tuple[float, str, int, float]] = []  # (temp, text, tokens, time)

    for temp in temps:
        print(f"  Generando con temperature={temp} ...")
        lr = measure_latency(
            prompt,
            stream=False,
            generation_config={"temperature": temp, "max_output_tokens": 1024},
            provider=provider,
        )
        poems.append((temp, lr.output_text, lr.output_tokens or 0, lr.total_s))
        print(f"    -> {lr.output_tokens} tokens, {lr.total_s:.2f} s")

    _display_poems(poems)


def _display_poems(poems: list[tuple[float, str, int, float]]) -> None:
    """Show 3 poems side-by-side if terminal is wide, otherwise sequentially."""
    try:
        term_w = os.get_terminal_size().columns
    except OSError:
        term_w = 80

    col_w = max(28, (term_w - 8) // 3)

    if term_w >= 100:
        _display_side_by_side(poems, col_w)
    else:
        _display_sequential(poems)


def _display_sequential(poems: list[tuple[float, str, int, float]]) -> None:
    for temp, text, tok, sec in poems:
        print(f"\n{'─' * 60}")
        print(f"  TEMPERATURE = {temp}  ({tok} tokens, {sec:.2f}s)")
        print(f"{'─' * 60}")
        print(text)


def _display_side_by_side(poems: list[tuple[float, str, int, float]], col_w: int) -> None:
    wrapped: list[list[str]] = []
    for temp, text, tok, sec in poems:
        lines = [f" TEMP={temp} ({tok}tok, {sec:.1f}s) ", "─" * col_w]
        for paragraph in text.split("\n"):
            lines.extend(textwrap.wrap(paragraph, width=col_w) or [""])
        wrapped.append(lines)

    max_lines = max(len(c) for c in wrapped)
    for col in wrapped:
        col.extend([""] * (max_lines - len(col)))

    print()
    for i in range(max_lines):
        row = " | ".join(wrapped[j][i].ljust(col_w) for j in range(len(wrapped)))
        print(f"  {row}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    provider = choose_provider()
    print(f"\n  Proveedor seleccionado: {provider.upper()}\n")

    step1_compare_tokens(provider)
    step2_ttft_streaming(provider)
    step3_budget_checker()  # no API calls needed
    step4_temperature_experiment(provider)

    banner("PRACTICA COMPLETADA")
    print("  Revisa la carpeta outputs/ para CSV y graficas.\n")


if __name__ == "__main__":
    main()
