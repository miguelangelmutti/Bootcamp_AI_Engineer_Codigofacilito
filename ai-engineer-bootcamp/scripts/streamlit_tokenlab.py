"""Streamlit app -- visualize the difference between streaming and non-streaming LLM calls.

Run with:
    streamlit run scripts/streamlit_tokenlab.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

from core.tokenlab import (
    LatencyResult,
    measure_latency,
    stream_chunks,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TokenLab - Streaming vs No Streaming",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ TokenLab — Streaming vs No Streaming")
st.caption("Compara visualmente como se recibe la respuesta de un LLM con y sin streaming.")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuracion")

    provider = st.selectbox("Proveedor", ["gemini", "groq"], index=1)

    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, step=0.1)
    max_tokens = st.slider("Max output tokens", 64, 2048, 512, step=64)

    st.divider()
    st.markdown(
        "**Streaming** envia tokens conforme se generan → menor TTFT.\n\n"
        "**No streaming** espera a que se genere toda la respuesta → mayor latencia percibida."
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

prompt = st.text_area(
    "Prompt",
    value="Explica en 3 parrafos como funcionan los tokens en los modelos de lenguaje.",
    height=100,
)

run = st.button("Comparar", type="primary", use_container_width=True)

if run and prompt.strip():
    gen_config = {"temperature": temperature, "max_output_tokens": max_tokens}

    col_stream, col_no_stream = st.columns(2)

    # ------------------------------------------------------------------
    # Streaming (left) — text appears progressively
    # ------------------------------------------------------------------
    with col_stream:
        st.subheader("Con Streaming")
        stream_metrics: dict = {}
        full_stream_text = st.write_stream(
            stream_chunks(
                prompt,
                generation_config=gen_config,
                provider=provider,
                _metrics_out=stream_metrics,
            )
        )

    # ------------------------------------------------------------------
    # Non-streaming (right) — spinner then full text
    # ------------------------------------------------------------------
    with col_no_stream:
        st.subheader("Sin Streaming")
        with st.spinner("Esperando respuesta completa..."):
            no_stream_result: LatencyResult = measure_latency(
                prompt,
                stream=False,
                generation_config=gen_config,
                provider=provider,
            )
        st.markdown(no_stream_result.output_text)

    # ------------------------------------------------------------------
    # Metrics comparison
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Comparacion de metricas")

    s_ttft = stream_metrics.get("ttft_s")
    s_total = stream_metrics.get("total_s", 0)
    s_tps = stream_metrics.get("tps")
    s_in = stream_metrics.get("input_tokens")
    s_out = stream_metrics.get("output_tokens")

    ns_total = no_stream_result.total_s
    ns_tps = no_stream_result.tps
    ns_in = no_stream_result.input_tokens
    ns_out = no_stream_result.output_tokens

    mc1, mc2 = st.columns(2)

    with mc1:
        st.markdown("**Con Streaming**")
        st.metric("TTFT (s)", f"{s_ttft:.4f}" if s_ttft else "N/A")
        st.metric("Tiempo total (s)", f"{s_total:.4f}")
        st.metric("Tokens/s", f"{s_tps:.1f}" if s_tps else "N/A")
        st.metric("Input tokens", s_in or "—")
        st.metric("Output tokens", s_out or "—")

    with mc2:
        st.markdown("**Sin Streaming**")
        st.metric(
            "TTFT (s)",
            f"{ns_total:.4f}",
            help="Sin streaming, TTFT = tiempo total (toda la respuesta llega junta)",
        )
        st.metric("Tiempo total (s)", f"{ns_total:.4f}")
        st.metric("Tokens/s", f"{ns_tps:.1f}" if ns_tps else "N/A")
        st.metric("Input tokens", ns_in or "—")
        st.metric("Output tokens", ns_out or "—")

    # -- Bar chart --
    st.divider()
    st.subheader("Grafica comparativa")

    import pandas as pd

    chart_data = pd.DataFrame(
        {
            "Metrica": ["TTFT (s)", "Tiempo total (s)"],
            "Streaming": [s_ttft or 0, s_total],
            "No Streaming": [ns_total, ns_total],
        }
    )
    chart_data = chart_data.set_index("Metrica")
    st.bar_chart(chart_data)

    if s_ttft and ns_total:
        pct = ((ns_total - s_ttft) / ns_total) * 100
        if pct > 0:
            st.success(
                f"Con streaming el usuario empieza a leer **{pct:.0f}%** antes "
                f"({s_ttft:.3f}s vs {ns_total:.3f}s)."
            )
        else:
            st.info("En este caso el TTFT fue similar o mayor al tiempo sin streaming.")

elif run:
    st.warning("Escribe un prompt antes de comparar.")
