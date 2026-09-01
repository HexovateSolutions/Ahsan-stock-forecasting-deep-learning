import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Equity Forecasting Terminal | Deep Learning Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Institutional Palette ────────────────────────────────────────────────────
C = {
    "base":      "#0D1117",   # deep obsidian background
    "surface":   "#161B22",   # elevated card surface
    "border":    "#21262D",   # 1-px structural borders
    "border2":   "#30363D",   # secondary / grid borders
    "text":      "#F0F6FC",   # primary off-white text
    "muted":     "#8B949E",   # secondary captions / unit labels
    "blue":      "#2F81F7",   # primary institutional accent (selections, links)
    "green":     "#3FB950",   # positive / gain / high R²
    "red":       "#F85149",   # negative / loss / low R²
    "violet":    "#A371F7",   # secondary series (Transformer / ANN)
    "gold":      "#E3B341",   # tertiary series (ANN / warning)
}

# ── Google Font + CSS Overrides ──────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ---------- Global Reset ---------- */
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {C['base']};
    color: {C['text']};
}}
.stApp {{ background-color: {C['base']}; }}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {{
    background-color: {C['surface']};
    border-right: 1px solid {C['border']};
}}
[data-testid="stSidebar"] * {{ color: {C['text']}; }}
[data-testid="stSidebar"] .stSelectbox label {{
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {C['muted']};
}}

/* ---------- Tabs ---------- */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    background-color: {C['surface']};
    border-bottom: 1px solid {C['border']};
    gap: 0;
}}
[data-testid="stTabs"] [data-baseweb="tab"] {{
    background-color: transparent;
    border-radius: 0;
    padding: 10px 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {C['muted']};
    border-bottom: 2px solid transparent;
}}
[data-testid="stTabs"] [aria-selected="true"] {{
    color: {C['text']};
    border-bottom: 2px solid {C['blue']};
    background-color: transparent;
}}
[data-testid="stTabs"] [data-baseweb="tab-panel"] {{
    background-color: {C['base']};
    padding-top: 1.5rem;
}}

/* ---------- Metric cards ---------- */
[data-testid="stMetric"] {{
    background-color: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 6px;
    padding: 16px 18px;
}}
[data-testid="stMetric"] label {{
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: {C['muted']} !important;
}}
[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    font-size: 1.45rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: {C['text']};
}}
[data-testid="stMetricDelta"] {{
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
}}

/* ---------- DataFrames ---------- */
[data-testid="stDataFrame"] {{
    border: 1px solid {C['border']};
    border-radius: 6px;
}}

/* ---------- Dividers ---------- */
hr {{
    border: none;
    border-top: 1px solid {C['border']};
    margin: 1.2rem 0;
}}

/* ---------- Section headers ---------- */
h1 {{ font-size: 1.3rem; font-weight: 700; color: {C['text']}; letter-spacing: -0.01em; }}
h2 {{ font-size: 1.0rem; font-weight: 600; color: {C['text']}; letter-spacing: 0.01em; }}
h3 {{ font-size: 0.85rem; font-weight: 600; color: {C['muted']}; letter-spacing: 0.08em; text-transform: uppercase; }}

/* ---------- KPI header strip ---------- */
.kpi-strip {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 6px;
    padding: 14px 20px;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 40px;
    flex-wrap: wrap;
}}
.kpi-strip .kpi-label {{
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {C['muted']};
    margin-bottom: 2px;
}}
.kpi-strip .kpi-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: {C['text']};
}}
.kpi-strip .kpi-positive {{ color: {C['green']}; }}
.kpi-strip .kpi-negative {{ color: {C['red']}; }}
.kpi-strip .kpi-neutral  {{ color: {C['blue']}; }}
.kpi-sep {{ width: 1px; height: 36px; background: {C['border']}; flex-shrink: 0; }}
.kpi-brand {{
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: {C['blue']};
    margin-right: auto;
}}

/* ---------- Methodology cards ---------- */
.method-card {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-left: 3px solid {C['blue']};
    border-radius: 0 6px 6px 0;
    padding: 14px 18px;
    margin-bottom: 10px;
}}
.method-card h4 {{
    font-size: 0.80rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {C['blue']};
    margin: 0 0 6px 0;
}}
.method-card p {{
    font-size: 0.83rem;
    color: {C['muted']};
    margin: 0;
    line-height: 1.55;
}}

/* ---------- Disclaimer box ---------- */
.disclaimer {{
    background: rgba(248, 81, 73, 0.07);
    border: 1px solid rgba(248, 81, 73, 0.35);
    border-radius: 6px;
    padding: 12px 18px;
    font-size: 0.78rem;
    color: {C['muted']};
    line-height: 1.6;
    margin-top: 1rem;
}}
.disclaimer strong {{ color: {C['red']}; font-weight: 600; }}

/* ---------- Empty/error state ---------- */
.empty-state {{
    background: {C['surface']};
    border: 1px dashed {C['border2']};
    border-radius: 6px;
    padding: 40px 24px;
    text-align: center;
    color: {C['muted']};
    font-size: 0.83rem;
}}
.empty-state .es-code {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    background: {C['base']};
    border: 1px solid {C['border']};
    border-radius: 4px;
    padding: 2px 8px;
    color: {C['red']};
    display: inline-block;
    margin-top: 6px;
}}
</style>
""", unsafe_allow_html=True)


# ── Plotly base layout (reused across all charts) ───────────────────────────
def chart_layout(**kwargs):
    base = dict(
        paper_bgcolor=C["base"],
        plot_bgcolor=C["base"],
        font=dict(family="Inter, sans-serif", color=C["muted"], size=11),
        xaxis=dict(
            gridcolor=C["border2"], gridwidth=1, linecolor=C["border"],
            tickfont=dict(size=10, color=C["muted"]),
            title_font=dict(size=10, color=C["muted"], weight=600),
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor=C["border2"], gridwidth=1, linecolor=C["border"],
            tickfont=dict(size=10, color=C["muted"]),
            title_font=dict(size=10, color=C["muted"], weight=600),
            zeroline=False,
        ),
        legend=dict(
            bgcolor=C["surface"], bordercolor=C["border"], borderwidth=1,
            font=dict(size=10, color=C["muted"]),
        ),
        margin=dict(l=48, r=20, t=32, b=48),
        hoverlabel=dict(
            bgcolor=C["surface"], bordercolor=C["border"],
            font=dict(family="JetBrains Mono, monospace", size=11, color=C["text"]),
        ),
    )
    base.update(kwargs)
    return base


# ── Data ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    results  = pd.read_csv("results/final_all_models_all_stocks.csv")
    forecasts = pd.read_csv("results/future_10day_forecasts.csv", index_col=0)
    walk_fwd  = pd.read_csv("results/walk_forward_validation_AAPL.csv")
    tuning    = pd.read_csv("results/tuning_results_AAPL.csv")
    return results, forecasts, walk_fwd, tuning

results_df, forecast_df, walk_df, tuning_df = load_data()

tickers = sorted(results_df["Ticker"].unique())
models  = sorted(results_df["Model"].unique())

# Model color mapping — consistent across all charts
MODEL_COLORS = {
    "GRU":         C["blue"],
    "LSTM":        C["green"],
    "Transformer": C["violet"],
    "ANN":         C["gold"],
}


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-size:0.65rem;font-weight:700;letter-spacing:0.14em;"
        f"text-transform:uppercase;color:{C['blue']};margin-bottom:18px;'>"
        "Equity Forecasting Terminal</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='font-size:0.62rem;letter-spacing:0.08em;text-transform:uppercase;"
        f"color:{C['muted']};margin-bottom:4px;font-weight:600;'>Asset</div>",
        unsafe_allow_html=True,
    )
    selected_ticker = st.selectbox("Asset", tickers, index=0, label_visibility="collapsed")

    st.markdown(
        f"<div style='font-size:0.62rem;letter-spacing:0.08em;text-transform:uppercase;"
        f"color:{C['muted']};margin-bottom:4px;margin-top:12px;font-weight:600;'>Model Architecture</div>",
        unsafe_allow_html=True,
    )
    default_model_idx = models.index("GRU") if "GRU" in models else 0
    selected_model = st.selectbox("Model", models, index=default_model_idx, label_visibility="collapsed")

    st.markdown(f"<hr style='border-top:1px solid {C['border']};margin:18px 0;'>", unsafe_allow_html=True)

    st.markdown(
        f"<div style='font-size:0.62rem;font-weight:700;letter-spacing:0.10em;"
        f"text-transform:uppercase;color:{C['muted']};margin-bottom:10px;'>Dataset Reference</div>",
        unsafe_allow_html=True,
    )

    meta_rows = [
        ("Tickers",    "AAPL, ABT, JPM, XOM, WMT"),
        ("Period",     "10 Years (daily OHLCV)"),
        ("Source",     "Yahoo Finance — yFinance"),
        ("Architectures", "ANN, LSTM, GRU, Transformer"),
        ("Forecast Horizon", "10 Trading Days"),
        ("Validation", "Walk-Forward (5 Folds)"),
    ]
    for label, val in meta_rows:
        st.markdown(
            f"<div style='margin-bottom:7px;'>"
            f"<span style='font-size:0.60rem;color:{C['muted']};text-transform:uppercase;"
            f"letter-spacing:0.08em;font-weight:600;'>{label}</span><br>"
            f"<span style='font-size:0.75rem;color:{C['text']};font-family:\"JetBrains Mono\",monospace;'>{val}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── Compute KPI values for the top bar ───────────────────────────────────────
sel_row = results_df[
    (results_df["Ticker"] == selected_ticker) &
    (results_df["Model"]  == selected_model)
].iloc[0]

fc_available = selected_ticker in forecast_df.columns
if fc_available:
    fc_series = forecast_df[selected_ticker]
    fc_day1   = fc_series.iloc[0]
    fc_day10  = fc_series.iloc[-1]
    fc_delta  = ((fc_day10 - fc_day1) / fc_day1) * 100
    trend_dir = "+" if fc_delta >= 0 else ""
    trend_cls = "kpi-positive" if fc_delta >= 0 else "kpi-negative"
    fc_delta_str = f"{trend_dir}{fc_delta:.2f}%"
    fc_arrow  = "▲" if fc_delta >= 0 else "▼"
else:
    fc_delta_str = "N/A"
    trend_cls    = "kpi-neutral"
    fc_arrow     = ""

r2_cls   = "kpi-positive" if sel_row["R2"] >= 0.85 else ("kpi-negative" if sel_row["R2"] < 0.5 else "kpi-neutral")
mape_cls = "kpi-positive" if sel_row["MAPE"] < 3.0 else ("kpi-negative" if sel_row["MAPE"] > 7.0 else "kpi-neutral")


# ── Top KPI Strip ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="kpi-strip">
  <span class="kpi-brand">Quantitative Equity Forecasting Dashboard</span>

  <div>
    <div class="kpi-label">Asset</div>
    <div class="kpi-value kpi-neutral">{selected_ticker}</div>
  </div>
  <div class="kpi-sep"></div>

  <div>
    <div class="kpi-label">Model Architecture</div>
    <div class="kpi-value kpi-neutral">{selected_model}</div>
  </div>
  <div class="kpi-sep"></div>

  <div>
    <div class="kpi-label">10-Day Forecast Drift {fc_arrow}</div>
    <div class="kpi-value {trend_cls}">{fc_delta_str}</div>
  </div>
  <div class="kpi-sep"></div>

  <div>
    <div class="kpi-label">R&sup2; Accuracy</div>
    <div class="kpi-value {r2_cls}">{sel_row['R2']:.4f}</div>
  </div>
  <div class="kpi-sep"></div>

  <div>
    <div class="kpi-label">MAPE</div>
    <div class="kpi-value {mape_cls}">{sel_row['MAPE']:.2f}%</div>
  </div>
  <div class="kpi-sep"></div>

  <div>
    <div class="kpi-label">RMSE</div>
    <div class="kpi-value kpi-neutral">{sel_row['RMSE']:.5f}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_forecast, tab_perf, tab_validation, tab_method = st.tabs([
    "10-Day Price Forecast",
    "Model Performance",
    "Walk-Forward & Tuning",
    "Methodology & Disclosure",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — 10-Day Forecast & Trend Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab_forecast:
    st.markdown(
        f"<h2 style='margin-bottom:4px;'>10-Day Forward Price Projection — {selected_ticker}</h2>"
        f"<p style='font-size:0.78rem;color:{C['muted']};margin-top:0;margin-bottom:1rem;'>"
        "Generated by GRU model trained on 10-year daily OHLCV data. "
        "Values are normalised (MinMaxScaler); absolute price reconstruction requires the original scaler.</p>",
        unsafe_allow_html=True,
    )

    if fc_available:
        days_labels = [f"Day {i+1}" for i in range(len(fc_series))]
        fc_values   = fc_series.values

        # Compute daily deltas
        deltas = np.diff(fc_values, prepend=fc_values[0])
        pct_chg = np.where(fc_values[:-1] != 0,
                           (deltas[1:] / fc_values[:-1]) * 100,
                           0.0)
        pct_chg = np.insert(pct_chg, 0, 0.0)

        # ── Forecast chart
        fig_fc = go.Figure()

        # Area fill beneath line
        fig_fc.add_trace(go.Scatter(
            x=days_labels, y=fc_values,
            mode="none", fill="tozeroy",
            fillcolor=f"rgba(47,129,247,0.07)",
            showlegend=False, hoverinfo="skip",
        ))
        # Main projection line
        fig_fc.add_trace(go.Scatter(
            x=days_labels, y=fc_values,
            mode="lines+markers",
            line=dict(color=C["blue"], width=2.5),
            marker=dict(size=7, color=C["blue"], line=dict(color=C["base"], width=2)),
            name="Projected Price",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Scaled Price: %{y:.6f}<extra></extra>"
            ),
        ))

        fig_fc.update_layout(
            **chart_layout(
                height=380,
                xaxis_title="TRADING DAY",
                yaxis_title="NORMALISED PRICE",
                yaxis_tickformat=".5f",
                showlegend=False,
            )
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # ── Projection data table
        st.markdown(
            f"<h3 style='margin-top:1.2rem;margin-bottom:10px;'>Daily Projection Grid</h3>",
            unsafe_allow_html=True,
        )

        table_rows = []
        for i, (d, v, p) in enumerate(zip(days_labels, fc_values, pct_chg)):
            direction = "+" if p >= 0 else ""
            table_rows.append({
                "Trading Day": d,
                "Scaled Forecast Price": f"{v:.6f}",
                "Daily Change (%)": f"{direction}{p:.4f}%",
                "Direction": "Positive" if p > 0 else ("Negative" if p < 0 else "Flat"),
            })

        table_df = pd.DataFrame(table_rows)

        def color_direction(val):
            if val == "Positive":
                return f"color: {C['green']}; font-weight: 600;"
            elif val == "Negative":
                return f"color: {C['red']}; font-weight: 600;"
            return f"color: {C['muted']};"

        styled_table = (
            table_df.style
            .map(color_direction, subset=["Direction"])
            .set_properties(**{
                "font-family": "JetBrains Mono, monospace",
                "font-size": "0.80rem",
            })
        )
        st.dataframe(styled_table, use_container_width=True, hide_index=True)

    else:
        st.markdown(
            f"<div class='empty-state'>"
            f"<div style='font-size:0.90rem;font-weight:600;color:{C['text']};margin-bottom:6px;'>"
            f"No Forecast Data Available</div>"
            f"<div>Forecast output for <b>{selected_ticker}</b> was not found in "
            f"<span class='es-code'>results/future_10day_forecasts.csv</span>.</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Performance & Comparative Analytics
# ═══════════════════════════════════════════════════════════════════════════════
with tab_perf:
    st.markdown(
        f"<h2 style='margin-bottom:4px;'>"
        f"Performance Metrics — {selected_model} on {selected_ticker}</h2>"
        f"<p style='font-size:0.78rem;color:{C['muted']};margin-top:0;margin-bottom:1rem;'>"
        f"Test-set evaluation on held-out data. Model trained for {int(sel_row['Epochs_Trained'])} epochs "
        f"with early stopping.</p>",
        unsafe_allow_html=True,
    )

    # ── 5 metric cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAE",      f"{sel_row['MAE']:.5f}")
    c2.metric("RMSE",     f"{sel_row['RMSE']:.5f}")
    c3.metric("MAPE",     f"{sel_row['MAPE']:.2f}%")
    c4.metric("R² Score", f"{sel_row['R2']:.4f}")
    c5.metric("MSE",      f"{sel_row['MSE']:.6f}")

    st.markdown(f"<hr style='border-top:1px solid {C['border']};margin:1.4rem 0;'>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    # ── Cross-model R² bar chart
    with col_left:
        st.markdown(
            f"<h3 style='margin-bottom:10px;'>R&sup2; by Architecture — {selected_ticker}</h3>",
            unsafe_allow_html=True,
        )
        sub = results_df[results_df["Ticker"] == selected_ticker].copy()
        sub["color"] = sub["Model"].map(MODEL_COLORS)

        fig_bar = go.Figure()
        for _, r in sub.iterrows():
            fig_bar.add_trace(go.Bar(
                x=[r["Model"]], y=[r["R2"]],
                marker_color=MODEL_COLORS.get(r["Model"], C["blue"]),
                marker_line=dict(color=C["border"], width=1),
                name=r["Model"],
                showlegend=True,
                hovertemplate=(
                    f"<b>{r['Model']}</b><br>"
                    f"R²: {r['R2']:.4f}<br>"
                    f"MAPE: {r['MAPE']:.2f}%<extra></extra>"
                ),
            ))

        fig_bar.update_layout(
            **chart_layout(
                height=340,
                xaxis_title="ARCHITECTURE",
                yaxis_title="R² SCORE",
                showlegend=False,
                barmode="group",
                yaxis_tickformat=".3f",
            )
        )
        # Reference line at 0
        fig_bar.add_hline(y=0, line_width=1, line_color=C["border2"])
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── R² heatmap — all tickers x all models
    with col_right:
        st.markdown(
            f"<h3 style='margin-bottom:10px;'>R&sup2; Heatmap — All Assets × Architectures</h3>",
            unsafe_allow_html=True,
        )
        pivot = results_df.pivot(index="Ticker", columns="Model", values="R2")
        ordered_models = [m for m in ["GRU", "LSTM", "Transformer", "ANN"] if m in pivot.columns]
        pivot = pivot[ordered_models]

        fig_heat = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[
                [0.0,  C["red"]],
                [0.5,  "#E3B341"],
                [1.0,  C["green"]],
            ],
            zmid=0.5,
            zmin=pivot.values.min(),
            zmax=pivot.values.max(),
            text=np.round(pivot.values, 3),
            texttemplate="%{text}",
            textfont=dict(family="JetBrains Mono, monospace", size=11),
            colorbar=dict(
                title=dict(text="R²", font=dict(size=10, color=C["muted"])),
                tickfont=dict(size=9, color=C["muted"]),
                bgcolor=C["surface"],
                bordercolor=C["border"],
                thickness=12,
            ),
        ))
        fig_heat.update_layout(
            **chart_layout(height=340, xaxis_title="ARCHITECTURE", yaxis_title="TICKER")
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown(f"<hr style='border-top:1px solid {C['border']};margin:1.4rem 0;'>", unsafe_allow_html=True)

    # ── Full results table
    st.markdown(
        f"<h3 style='margin-bottom:10px;'>Comprehensive Performance Table — All Assets & Architectures</h3>",
        unsafe_allow_html=True,
    )

    display_df = results_df.copy()
    display_df = display_df.rename(columns={"Epochs_Trained": "Epochs"})
    display_df["MAE"]   = display_df["MAE"].map(lambda x: f"{x:.5f}")
    display_df["MSE"]   = display_df["MSE"].map(lambda x: f"{x:.6f}")
    display_df["RMSE"]  = display_df["RMSE"].map(lambda x: f"{x:.5f}")
    display_df["MAPE"]  = display_df["MAPE"].map(lambda x: f"{x:.2f}%")
    display_df["R2"]    = display_df["R2"].map(lambda x: f"{x:.4f}")
    display_df["Epochs"] = display_df["Epochs"].astype(int)
    display_df = display_df.sort_values(["Ticker", "Model"])

    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Walk-Forward Validation & Hyperparameter Tuning
# ═══════════════════════════════════════════════════════════════════════════════
with tab_validation:
    st.markdown(
        f"<h2 style='margin-bottom:4px;'>Walk-Forward Validation — AAPL (GRU)</h2>"
        f"<p style='font-size:0.78rem;color:{C['muted']};margin-top:0;margin-bottom:1rem;'>"
        "Sequential 5-fold re-training protocol. Each fold advances the training window by one period, "
        "simulating realistic live-deployment conditions where the model cannot see future data.</p>",
        unsafe_allow_html=True,
    )

    col_wf1, col_wf2 = st.columns(2)

    with col_wf1:
        st.markdown(
            f"<h3 style='margin-bottom:10px;'>R&sup2; Stability Across Folds</h3>",
            unsafe_allow_html=True,
        )
        r2_colors = [C["green"] if v >= 0.85 else (C["gold"] if v >= 0.5 else C["red"]) for v in walk_df["R2"]]

        fig_wf1 = go.Figure()
        fig_wf1.add_trace(go.Bar(
            x=walk_df["Fold"].astype(str),
            y=walk_df["R2"],
            marker_color=r2_colors,
            marker_line=dict(color=C["border"], width=1),
            hovertemplate="<b>Fold %{x}</b><br>R²: %{y:.4f}<extra></extra>",
            name="R²",
        ))
        fig_wf1.add_hline(y=0, line_width=1, line_color=C["border2"])
        fig_wf1.update_layout(
            **chart_layout(
                height=320,
                xaxis_title="FOLD",
                yaxis_title="R² SCORE",
                showlegend=False,
                yaxis_tickformat=".3f",
                xaxis=dict(
                    gridcolor=C["border2"], gridwidth=1, linecolor=C["border"],
                    tickfont=dict(size=10, color=C["muted"]),
                    title_font=dict(size=10, color=C["muted"]),
                    zeroline=False,
                    tickprefix="Fold ",
                ),
            )
        )
        st.plotly_chart(fig_wf1, use_container_width=True)

    with col_wf2:
        st.markdown(
            f"<h3 style='margin-bottom:10px;'>MAPE Error Progression</h3>",
            unsafe_allow_html=True,
        )
        fig_wf2 = go.Figure()
        fig_wf2.add_trace(go.Scatter(
            x=walk_df["Fold"].astype(str),
            y=walk_df["MAPE"],
            mode="lines+markers",
            line=dict(color=C["red"], width=2.5),
            marker=dict(size=7, color=C["red"], line=dict(color=C["base"], width=2)),
            hovertemplate="<b>Fold %{x}</b><br>MAPE: %{y:.4f}%<extra></extra>",
            name="MAPE",
        ))
        fig_wf2.update_layout(
            **chart_layout(
                height=320,
                xaxis_title="FOLD",
                yaxis_title="MAPE (%)",
                showlegend=False,
                yaxis_tickformat=".2f",
                yaxis_ticksuffix="%",
            )
        )
        st.plotly_chart(fig_wf2, use_container_width=True)

    st.markdown(
        f"<h3 style='margin-bottom:10px;margin-top:1rem;'>Fold-by-Fold Metrics</h3>",
        unsafe_allow_html=True,
    )
    wf_display = walk_df.copy()
    wf_display.columns = [c.upper() for c in wf_display.columns]
    wf_display["MAE"]  = wf_display["MAE"].map(lambda x: f"{x:.5f}")
    wf_display["MSE"]  = wf_display["MSE"].map(lambda x: f"{x:.6f}")
    wf_display["RMSE"] = wf_display["RMSE"].map(lambda x: f"{x:.5f}")
    wf_display["MAPE"] = wf_display["MAPE"].map(lambda x: f"{x:.2f}%")
    wf_display["R2"]   = wf_display["R2"].map(lambda x: f"{x:.4f}")
    st.dataframe(wf_display, use_container_width=True, hide_index=True)

    st.markdown(f"<hr style='border-top:1px solid {C['border']};margin:1.6rem 0;'>", unsafe_allow_html=True)

    # ── Hyperparameter tuning section
    st.markdown(
        f"<h2 style='margin-bottom:4px;'>Hyperparameter Sensitivity Analysis — GRU on AAPL</h2>"
        f"<p style='font-size:0.78rem;color:{C['muted']};margin-top:0;margin-bottom:1rem;'>"
        "Isolated ablation study: each hyperparameter is varied while others are held at baseline. "
        "Final combined configuration is included for reference.</p>",
        unsafe_allow_html=True,
    )

    hp_colors = {
        "sequence_length": C["blue"],
        "batch_size":      C["green"],
        "dropout":         C["violet"],
        "optimizer":       C["gold"],
        "final_combined":  C["muted"],
    }

    fig_hp = go.Figure()
    for hp_name, grp in tuning_df.groupby("Hyperparameter"):
        clr = hp_colors.get(hp_name, C["muted"])
        fig_hp.add_trace(go.Bar(
            x=grp["Value"],
            y=grp["R2"],
            name=hp_name.replace("_", " ").title(),
            marker_color=clr,
            marker_line=dict(color=C["border"], width=1),
            hovertemplate=(
                f"<b>{hp_name}</b><br>"
                "Value: %{x}<br>"
                "R²: %{y:.4f}<extra></extra>"
            ),
        ))

    fig_hp.update_layout(
        **chart_layout(
            height=400,
            xaxis_title="HYPERPARAMETER VALUE",
            yaxis_title="R² SCORE",
            barmode="group",
            showlegend=True,
            yaxis_tickformat=".3f",
        )
    )
    st.plotly_chart(fig_hp, use_container_width=True)

    st.markdown(
        f"<h3 style='margin-bottom:10px;'>Hyperparameter Configuration Table</h3>",
        unsafe_allow_html=True,
    )
    tun_display = tuning_df.copy()
    tun_display.columns = [c.upper() for c in tun_display.columns]
    tun_display["MAE"]  = tun_display["MAE"].map(lambda x: f"{x:.5f}")
    tun_display["MSE"]  = tun_display["MSE"].map(lambda x: f"{x:.6f}")
    tun_display["RMSE"] = tun_display["RMSE"].map(lambda x: f"{x:.5f}")
    tun_display["MAPE"] = tun_display["MAPE"].map(lambda x: f"{x:.2f}%")
    tun_display["R2"]   = tun_display["R2"].map(lambda x: f"{x:.4f}")
    st.dataframe(tun_display, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Model Methodology & Disclosure
# ═══════════════════════════════════════════════════════════════════════════════
with tab_method:
    st.markdown(
        f"<h2 style='margin-bottom:4px;'>Model Methodology & Quantitative Disclosure</h2>"
        f"<p style='font-size:0.78rem;color:{C['muted']};margin-top:0;margin-bottom:1.2rem;'>"
        "Technical description of the forecasting pipeline architecture, data inputs, "
        "and evaluation protocol.</p>",
        unsafe_allow_html=True,
    )

    method_sections = [
        (
            "Data Inputs & Preprocessing",
            "Historical daily OHLCV (Open, High, Low, Close, Volume) data for five U.S. equities — "
            "AAPL, ABT, JPM, XOM, and WMT — sourced via the <code style='font-family:JetBrains Mono,"
            "monospace;font-size:0.78rem;color:#A371F7;'>yfinance</code> Python library. "
            "Approximately 10 years of trading days per asset (~2,500 observations). "
            "Closing prices are normalised using MinMaxScaler (range [0, 1]) prior to sequence construction. "
            "A sliding window of configurable length (default 60 trading days) is used to create "
            "input-output pairs for supervised learning.",
        ),
        (
            "Architecture Comparison",
            "<b style='color:#F0F6FC;'>ANN</b> — Fully-connected feedforward network; treats the flattened "
            "sequence as a vector input. Serves as a non-sequential baseline. "
            "<br><br>"
            "<b style='color:#3FB950;'>LSTM</b> — Long Short-Term Memory recurrent network; captures long-range "
            "temporal dependencies via gated cell states. "
            "<br><br>"
            "<b style='color:#2F81F7;'>GRU</b> — Gated Recurrent Unit; streamlined variant of LSTM with fewer "
            "parameters. Achieved the strongest average R² across all five assets. "
            "<br><br>"
            "<b style='color:#A371F7;'>Transformer</b> — Multi-head self-attention encoder; processes the full "
            "sequence in parallel. Performance is asset-dependent due to limited training data "
            "relative to model capacity.",
        ),
        (
            "Evaluation Protocol",
            "Models are evaluated on a held-out test split (typically the final 20% of the time series) "
            "using five standard regression metrics: MAE (Mean Absolute Error), MSE (Mean Squared Error), "
            "RMSE (Root MSE), MAPE (Mean Absolute Percentage Error), and R² (coefficient of determination). "
            "A 5-fold walk-forward validation scheme is additionally applied to the best-performing "
            "architecture (GRU on AAPL) to assess temporal stability.",
        ),
        (
            "Hyperparameter Tuning",
            "An isolated ablation approach is used: sequence length, batch size, dropout rate, and "
            "optimiser (Adam vs. RMSProp) are each varied independently while other settings remain at "
            "baseline. A final combined configuration is trained using the best-identified value per "
            "parameter. No automated search (e.g. grid search or Bayesian optimisation) is applied.",
        ),
        (
            "Known Limitations",
            "Forecast values are expressed in normalised units; recovering absolute price levels requires "
            "the original MinMaxScaler object, which is fitted per training run and asset. "
            "Models are trained on historical patterns only and do not incorporate macroeconomic "
            "indicators, news sentiment, earnings data, or market microstructure features. "
            "Performance degrades significantly for assets exhibiting high structural breaks "
            "(e.g. JPM, WMT in this dataset). Walk-forward fold 5 (AAPL) shows negative R², "
            "illustrating the risk of model drift over time without retraining.",
        ),
    ]

    for title, body in method_sections:
        st.markdown(
            f"<div class='method-card'>"
            f"<h4>{title}</h4>"
            f"<p>{body}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='disclaimer'>"
        "<strong>Financial Disclaimer:</strong> "
        "This dashboard is produced solely for educational and research purposes. "
        "All forecasts, metrics, and visualisations represent outputs of experimental deep learning models "
        "trained on historical price data. Past price behaviour is not indicative of future returns. "
        "Nothing contained herein constitutes financial advice, an investment recommendation, "
        "or a solicitation to buy or sell any security. "
        "Users should conduct independent analysis and consult a qualified financial professional "
        "before making any investment decisions."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Quick architecture reference table
    st.markdown(
        f"<h3 style='margin-top:1.4rem;margin-bottom:10px;'>Architecture Reference</h3>",
        unsafe_allow_html=True,
    )
    arch_df = pd.DataFrame([
        {"Architecture": "ANN",         "Type": "Feedforward",             "Strengths": "Fast training, simple baseline",         "Weaknesses": "Ignores temporal order"},
        {"Architecture": "LSTM",        "Type": "Recurrent",               "Strengths": "Long-range dependencies, gated memory",  "Weaknesses": "Slower training, vanishing gradient risk"},
        {"Architecture": "GRU",         "Type": "Recurrent (streamlined)", "Strengths": "Best R² overall, fewer parameters",      "Weaknesses": "Still sequential; no parallelism"},
        {"Architecture": "Transformer", "Type": "Attention-based",         "Strengths": "Parallel processing, global context",    "Weaknesses": "Data-hungry; underperforms on small datasets"},
    ])
    st.dataframe(arch_df, use_container_width=True, hide_index=True)


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown(f"<hr style='border-top:1px solid {C['border']};margin:1.8rem 0 0.6rem;'>", unsafe_allow_html=True)
st.markdown(
    f"<div style='font-size:0.65rem;color:{C['muted']};display:flex;justify-content:space-between;"
    f"flex-wrap:wrap;gap:6px;padding-bottom:0.5rem;'>"
    f"<span>Built with Streamlit · TensorFlow/Keras · yFinance · Plotly</span>"
    f"<span>Assets: AAPL · ABT · JPM · XOM · WMT &nbsp;|&nbsp; 10-year daily OHLCV &nbsp;|&nbsp; "
    f"Architectures: ANN · LSTM · GRU · Transformer</span>"
    f"</div>",
    unsafe_allow_html=True,
)
