import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NeuroTrend | AI Stock Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════
# PALETTE — restrained, single accent family + functional status colors
# ══════════════════════════════════════════════════════════════════════════
C = {
    "base":     "#0B0E14",
    "surface":  "rgba(22, 27, 38, 0.72)",
    "surface2": "#141925",
    "border":   "rgba(120, 140, 175, 0.16)",
    "border2":  "rgba(120, 140, 175, 0.24)",
    "text":     "#E7ECF3",
    "muted":    "#8993A8",
    "accent":   "#4C8DFF",   # single primary accent — steel blue
    "accent2":  "#6E7FE0",   # secondary tone, close family (indigo)
    "gold":     "#C9A24B",   # tertiary series accent — muted gold, not neon
    "green":    "#3FBF8F",
    "red":      "#E0596B",
    "amber":    "#D9A441",
}

MODEL_COLORS = {
    "GRU":         C["accent"],
    "LSTM":        C["green"],
    "Transformer": C["accent2"],
    "ANN":         C["gold"],
}

# ══════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — clean, low-glow, professional fintech aesthetic
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {C['text']};
}}

.stApp {{
    background: linear-gradient(160deg, #0D1119 0%, {C['base']} 60%, #090B10 100%);
    background-attachment: fixed;
}}

[data-testid="stSidebar"] {{
    background: {C['surface2']};
    border-right: 1px solid {C['border']};
}}
[data-testid="stSidebar"] * {{ color: {C['text']}; }}

[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    background: transparent;
    border-bottom: 1px solid {C['border']};
    gap: 4px;
}}
[data-testid="stTabs"] [data-baseweb="tab"] {{
    background: transparent;
    padding: 10px 18px;
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: {C['muted']};
    border-bottom: 2px solid transparent;
    border-radius: 4px 4px 0 0;
    transition: color 0.15s ease;
}}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {{ color: {C['text']}; }}
[data-testid="stTabs"] [aria-selected="true"] {{
    color: {C['text']} !important;
    border-bottom: 2px solid {C['accent']};
}}

[data-testid="stMetric"] {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 8px;
    padding: 16px 18px;
}}
[data-testid="stMetric"] label {{
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: {C['muted']} !important;
}}
[data-testid="stMetricValue"] {{
    font-size: 1.45rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: {C['text']};
}}

[data-testid="stDataFrame"] {{
    border: 1px solid {C['border']};
    border-radius: 8px;
    overflow: hidden;
}}

hr {{ border: none; border-top: 1px solid {C['border']}; margin: 1.2rem 0; }}

h1 {{ font-weight: 700; color: {C['text']}; }}
h2 {{ font-size: 1.1rem; font-weight: 700; color: {C['text']}; letter-spacing: -0.005em; }}
h3 {{ font-size: 0.82rem; font-weight: 700; color: {C['muted']}; letter-spacing: 0.07em; text-transform: uppercase; }}

.hero {{
    padding: 22px 26px;
    border-radius: 10px;
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-left: 3px solid {C['accent']};
    margin-bottom: 1.1rem;
}}
.hero-title {{
    font-size: 1.55rem;
    font-weight: 700;
    color: {C['text']};
    letter-spacing: -0.01em;
    margin: 0;
}}
.hero-sub {{
    font-size: 0.84rem;
    color: {C['muted']};
    margin-top: 5px;
    max-width: 780px;
    line-height: 1.55;
}}

.kpi-strip {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 10px;
    padding: 16px 22px;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 36px;
    flex-wrap: wrap;
}}
.kpi-strip .kpi-label {{ font-size: 0.63rem; font-weight: 600; letter-spacing: 0.09em; text-transform: uppercase; color: {C['muted']}; margin-bottom: 3px; }}
.kpi-strip .kpi-value {{ font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; }}
.kpi-positive {{ color: {C['green']}; }}
.kpi-negative {{ color: {C['red']}; }}
.kpi-neutral  {{ color: {C['accent']}; }}
.kpi-sep {{ width: 1px; height: 34px; background: {C['border']}; flex-shrink: 0; }}

.glass-card {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-left: 3px solid {C['accent']};
    border-radius: 6px;
    padding: 16px 20px;
    margin-bottom: 12px;
}}
.glass-card h4 {{ font-size: 0.80rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: {C['accent']}; margin: 0 0 8px 0; }}
.glass-card p {{ font-size: 0.85rem; color: {C['muted']}; margin: 0; line-height: 1.6; }}

.plain-box {{
    background: rgba(76,141,255,0.06);
    border: 1px solid rgba(76,141,255,0.20);
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 0.85rem;
    color: {C['text']};
    line-height: 1.65;
    margin-bottom: 14px;
}}
.plain-box b {{ color: {C['accent']}; }}

.disclaimer {{
    background: rgba(224,89,107,0.06);
    border: 1px solid rgba(224,89,107,0.28);
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 0.78rem;
    color: {C['muted']};
    line-height: 1.6;
    margin-top: 1rem;
}}
.disclaimer strong {{ color: {C['red']}; }}

.empty-state {{
    background: {C['surface']};
    border: 1px dashed {C['border2']};
    border-radius: 8px;
    padding: 40px 24px;
    text-align: center;
    color: {C['muted']};
    font-size: 0.85rem;
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# CHART THEME
# ══════════════════════════════════════════════════════════════════════════
def chart_layout(**kwargs):
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=C["muted"], size=11),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor=C["border"],
                    tickfont=dict(size=10, color=C["muted"]),
                    title_font=dict(size=10, color=C["muted"]), zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor=C["border"],
                    tickfont=dict(size=10, color=C["muted"]),
                    title_font=dict(size=10, color=C["muted"]), zeroline=False),
        legend=dict(bgcolor="rgba(20,25,37,0.85)", bordercolor=C["border"], borderwidth=1,
                    font=dict(size=10, color=C["muted"])),
        margin=dict(l=48, r=20, t=32, b=48),
        hoverlabel=dict(bgcolor=C["surface2"], bordercolor=C["border"],
                          font=dict(family="JetBrains Mono, monospace", size=11, color=C["text"])),
    )
    base.update(kwargs)
    return base


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING (robust)
# ══════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    results   = pd.read_csv("results/final_all_models_all_stocks.csv")
    forecasts = pd.read_csv("results/future_10day_forecasts.csv", index_col=0)
    walk_fwd  = pd.read_csv("results/walk_forward_validation_AAPL.csv")
    tuning    = pd.read_csv("results/tuning_results_AAPL.csv")
    return results, forecasts, walk_fwd, tuning

try:
    results_df, forecast_df, walk_df, tuning_df = load_data()
except FileNotFoundError as e:
    st.markdown(
        f"<div class='empty-state'><b style='color:{C['text']};font-size:1rem;'>"
        f"Required data file missing</b><br><br>{e}<br><br>"
        "Make sure the <code>results/</code> folder is present alongside this app "
        "and contains all four CSV outputs.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

tickers = sorted(results_df["Ticker"].unique())
models  = sorted(results_df["Model"].unique())


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        f"<div style='font-size:0.68rem;font-weight:700;letter-spacing:0.12em;"
        f"text-transform:uppercase;color:{C['accent']};margin-bottom:2px;'>NeuroTrend</div>"
        f"<div style='font-size:0.72rem;color:{C['muted']};margin-bottom:20px;'>AI Stock Forecasting Dashboard</div>",
        unsafe_allow_html=True,
    )

    st.markdown(f"<div style='font-size:0.62rem;letter-spacing:0.08em;text-transform:uppercase;color:{C['muted']};font-weight:600;margin-bottom:4px;'>Asset</div>", unsafe_allow_html=True)
    selected_ticker = st.selectbox("Asset", tickers, index=0, label_visibility="collapsed")

    st.markdown(f"<div style='font-size:0.62rem;letter-spacing:0.08em;text-transform:uppercase;color:{C['muted']};font-weight:600;margin-top:12px;margin-bottom:4px;'>Model Architecture</div>", unsafe_allow_html=True)
    default_idx = models.index("GRU") if "GRU" in models else 0
    selected_model = st.selectbox("Model", models, index=default_idx, label_visibility="collapsed")

    st.markdown(f"<hr style='margin:18px 0;'>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.62rem;font-weight:700;letter-spacing:0.09em;text-transform:uppercase;color:{C['muted']};margin-bottom:10px;'>Dataset Reference</div>", unsafe_allow_html=True)

    meta_rows = [
        ("Tickers", ", ".join(tickers)),
        ("Period", "Approx. 10 years, daily OHLCV"),
        ("Source", "Yahoo Finance (yfinance)"),
        ("Architectures", ", ".join(models)),
        ("Forecast Horizon", "10 trading days"),
        ("Validation", "5-fold walk-forward"),
    ]
    for label, val in meta_rows:
        st.markdown(
            f"<div style='margin-bottom:8px;'><span style='font-size:0.60rem;color:{C['muted']};text-transform:uppercase;"
            f"letter-spacing:0.07em;font-weight:600;'>{label}</span><br>"
            f"<span style='font-size:0.75rem;color:{C['text']};font-family:\"JetBrains Mono\",monospace;'>{val}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown(f"<hr style='margin:18px 0;'>", unsafe_allow_html=True)
    with st.expander("Glossary — what these metrics mean"):
        st.markdown(
            "- **R² (accuracy score)** — how much of the price movement the model explains. Closer to 1.0 is better.\n"
            "- **MAPE** — average forecast error, shown as a percentage. Lower is better.\n"
            "- **RMSE / MAE / MSE** — error size in the model's internal (scaled) units. Lower is better; mainly used to compare models against each other.\n"
            "- **Walk-forward validation** — the model is retrained on rolling time windows to check it stays accurate over time, not just on one test split."
        )


# ══════════════════════════════════════════════════════════════════════════
# DERIVED VALUES
# ══════════════════════════════════════════════════════════════════════════
match = results_df[(results_df["Ticker"] == selected_ticker) & (results_df["Model"] == selected_model)]
if match.empty:
    st.error(f"No results found for {selected_ticker} / {selected_model}.")
    st.stop()
sel_row = match.iloc[0]

fc_available = selected_ticker in forecast_df.columns
if fc_available:
    fc_series = forecast_df[selected_ticker].astype(float)
    fc_day1, fc_day10 = fc_series.iloc[0], fc_series.iloc[-1]
    fc_delta = ((fc_day10 - fc_day1) / fc_day1) * 100 if fc_day1 != 0 else 0.0
    trend_cls = "kpi-positive" if fc_delta >= 0 else "kpi-negative"
    fc_delta_str = f"{'+' if fc_delta >= 0 else ''}{fc_delta:.2f}%"
    fc_arrow = "Up" if fc_delta >= 0 else "Down"
else:
    fc_delta_str, trend_cls, fc_arrow = "N/A", "kpi-neutral", ""

r2_cls   = "kpi-positive" if sel_row["R2"] >= 0.85 else ("kpi-negative" if sel_row["R2"] < 0.5 else "kpi-neutral")
mape_cls = "kpi-positive" if sel_row["MAPE"] < 3.0 else ("kpi-negative" if sel_row["MAPE"] > 7.0 else "kpi-neutral")


# ══════════════════════════════════════════════════════════════════════════
# HERO + KPI STRIP
# ══════════════════════════════════════════════════════════════════════════
st.markdown(
    f"""<div class="hero">
    <div class="hero-title">NeuroTrend — AI Stock Forecasting Dashboard</div>
    <div class="hero-sub">Deep learning price projections and model diagnostics across five equities,
    benchmarked with ANN, LSTM, GRU, and Transformer architectures on 10 years of Yahoo Finance data.</div>
    </div>""",
    unsafe_allow_html=True,
)

st.markdown(f"""
<div class="kpi-strip">
  <div><div class="kpi-label">Asset</div><div class="kpi-value kpi-neutral">{selected_ticker}</div></div>
  <div class="kpi-sep"></div>
  <div><div class="kpi-label">Model</div><div class="kpi-value kpi-neutral">{selected_model}</div></div>
  <div class="kpi-sep"></div>
  <div><div class="kpi-label">10-Day Forecast Drift ({fc_arrow})</div><div class="kpi-value {trend_cls}">{fc_delta_str}</div></div>
  <div class="kpi-sep"></div>
  <div><div class="kpi-label">R&sup2; Accuracy</div><div class="kpi-value {r2_cls}">{sel_row['R2']:.4f}</div></div>
  <div class="kpi-sep"></div>
  <div><div class="kpi-label">MAPE</div><div class="kpi-value {mape_cls}">{sel_row['MAPE']:.2f}%</div></div>
  <div class="kpi-sep"></div>
  <div><div class="kpi-label">RMSE</div><div class="kpi-value kpi-neutral">{sel_row['RMSE']:.5f}</div></div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════
tab_overview, tab_forecast, tab_perf, tab_validation, tab_method = st.tabs([
    "Overview",
    "10-Day Forecast",
    "Model Performance",
    "Validation & Tuning",
    "Methodology",
])


# ─────────────────────────────────────────────────────────────────────────
# TAB 0 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────
with tab_overview:
    st.markdown(
        f"<div class='plain-box'>"
        f"<b>What this dashboard shows:</b> four different deep learning models were trained to predict "
        f"future closing prices for five well-known U.S. stocks using ten years of historical trading data. "
        f"Use the sidebar to pick a stock and a model, then compare their accuracy, stability over time, "
        f"and 10-day price projection. The <b>Model Performance</b> and <b>Validation &amp; Tuning</b> tabs "
        f"contain the technical detail behind each result.</div>",
        unsafe_allow_html=True,
    )

    best_per_ticker = results_df.loc[results_df.groupby("Ticker")["R2"].idxmax()]
    c1, c2, c3 = st.columns(3)
    c1.metric("Assets Covered", len(tickers))
    c2.metric("Architectures Compared", len(models))
    c3.metric(
        "Best Overall R²",
        f"{results_df['R2'].max():.4f}",
        help="The single highest accuracy score achieved across every stock/model combination tested."
    )

    st.markdown(f"<h3 style='margin-top:1.4rem;margin-bottom:10px;'>Best Model Per Asset</h3>", unsafe_allow_html=True)
    bp = best_per_ticker[["Ticker", "Model", "R2", "MAPE", "RMSE"]].sort_values("R2", ascending=False).copy()
    bp["R2"] = bp["R2"].map(lambda x: f"{x:.4f}")
    bp["MAPE"] = bp["MAPE"].map(lambda x: f"{x:.2f}%")
    bp["RMSE"] = bp["RMSE"].map(lambda x: f"{x:.5f}")
    st.dataframe(bp, use_container_width=True, hide_index=True)

    avg_by_model = results_df.groupby("Model")["R2"].mean().sort_values(ascending=False)
    fig_avg = go.Figure(go.Bar(
        x=avg_by_model.index, y=avg_by_model.values,
        marker_color=[MODEL_COLORS.get(m, C["accent"]) for m in avg_by_model.index],
        marker_line=dict(color=C["border"], width=1),
        hovertemplate="<b>%{x}</b><br>Avg R²: %{y:.4f}<extra></extra>",
    ))
    fig_avg.update_layout(**chart_layout(height=300, xaxis_title="ARCHITECTURE", yaxis_title="AVERAGE R² (ALL ASSETS)", yaxis_tickformat=".3f"))
    st.markdown(f"<h3 style='margin-bottom:10px;'>Average Accuracy by Architecture</h3>", unsafe_allow_html=True)
    st.plotly_chart(fig_avg, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────
# TAB 1 — 10-DAY FORECAST
# ─────────────────────────────────────────────────────────────────────────
with tab_forecast:
    st.markdown(
        f"<h2 style='margin-bottom:4px;'>10-Day Forward Price Projection — {selected_ticker}</h2>"
        f"<p style='font-size:0.82rem;color:{C['muted']};margin-top:0;margin-bottom:1rem;'>"
        f"Forecast generated by the {selected_model} model trained on 10 years of daily OHLCV data. "
        "Values are shown in the model's normalised (0–1) scale; absolute price reconstruction requires "
        "the original scaler used at training time.</p>",
        unsafe_allow_html=True,
    )

    if fc_available:
        days_labels = [f"Day {i+1}" for i in range(len(fc_series))]
        fc_values = fc_series.values
        pct_chg = fc_series.pct_change().fillna(0).values * 100

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=days_labels, y=fc_values, mode="none", fill="tozeroy",
            fillcolor="rgba(76,141,255,0.08)", showlegend=False, hoverinfo="skip",
        ))
        fig_fc.add_trace(go.Scatter(
            x=days_labels, y=fc_values, mode="lines+markers",
            line=dict(color=C["accent"], width=2.5),
            marker=dict(size=7, color=C["accent"], line=dict(color=C["base"], width=2)),
            name="Projected Price",
            hovertemplate="<b>%{x}</b><br>Scaled Price: %{y:.6f}<extra></extra>",
        ))
        fig_fc.update_layout(**chart_layout(height=380, xaxis_title="TRADING DAY", yaxis_title="NORMALISED PRICE",
                                             yaxis_tickformat=".5f", showlegend=False))
        st.plotly_chart(fig_fc, use_container_width=True)

        st.markdown(f"<h3 style='margin-top:1.2rem;margin-bottom:10px;'>Daily Projection Grid</h3>", unsafe_allow_html=True)
        table_df = pd.DataFrame({
            "Trading Day": days_labels,
            "Scaled Forecast Price": [f"{v:.6f}" for v in fc_values],
            "Daily Change (%)": [f"{'+' if p >= 0 else ''}{p:.4f}%" for p in pct_chg],
            "Direction": ["Positive" if p > 0 else ("Negative" if p < 0 else "Flat") for p in pct_chg],
        })

        def color_direction(val):
            if val == "Positive": return f"color: {C['green']}; font-weight: 600;"
            if val == "Negative": return f"color: {C['red']}; font-weight: 600;"
            return f"color: {C['muted']};"

        styled_table = table_df.style.map(color_direction, subset=["Direction"]).set_properties(
            **{"font-family": "JetBrains Mono, monospace", "font-size": "0.80rem"})
        st.dataframe(styled_table, use_container_width=True, hide_index=True)
    else:
        st.markdown(
            f"<div class='empty-state'><b style='color:{C['text']};'>No forecast data available</b><br>"
            f"No 10-day projection was found for <b>{selected_ticker}</b> in "
            f"<code>results/future_10day_forecasts.csv</code>.</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────
# TAB 2 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────
with tab_perf:
    st.markdown(
        f"<h2 style='margin-bottom:4px;'>Performance Metrics — {selected_model} on {selected_ticker}</h2>"
        f"<p style='font-size:0.82rem;color:{C['muted']};margin-top:0;margin-bottom:1rem;'>"
        f"Evaluated on a held-out test split, after training for {int(sel_row['Epochs_Trained'])} epochs with early stopping.</p>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAE", f"{sel_row['MAE']:.5f}", help="Average absolute error between predicted and actual values (scaled units).")
    c2.metric("RMSE", f"{sel_row['RMSE']:.5f}", help="Root mean squared error; penalises large mistakes more heavily than MAE.")
    c3.metric("MAPE", f"{sel_row['MAPE']:.2f}%", help="Average percentage error — the easiest metric to interpret in plain terms.")
    c4.metric("R² Score", f"{sel_row['R2']:.4f}", help="Share of price variation explained by the model. 1.0 = perfect fit.")
    c5.metric("MSE", f"{sel_row['MSE']:.6f}", help="Mean squared error, used internally to compare models during training.")

    st.markdown("<hr>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(f"<h3 style='margin-bottom:10px;'>R&sup2; by Architecture — {selected_ticker}</h3>", unsafe_allow_html=True)
        sub = results_df[results_df["Ticker"] == selected_ticker]
        fig_bar = go.Figure()
        for _, r in sub.iterrows():
            fig_bar.add_trace(go.Bar(
                x=[r["Model"]], y=[r["R2"]],
                marker_color=MODEL_COLORS.get(r["Model"], C["accent"]),
                marker_line=dict(color=C["border"], width=1), name=r["Model"], showlegend=False,
                hovertemplate=f"<b>{r['Model']}</b><br>R²: {r['R2']:.4f}<br>MAPE: {r['MAPE']:.2f}%<extra></extra>",
            ))
        fig_bar.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.08)")
        fig_bar.update_layout(**chart_layout(height=340, xaxis_title="ARCHITECTURE", yaxis_title="R² SCORE", yaxis_tickformat=".3f"))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        st.markdown(f"<h3 style='margin-bottom:10px;'>R&sup2; Heatmap — All Assets × Architectures</h3>", unsafe_allow_html=True)
        ordered_models = [m for m in ["GRU", "LSTM", "Transformer", "ANN"] if m in results_df["Model"].unique()]
        pivot = results_df.pivot(index="Ticker", columns="Model", values="R2")[ordered_models]
        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            colorscale=[[0.0, C["red"]], [0.5, C["amber"]], [1.0, C["green"]]],
            zmin=pivot.values.min(), zmax=pivot.values.max(),
            text=np.round(pivot.values, 3), texttemplate="%{text}",
            textfont=dict(family="JetBrains Mono, monospace", size=11),
            colorbar=dict(title=dict(text="R²", font=dict(size=10, color=C["muted"])),
                           tickfont=dict(size=9, color=C["muted"]), thickness=12),
        ))
        fig_heat.update_layout(**chart_layout(height=340, xaxis_title="ARCHITECTURE", yaxis_title="TICKER"))
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='margin-bottom:10px;'>Comprehensive Performance Table</h3>", unsafe_allow_html=True)
    display_df = results_df.rename(columns={"Epochs_Trained": "Epochs"}).copy()
    display_df["MAE"] = display_df["MAE"].map(lambda x: f"{x:.5f}")
    display_df["MSE"] = display_df["MSE"].map(lambda x: f"{x:.6f}")
    display_df["RMSE"] = display_df["RMSE"].map(lambda x: f"{x:.5f}")
    display_df["MAPE"] = display_df["MAPE"].map(lambda x: f"{x:.2f}%")
    display_df["R2"] = display_df["R2"].map(lambda x: f"{x:.4f}")
    display_df["Epochs"] = display_df["Epochs"].astype(int)
    st.dataframe(display_df.sort_values(["Ticker", "Model"]), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────
# TAB 3 — WALK-FORWARD VALIDATION & TUNING
# ─────────────────────────────────────────────────────────────────────────
with tab_validation:
    st.markdown(
        f"<h2 style='margin-bottom:4px;'>Walk-Forward Validation — AAPL (GRU)</h2>"
        f"<p style='font-size:0.82rem;color:{C['muted']};margin-top:0;margin-bottom:1rem;'>"
        "Sequential 5-fold retraining. Each fold advances the training window forward in time, simulating "
        "live deployment where the model can never see the future.</p>",
        unsafe_allow_html=True,
    )

    col_wf1, col_wf2 = st.columns(2)
    with col_wf1:
        st.markdown(f"<h3 style='margin-bottom:10px;'>R&sup2; Stability Across Folds</h3>", unsafe_allow_html=True)
        r2_colors = [C["green"] if v >= 0.85 else (C["amber"] if v >= 0.5 else C["red"]) for v in walk_df["R2"]]
        fig_wf1 = go.Figure(go.Bar(
            x=walk_df["Fold"].astype(str), y=walk_df["R2"], marker_color=r2_colors,
            marker_line=dict(color=C["border"], width=1),
            hovertemplate="<b>Fold %{x}</b><br>R²: %{y:.4f}<extra></extra>",
        ))
        fig_wf1.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.08)")
        fig_wf1.update_layout(**chart_layout(height=320, xaxis_title="FOLD", yaxis_title="R² SCORE", yaxis_tickformat=".3f"))
        st.plotly_chart(fig_wf1, use_container_width=True)

    with col_wf2:
        st.markdown(f"<h3 style='margin-bottom:10px;'>MAPE Error Progression</h3>", unsafe_allow_html=True)
        fig_wf2 = go.Figure(go.Scatter(
            x=walk_df["Fold"].astype(str), y=walk_df["MAPE"], mode="lines+markers",
            line=dict(color=C["red"], width=2.5),
            marker=dict(size=7, color=C["red"], line=dict(color=C["base"], width=2)),
            hovertemplate="<b>Fold %{x}</b><br>MAPE: %{y:.4f}%<extra></extra>",
        ))
        fig_wf2.update_layout(**chart_layout(height=320, xaxis_title="FOLD", yaxis_title="MAPE (%)", yaxis_tickformat=".2f", yaxis_ticksuffix="%"))
        st.plotly_chart(fig_wf2, use_container_width=True)

    st.markdown(f"<h3 style='margin-bottom:10px;margin-top:1rem;'>Fold-by-Fold Metrics</h3>", unsafe_allow_html=True)
    wf_display = walk_df.copy()
    wf_display.columns = [c.upper() for c in wf_display.columns]
    wf_display["MAE"] = wf_display["MAE"].map(lambda x: f"{x:.5f}")
    wf_display["MSE"] = wf_display["MSE"].map(lambda x: f"{x:.6f}")
    wf_display["RMSE"] = wf_display["RMSE"].map(lambda x: f"{x:.5f}")
    wf_display["MAPE"] = wf_display["MAPE"].map(lambda x: f"{x:.2f}%")
    wf_display["R2"] = wf_display["R2"].map(lambda x: f"{x:.4f}")
    st.dataframe(wf_display, use_container_width=True, hide_index=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        f"<h2 style='margin-bottom:4px;'>Hyperparameter Sensitivity — GRU on AAPL</h2>"
        f"<p style='font-size:0.82rem;color:{C['muted']};margin-top:0;margin-bottom:1rem;'>"
        "Each hyperparameter is varied independently while the rest are held at baseline. "
        "'Final combined' uses the best value found for every parameter together.</p>",
        unsafe_allow_html=True,
    )

    hp_colors = {"sequence_length": C["accent"], "batch_size": C["green"], "dropout": C["accent2"],
                 "optimizer": C["gold"], "final_combined": C["muted"]}
    fig_hp = go.Figure()
    for hp_name, grp in tuning_df.groupby("Hyperparameter"):
        fig_hp.add_trace(go.Bar(
            x=grp["Value"], y=grp["R2"], name=hp_name.replace("_", " ").title(),
            marker_color=hp_colors.get(hp_name, C["muted"]), marker_line=dict(color=C["border"], width=1),
            hovertemplate=f"<b>{hp_name}</b><br>Value: %{{x}}<br>R²: %{{y:.4f}}<extra></extra>",
        ))
    fig_hp.update_layout(**chart_layout(height=400, xaxis_title="HYPERPARAMETER VALUE", yaxis_title="R² SCORE", barmode="group", yaxis_tickformat=".3f"))
    st.plotly_chart(fig_hp, use_container_width=True)

    st.markdown(f"<h3 style='margin-bottom:10px;'>Hyperparameter Configuration Table</h3>", unsafe_allow_html=True)
    tun_display = tuning_df.copy()
    tun_display.columns = [c.upper() for c in tun_display.columns]
    tun_display["MAE"] = tun_display["MAE"].map(lambda x: f"{x:.5f}")
    tun_display["MSE"] = tun_display["MSE"].map(lambda x: f"{x:.6f}")
    tun_display["RMSE"] = tun_display["RMSE"].map(lambda x: f"{x:.5f}")
    tun_display["MAPE"] = tun_display["MAPE"].map(lambda x: f"{x:.2f}%")
    tun_display["R2"] = tun_display["R2"].map(lambda x: f"{x:.4f}")
    st.dataframe(tun_display, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────
# TAB 4 — METHODOLOGY
# ─────────────────────────────────────────────────────────────────────────
with tab_method:
    st.markdown(
        f"<h2 style='margin-bottom:4px;'>Methodology & Disclosure</h2>"
        f"<p style='font-size:0.82rem;color:{C['muted']};margin-top:0;margin-bottom:1.2rem;'>"
        "Technical description of the data pipeline, model architectures, and evaluation protocol.</p>",
        unsafe_allow_html=True,
    )

    method_sections = [
        ("Data Inputs & Preprocessing",
         "Historical daily OHLCV data for five U.S. equities — AAPL, ABT, JPM, XOM, and WMT — sourced via the "
         "<code>yfinance</code> Python library, covering roughly 10 years (~2,500 trading days) per asset. "
         "Closing prices are normalised with MinMaxScaler (range 0–1) before a sliding window (default 60 days) "
         "converts the series into supervised input-output pairs."),
        ("Architecture Comparison",
         f"<b style='color:{C['gold']};'>ANN</b> — feedforward baseline that ignores sequence order.<br><br>"
         f"<b style='color:{C['green']};'>LSTM</b> — recurrent network with gated memory for long-range patterns.<br><br>"
         f"<b style='color:{C['accent']};'>GRU</b> — streamlined recurrent variant; achieved the strongest average R² overall.<br><br>"
         f"<b style='color:{C['accent2']};'>Transformer</b> — self-attention encoder; performance varies by asset "
         "since it needs more data than this dataset provides to reach its full potential."),
        ("Evaluation Protocol",
         "Each model is scored on a held-out test split (final ~20% of the series) using MAE, MSE, RMSE, MAPE, "
         "and R². A 5-fold walk-forward validation is additionally applied to the best-performing configuration "
         "(GRU on AAPL) to check accuracy stays consistent as time progresses."),
        ("Hyperparameter Tuning",
         "Sequence length, batch size, dropout rate, and optimiser (Adam vs. RMSProp) were each varied "
         "independently while the rest stayed at baseline. A final combined run uses the best value found per "
         "parameter. No automated search (grid/Bayesian) was used."),
        ("Known Limitations",
         "Forecasts are in normalised units; recovering absolute prices needs the original per-asset scaler. "
         "Models rely on price history only — no macro indicators, news sentiment, or earnings data are used. "
         "Accuracy drops for assets with structural breaks, and one AAPL validation fold shows negative R², "
         "which illustrates the risk of model drift without periodic retraining."),
    ]
    for title, body in method_sections:
        st.markdown(f"<div class='glass-card'><h4>{title}</h4><p>{body}</p></div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='disclaimer'><strong>Disclaimer:</strong> "
        "This dashboard is produced for educational and research purposes. All forecasts and metrics come from "
        "experimental deep learning models trained on historical price data. Past performance is not indicative "
        "of future results, and nothing here constitutes financial advice or a recommendation to buy or sell any "
        "security. Consult a qualified financial professional before making investment decisions.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(f"<h3 style='margin-top:1.4rem;margin-bottom:10px;'>Architecture Reference</h3>", unsafe_allow_html=True)
    arch_df = pd.DataFrame([
        {"Architecture": "ANN", "Type": "Feedforward", "Strengths": "Fast training, simple baseline", "Weaknesses": "Ignores temporal order"},
        {"Architecture": "LSTM", "Type": "Recurrent", "Strengths": "Long-range dependencies, gated memory", "Weaknesses": "Slower training, vanishing gradient risk"},
        {"Architecture": "GRU", "Type": "Recurrent (streamlined)", "Strengths": "Best overall R², fewer parameters", "Weaknesses": "Still sequential; no parallelism"},
        {"Architecture": "Transformer", "Type": "Attention-based", "Strengths": "Parallel processing, global context", "Weaknesses": "Data-hungry; underperforms on small datasets"},
    ])
    st.dataframe(arch_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    f"<div style='font-size:0.65rem;color:{C['muted']};display:flex;justify-content:space-between;"
    f"flex-wrap:wrap;gap:6px;padding-bottom:0.5rem;'>"
    f"<span>Built with Streamlit · TensorFlow/Keras · yFinance · Plotly</span>"
    f"<span>Assets: {' · '.join(tickers)} &nbsp;|&nbsp; 10-year daily OHLCV &nbsp;|&nbsp; "
    f"Architectures: {' · '.join(models)}</span></div>",
    unsafe_allow_html=True,
)
