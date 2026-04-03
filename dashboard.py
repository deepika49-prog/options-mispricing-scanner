# dashboard.py

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, dash_table, Input, Output, State
import dash_table
from loguru import logger

from data import get_options_data, get_risk_free_rate
from pricing import price_options
from signals import generate_signals, get_vol_surface_data


# Color scheme

COLORS = {
    "bg": "#0f1117",
    "card": "#1a1d27",
    "border": "#2a2d3a",
    "text": "#e2e8f0",
    "muted": "#94a3b8",
    "accent": "#6366f1",       # indigo
    "overpriced": "#f87171",   # red — market overpriced = potential sell
    "underpriced": "#34d399",  # green — market underpriced = potential buy
    "neutral": "#60a5fa",      # blue
}


# Chart builders

def build_vol_surface_chart(surface_df: pd.DataFrame, ticker: str):
    """
    Builds a 3D surface plot of implied volatility.
    X axis: strike price
    Y axis: time to expiry (in months, for readability)
    Z axis: implied volatility (as %)
    """
    if surface_df.empty:
        return go.Figure().add_annotation(text="No vol surface data", showarrow=False)

    # Pivot to a grid for the surface plot
    
    df = surface_df.copy()
    df["expiry_months"] = (df["time_to_expiry"] * 12).round(1)
    df["iv_pct"] = df["smoothed_iv"] * 100  # convert to percentage

    # Create pivot table (average IV at each strike/expiry combination)
    pivot = df.pivot_table(
        values="iv_pct",
        index="expiry_months",
        columns="strike",
        aggfunc="mean"
    ).sort_index()

    fig = go.Figure(data=[go.Surface(
        z=pivot.values,
        x=pivot.columns.tolist(),   # strikes
        y=pivot.index.tolist(),     # expiry in months
        colorscale="Viridis",
        colorbar=dict(
            title=dict(text="IV (%)", font=dict(color=COLORS["text"], size=12)),
            tickfont=dict(color=COLORS["text"], size=11),
        ),
        hovertemplate=(
            "Strike: $%{x}<br>"
            "Expiry: %{y} months<br>"
            "IV: %{z:.1f}%<extra></extra>"
        )
    )])

    fig.update_layout(
        title=dict(text=f"{ticker} — Implied Volatility Surface",
                   font=dict(color=COLORS["text"], size=16)),
        scene=dict(
            xaxis=dict(title="Strike ($)", color=COLORS["muted"],
                       gridcolor=COLORS["border"], backgroundcolor=COLORS["card"]),
            yaxis=dict(title="Expiry (months)", color=COLORS["muted"],
                       gridcolor=COLORS["border"], backgroundcolor=COLORS["card"]),
            zaxis=dict(title="IV (%)", color=COLORS["muted"],
                       gridcolor=COLORS["border"], backgroundcolor=COLORS["card"]),
            bgcolor=COLORS["card"],
        ),
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"]),
        margin=dict(l=0, r=0, t=40, b=0),
        height=480,
    )
    return fig


def build_edge_distribution_chart(signals_df: pd.DataFrame, ticker: str):
    """
    A horizontal bar chart showing the edge % for each top signal.
    Color-coded: red bars = overpriced, green bars = underpriced.
    """
    if signals_df.empty:
        return go.Figure().add_annotation(text="No signals to display", showarrow=False)

    df = signals_df.head(15).copy()
    df["label"] = df.apply(
        lambda r: f"{r['type'].upper()} ${r['strike']:.0f} exp {r['expiry']}", axis=1
    )
    df["color"] = df["direction"].map({
        "overpriced": COLORS["overpriced"],
        "underpriced": COLORS["underpriced"]
    })

    fig = go.Figure(go.Bar(
        x=df["edge_pct"],
        y=df["label"],
        orientation="h",
        marker_color=df["color"].tolist(),
        hovertemplate="Edge: %{x:.2f}%<extra></extra>",
    ))

    fig.add_vline(x=0, line_color=COLORS["muted"], line_width=1)

    fig.update_layout(
        title=dict(text=f"{ticker} — Top Mispricing Signals",
                   font=dict(color=COLORS["text"], size=14)),
        xaxis=dict(title="Edge (%)", color=COLORS["muted"],
                   gridcolor=COLORS["border"], zerolinecolor=COLORS["muted"]),
        yaxis=dict(color=COLORS["text"], tickfont=dict(size=11)),
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"]),
        margin=dict(l=0, r=20, t=40, b=20),
        height=420,
    )
    return fig


# Dashboard layout and app

def create_app():
    app = Dash(__name__, title="Options Mispricing Scanner")

    # Styles 
    card_style = {
        "backgroundColor": COLORS["card"],
        "border": f"1px solid {COLORS['border']}",
        "borderRadius": "12px",
        "padding": "20px",
        "marginBottom": "20px",
    }

    app.layout = html.Div(style={"backgroundColor": COLORS["bg"], "minHeight": "100vh",
                                  "padding": "24px", "fontFamily": "Inter, sans-serif"}, children=[

        # Header
        html.Div(style={"marginBottom": "24px"}, children=[
            html.H1("Options Mispricing Scanner",
                    style={"color": COLORS["text"], "fontSize": "24px",
                           "fontWeight": "600", "margin": "0 0 4px 0"}),
            html.P("Compares Black-Scholes theoretical prices against live market prices to find mispriced contracts.",
                   style={"color": COLORS["muted"], "fontSize": "14px", "margin": 0}),
        ]),

        # Controls
        html.Div(style={**card_style, "display": "flex", "gap": "16px",
                        "alignItems": "flex-end", "flexWrap": "wrap"}, children=[
            html.Div(children=[
                html.Label("Ticker symbol", style={"color": COLORS["muted"],
                            "fontSize": "12px", "marginBottom": "6px", "display": "block"}),
                dcc.Input(id="ticker-input", type="text", value="AAPL", debounce=False,
                          placeholder="e.g. AAPL, SPY, TSLA",
                          style={"backgroundColor": COLORS["bg"], "color": COLORS["text"],
                                 "border": f"1px solid {COLORS['border']}", "borderRadius": "8px",
                                 "padding": "8px 12px", "fontSize": "14px", "width": "180px"}),
            ]),
            html.Div(children=[
                html.Label("Top N signals", style={"color": COLORS["muted"],
                           "fontSize": "12px", "marginBottom": "6px", "display": "block"}),
                dcc.Slider(id="top-n-slider", min=5, max=50, step=5, value=20,
                           marks={5: "5", 20: "20", 50: "50"},
                           tooltip={"placement": "bottom"},
                           className="",
                           ),
            ], style={"flex": "1", "minWidth": "200px"}),
            html.Button("Scan", id="scan-btn", n_clicks=0,
                        style={"backgroundColor": COLORS["accent"], "color": "white",
                               "border": "none", "borderRadius": "8px", "padding": "10px 28px",
                               "fontSize": "14px", "fontWeight": "600", "cursor": "pointer"}),
        ]),

        # Status bar
        html.Div(id="status-bar", style={"color": COLORS["muted"], "fontSize": "13px",
                                          "marginBottom": "16px", "minHeight": "20px"}),

        # Summary metric cards
        html.Div(id="metric-cards", style={"display": "grid",
                 "gridTemplateColumns": "repeat(auto-fit, minmax(160px, 1fr))",
                 "gap": "16px", "marginBottom": "20px"}),

        # Charts row
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                        "gap": "20px", "marginBottom": "20px"}, children=[
            html.Div(style=card_style, children=[dcc.Graph(id="vol-surface-chart")]),
            html.Div(style=card_style, children=[dcc.Graph(id="edge-chart")]),
        ]),

        # Signals table
        html.Div(style=card_style, children=[
            html.H3("Ranked signals", style={"color": COLORS["text"], "fontSize": "16px",
                                              "fontWeight": "500", "margin": "0 0 16px 0"}),
            html.Div(id="signals-table"),
        ]),

        # Hidden data store
        dcc.Store(id="signals-store"),
    ])

    # Callback
    @app.callback(
        Output("status-bar", "children"),
        Output("metric-cards", "children"),
        Output("vol-surface-chart", "figure"),
        Output("edge-chart", "figure"),
        Output("signals-table", "children"),
        Output("signals-store", "data"),
        Input("scan-btn", "n_clicks"),
        State("ticker-input", "value"),
        State("top-n-slider", "value"),
        prevent_initial_call=False,
    )
    def run_scan(n_clicks, ticker, top_n):
        ticker = (ticker or "AAPL").upper().strip()
        top_n = top_n or 20

        try:
            # 1. Fetch data
            rfr = get_risk_free_rate()
            raw_data = get_options_data(ticker)

            # 2. Price
            priced = price_options(raw_data, rfr)

            # 3. Generate signals
            signals = generate_signals(priced, top_n=top_n)
            surface_data = get_vol_surface_data(priced)

            status = (f"Scanned {ticker} — {len(priced)} contracts priced, "
                      f"{len(signals)} signals found. "
                      f"Risk-free rate: {rfr*100:.2f}%  |  "
                      f"Spot: ${raw_data['spot_price']:.2f}")

            # Metric cards
            if not signals.empty:
                overpriced_count = (signals["direction"] == "overpriced").sum()
                underpriced_count = (signals["direction"] == "underpriced").sum()
                max_edge = signals["edge_pct"].abs().max()
                avg_confidence = signals["confidence"].mean()
            else:
                overpriced_count = underpriced_count = max_edge = avg_confidence = 0

            def metric_card(label, value, color=COLORS["text"]):
                return html.Div(style={"backgroundColor": COLORS["bg"],
                                       "borderRadius": "8px", "padding": "14px"}, children=[
                    html.P(label, style={"color": COLORS["muted"], "fontSize": "12px",
                                         "margin": "0 0 4px 0"}),
                    html.P(str(value), style={"color": color, "fontSize": "22px",
                                              "fontWeight": "500", "margin": 0}),
                ])

            cards = [
                metric_card("Overpriced signals", overpriced_count, COLORS["overpriced"]),
                metric_card("Underpriced signals", underpriced_count, COLORS["underpriced"]),
                metric_card("Max edge", f"{max_edge:.1f}%", COLORS["accent"]),
                metric_card("Avg confidence", f"{avg_confidence:.0f}/100"),
            ]

            # Charts
            vol_fig = build_vol_surface_chart(surface_data, ticker)
            edge_fig = build_edge_distribution_chart(signals, ticker)

            # Table
            if signals.empty:
                table = html.P("No signals passed the liquidity filters.",
                               style={"color": COLORS["muted"]})
            else:
                display_cols = ["ticker", "type", "strike", "expiry", "mid",
                                "model_price", "edge", "edge_pct", "direction",
                                "delta", "vega", "open_interest", "confidence"]
                display_cols = [c for c in display_cols if c in signals.columns]
                table_df = signals[display_cols].reset_index(drop=True)
                table_df.index += 1

                table = dash_table.DataTable(
                    data=table_df.to_dict("records"),
                    columns=[{"name": c.replace("_", " ").title(), "id": c} for c in display_cols],
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": COLORS["bg"],
                                  "color": COLORS["muted"],
                                  "fontSize": "12px", "fontWeight": "500",
                                  "border": f"1px solid {COLORS['border']}",
                                  "padding": "8px 12px"},
                    style_cell={"backgroundColor": COLORS["card"],
                                "color": COLORS["text"],
                                "fontSize": "13px",
                                "border": f"1px solid {COLORS['border']}",
                                "padding": "8px 12px",
                                "fontFamily": "Inter, sans-serif"},
                    style_data_conditional=[
                        {
                            "if": {"filter_query": '{direction} = "overpriced"',
                                   "column_id": "direction"},
                            "color": COLORS["overpriced"], "fontWeight": "500",
                        },
                        {
                            "if": {"filter_query": '{direction} = "underpriced"',
                                   "column_id": "direction"},
                            "color": COLORS["underpriced"], "fontWeight": "500",
                        },
                        {
                            "if": {"filter_query": "{edge_pct} > 15"},
                            "backgroundColor": "rgba(99, 102, 241, 0.08)",
                        },
                    ],
                    sort_action="native",
                    filter_action="native",
                    page_size=20,
                )

            return status, cards, vol_fig, edge_fig, table, signals.to_json()

        except Exception as e:
            logger.exception(f"Scan failed for {ticker}: {e}")
            empty_fig = go.Figure()
            empty_fig.update_layout(
                paper_bgcolor=COLORS["card"],
                plot_bgcolor=COLORS["card"],
                font=dict(color=COLORS["text"])
            )
            error_msg = f"Error scanning {ticker}: {str(e)}"
            return error_msg, [], empty_fig, empty_fig, html.P(error_msg,
                   style={"color": COLORS["overpriced"]}), None

    return app


if __name__ == "__main__":
    app = create_app()
    logger.info("Starting Options Mispricing Scanner dashboard...")
    logger.info("Open your browser to: http://127.0.0.1:8050")
    app.run(debug=True, port=8050)
