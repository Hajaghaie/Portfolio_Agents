"""
Visualization utilities using Plotly. it will generate a dashboard with the portfolio allocation, the individual asset metrics, and the portfolio metrics. 
and saved it as a html file in the output folder. 
"""
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from typing import Dict

def load_json_data(file_path: str) -> dict:
    """Loads data from a JSON file."""
    if not os.path.exists(file_path):
        logging.error(f"File not found at {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {file_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
        return None

def create_allocation_pie_chart(allocation_data: dict) -> go.Figure:
    """Creates a pie chart for portfolio allocation."""
    if not allocation_data or not isinstance(allocation_data, dict):
        return None
    labels = list(allocation_data.keys())
    values = list(allocation_data.values())
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, 
                                hole=.3, 
                                textinfo='percent+label',
                                title='Portfolio Allocation',
                                hoverinfo='label+percent+value')])
    fig.update_layout(title_text='Portfolio Allocation')
    return fig

def create_asset_metrics_bars(metrics_data: dict, allocation_data: dict) -> go.Figure:
    """Creates bar charts comparing individual asset metrics."""
    if not metrics_data or not isinstance(metrics_data, dict) or not allocation_data:
        return None
    assets = list(allocation_data.keys())
    asset_metrics = {asset: metrics_data.get(asset) for asset in assets if isinstance(metrics_data.get(asset), dict)}
    if not asset_metrics:
        logging.warning("No valid individual asset metrics found for assets in allocation.")
        return None
    plot_data = {
        'Annualized Return': [asset_metrics[asset].get('annualized_return') for asset in assets],
        'Volatility': [asset_metrics[asset].get('annualized_volatility') for asset in assets],
        'Sharpe Ratio': [asset_metrics[asset].get('sharpe_ratio') for asset in assets],
        'Beta': [asset_metrics[asset].get('beta') for asset in assets]
    }
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=('Annualized Return', 'Annualized Volatility', 
                                        'Sharpe Ratio', 'Beta'),
                        shared_xaxes=False, 
                        vertical_spacing=0.15, horizontal_spacing=0.1)
    fig.add_trace(go.Bar(x=assets, y=plot_data['Annualized Return'], name='Ann. Return', 
                       text=[f'{y:.2%}' if y is not None else 'N/A' for y in plot_data['Annualized Return']], 
                       textposition='auto'), row=1, col=1)
    fig.add_trace(go.Bar(x=assets, y=plot_data['Volatility'], name='Ann. Volatility', 
                       text=[f'{y:.2%}' if y is not None else 'N/A' for y in plot_data['Volatility']], 
                       textposition='auto'), row=1, col=2)
    fig.add_trace(go.Bar(x=assets, y=plot_data['Sharpe Ratio'], name='Sharpe Ratio', 
                       text=[f'{y:.2f}' if y is not None else 'N/A' for y in plot_data['Sharpe Ratio']], 
                       textposition='auto'), row=2, col=1)
    fig.add_trace(go.Bar(x=assets, y=plot_data['Beta'], name='Beta', 
                       text=[f'{y:.2f}' if y is not None else 'N/A' for y in plot_data['Beta']], 
                       textposition='auto'), row=2, col=2)
    fig.update_layout(title_text='Individual Asset Metrics Comparison', 
                      height=700, showlegend=True)
    fig.update_yaxes(tickformat=".1%", row=1, col=1)
    fig.update_yaxes(tickformat=".1%", row=1, col=2)
    fig.update_yaxes(row=2, col=1)
    fig.update_yaxes(row=2, col=2)
    if len(assets) > 5:
         fig.update_xaxes(tickangle=-45)
    return fig

def create_portfolio_metrics_bar(metrics_data: dict) -> go.Figure:
    """Creates a bar chart for portfolio-level metrics."""
    portfolio_metrics = metrics_data.get('portfolio') if isinstance(metrics_data, dict) else None
    if not portfolio_metrics or not isinstance(portfolio_metrics, dict) or 'error' in portfolio_metrics:
        logging.warning("Portfolio metrics missing or contain error. Cannot plot.")
        return None
    metrics_to_plot = {
        'Total Return': portfolio_metrics.get('total_return'),
        'Ann. Return': portfolio_metrics.get('annualized_return'),
        'Ann. Volatility': portfolio_metrics.get('annualized_volatility'),
        'Sharpe Ratio': portfolio_metrics.get('sharpe_ratio'),
        'Max Drawdown': portfolio_metrics.get('max_drawdown'),
        'Exp. Return (CAPM)': portfolio_metrics.get('expected_return_capm')
    }
    labels = [k for k, v in metrics_to_plot.items() if v is not None]
    values = [v for v in metrics_to_plot.values() if v is not None]
    if not labels:
        logging.warning("No valid portfolio metrics found to plot.")
        return None
    fig = go.Figure(data=[go.Bar(x=labels, y=values, 
                                text=[f'{v:.2%}' if "Return" in l or "Volatility" in l or "Drawdown" in l else f'{v:.2f}' for l, v in zip(labels, values)],
                                textposition='auto',
                                marker_color='skyblue')])
    fig.update_layout(title_text='Overall Portfolio Performance Metrics')
    fig.update_yaxes(title_text="Value")
    return fig

def main(output_dir=None):
    """Main function to load data, generate plots, and display/save. Accepts output_dir for all file paths."""
    logging.basicConfig(level=logging.INFO)
    if output_dir is None:
        output_dir = "output"
    allocation_file = os.path.join(output_dir, "portfolio_allocation.json")
    metrics_file = os.path.join(output_dir, "portfolio_metrics.json")
    html_output_file = os.path.join(output_dir, "portfolio_visualization.html")
    logging.info(f"Loading allocation data from {allocation_file}...")
    allocation_data = load_json_data(allocation_file)
    logging.info(f"Loading metrics data from {metrics_file}...")
    metrics_data = load_json_data(metrics_file)
    if not allocation_data:
        logging.error("Cannot proceed without allocation data.")
        return
    pie_fig = create_allocation_pie_chart(allocation_data)
    asset_bars_fig = create_asset_metrics_bars(metrics_data, allocation_data)
    portfolio_bar_fig = create_portfolio_metrics_bar(metrics_data)
    num_plots = sum(1 for fig in [pie_fig, asset_bars_fig, portfolio_bar_fig] if fig is not None)
    if num_plots == 0:
        logging.error("No valid plots could be generated.")
        return
    rows = 2
    cols = 2
    dashboard_fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{"type": "domain"}, {"type": "xy"}],
               [{"type": "xy", "colspan": 2}, None]],
        subplot_titles=('Portfolio Allocation', 'Overall Portfolio Metrics', 'Individual Asset Comparison')
    )
    plot_row = 1
    plot_col = 1
    if pie_fig:
        dashboard_fig.add_trace(pie_fig.data[0], row=plot_row, col=plot_col)
        plot_col += 1
    if portfolio_bar_fig:
        for trace in portfolio_bar_fig.data:
             dashboard_fig.add_trace(trace, row=plot_row, col=plot_col)
    plot_row += 1
    plot_col = 1
    if asset_bars_fig:
        for trace in asset_bars_fig.data:
            new_row = 2
            new_col = 1
            dashboard_fig.add_trace(trace, row=new_row, col=new_col)
    dashboard_fig.update_layout(
        title_text="Portfolio Analysis Dashboard",
        height=900,
        showlegend=True,
    )
    logging.info(f"Saving visualization to {html_output_file}...")
    dashboard_fig.write_html(html_output_file)
    logging.info("Visualization saved successfully.")
    print(f"Visualization saved to {html_output_file}")

if __name__ == "__main__":
    main() 