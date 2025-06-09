# Agentic Portfolio Construction: A Multi-Agent Architecture for LLM-Driven Financial Asset Allocation

This repository contains the implementation of a LangGraph-based multi-agent framework for personalized investment portfolio construction. The system integrates large language models (LLMs), financial data APIs, and modular agent workflows to build, evaluate, and report tailored investment strategies. It use Tavily for sentiment and market news, and Yahoo! Finance to fetch the latest market data.

## 🔍 Overview

The framework was developed as part of a research project exploring the use of agentic architectures for financial decision-making. Key features include:

- Dynamic user profiling via natural language
- Real-time asset selection and analysis
- Modular agents for risk assessment, asset scoring, allocation, and reporting
- Support for quantitative metrics (Sharpe, CAPM, drawdown, momentum, etc.)

## 🧠 Architecture

The system is powered by [LangGraph](https://github.com/langchain-ai/langgraph), a framework for building stateful LLM workflows. Agents are defined as individual nodes with specialized roles:

- **Client Profiler**: Parses user input into structured investment goals
- **Asset Selector**: Retrieves assets using web APIs and LLM-based search
- **Metric Evaluator**: Computes financial indicators
- **Risk Assessor**: Validates risk alignment
- **Portfolio Allocator**: Assigns asset weights based on profile and metrics
- **Commentary Generator**: Explains portfolio reasoning in plain language

## 🧱 Components

- `portfolio_agents/agents.py`: Core agent workflow logic, node definitions, and LangGraph orchestration
- `portfolio_agents/data.py`: Financial data fetching and market news integration
- `portfolio_agents/metrics.py`: Financial metric calculations and portfolio validation
- `portfolio_agents/report.py`: Output formatting and report generation
- `portfolio_agents/visualization.py`: Portfolio and metric visualization utilities (Plotly)
- `portfolio_agents/config.py`: API keys and configuration settings
- `portfolio_agents/cli.py`: Command-line interface for running workflows and generating outputs
- `portfolio_agents/__main__.py`: Entry point for CLI execution
- `output/`: Generated reports, metrics, and visualizations for each run
- `requirements.txt`: Python dependencies

## 🚀 Getting Started

### Requirements

- Python 3.10+
- langgraph
- langchain
- langchain-openai
- langchain-community
- pydantic
- python-dotenv
- yfinance
- pandas
- numpy
- plotly
- typing-extensions
- [OpenAI API key](https://platform.openai.com/account/api-keys)
- [Tavily API key](https://app.tavily.com/)

### Installation

```bash
git clone https://github.com/Hajaghaie/Portfolio_Agents
cd Portfolio_Agents
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root with your API keys:

```
OPENAI_API_KEY="your-openai-key-here"
TAVILY_API_KEY="your-tavily-key-here"
```

### Usage

Run the CLI to start the portfolio construction workflow:

```bash
python -m portfolio_agents
```

You will be prompted to enter your investment amount, time horizon, risk tolerance, and any preferences. Outputs (report, metrics, visualizations) will be saved in a timestamped folder under `output/`.
