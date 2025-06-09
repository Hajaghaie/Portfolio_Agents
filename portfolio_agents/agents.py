"""
Agent workflow logic using LangGraph and multi-agent orchestration. 
This is the main file that defines the workflow and the nodes. The agents here are the main agents that are used to generate the portfolio. 
For more information on the agents and the orchestration, please refer to the original paper. 
"""
import logging
import json
from typing import Dict, List, Optional, TypedDict
import pandas as pd
from pydantic.v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from .metrics import calculate_metrics_node as metrics_node, validate_portfolio_calculations, calculate_financial_metrics
from .report import structure_output_report
from .data import fetch_financial_data, fetch_market_news as fetch_news, fetch_data_node as data_node
from .config import BENCHMARK_TICKER, OPENAI_API_KEY, TAVILY_API_KEY

#State definition
class PortfolioGenerationState(TypedDict, total=False):
    initial_request: str
    user_profile: dict
    asset_universe: Optional[list[str]]
    market_news: Optional[str]
    financial_data: Optional[dict]
    metrics: Optional[dict]
    proposed_portfolio: Optional[dict]
    validation_result: Optional[dict]
    llm_commentary: Optional[str]
    final_report: Optional[str]
    error_message: Optional[str]
    step: Optional[str]

#LLM and tool initialization
llm = ChatOpenAI(model="gpt-4o", temperature=0.1, max_retries=2)
tavily_tool = TavilySearchResults(max_results=3) if TAVILY_API_KEY else None

#Node: Parse User Request
class UserProfileSchema(BaseModel):
    goal: str = Field(description="Primary investment goal (e.g., retirement, growth, income)")
    risk_tolerance: str = Field(description="User's risk tolerance (e.g., low, medium, high, conservative, aggressive)")
    time_horizon: str = Field(description="Investment time horizon (e.g., 5 years, 10-20 years, long-term)")
    initial_capital: Optional[float] = Field(description="Optional initial investment amount")
    preferences: Optional[Dict | List[str]] = Field(description="Optional user preferences like specific sectors (e.g. {'sector_focus': 'tech'}), ethical considerations, or assets to avoid/include.")
    specific_preferences: Optional[str] = Field(description="Specific preferences mentioned by the user as a single string.")
    suggested_assets: Optional[List[str]] = Field(description="A list of asset tickers. If the user specifies an exact number of tickers in their request (e.g., 'select 20 tickers'), this list MUST contain exactly that many tickers. This rule is paramount. Otherwise, if the user provides 5 or more assets without a specific count, use those, potentially adding more to reach around 15. If the user provides fewer than 5 and no specific count, generate around 20 diverse assets. The list MUST contain only valid tickers.")
    start_date: Optional[str] = Field(None, description="Optional start date for financial data in YYYY-MM-DD format. If provided along with end_date, this will be used instead of time_horizon.")
    end_date: Optional[str] = Field(None, description="Optional end date for financial data in YYYY-MM-DD format. If provided along with start_date, this will be used instead of time_horizon.")

def parse_user_request(state: PortfolioGenerationState) -> Dict:
    """Parses the initial user request using LLM to extract structured profile."""
    logging.info("Parsing user request with LLM...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert financial analyst assistant. Parse the user's request to understand their investment profile.\nExtract the goal, risk tolerance, time horizon, initial capital, and any specific preferences mentioned (store simple preferences as a string in 'specific_preferences').\nIdentify any specific assets the user suggested.\nAlso, extract `start_date` and `end_date` (in YYYY-MM-DD format) if the user specifies a precise date range for analysis. If a date range is given, it should take precedence over a general time horizon.\n\n**Asset Generation Rules (Strict Adherence Required):**\n1. **CRITICAL & ABSOLUTE REQUIREMENT: If the user explicitly requests a specific number of tickers (e.g., \"select 20 tickers\", \"give me 10 stocks\"), you MUST generate EXACTLY that number of diverse assets matching the profile.** This instruction overrides any other general guidelines on asset count, including any defaults suggested in field descriptions. The 'suggested_assets' field in your JSON output must reflect this exact count.\n2. If the user suggests 5 or more specific assets AND does not specify an exact number, use those primarily, potentially adding more diverse assets to reach a count of around 15.\n3. If the user suggests fewer than 5 assets AND does NOT specify an exact number, generate a diverse list of approximately 20 suitable assets (considering stocks, bonds, ETFs relevant to the profile).\n\nPopulate the 'suggested_assets' field with the final list of tickers. Ensure the list contains ONLY valid tickers.\nOutput ONLY the JSON object matching the required schema. **IMPORTANT: Do NOT include any comments (like //) inside the JSON output.** Today's Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}."""),
        ("human", "Here is the user request: {request}\n\nOutput ONLY the JSON object matching the required schema. Ensure NO comments are included in the JSON."),
    ])
    parser = JsonOutputParser(pydantic_object=UserProfileSchema)
    chain = prompt | llm | parser
    try:
        llm_output_raw = chain.invoke({"request": state['initial_request']})
        if isinstance(llm_output_raw, dict):
            user_profile = llm_output_raw
        elif isinstance(llm_output_raw, UserProfileSchema):
            user_profile = llm_output_raw.dict()
        else:
            if isinstance(llm_output_raw, str):
                try:
                    llm_output_raw_parsed = json.loads(llm_output_raw)
                    if isinstance(llm_output_raw_parsed, dict):
                        user_profile = llm_output_raw_parsed
                    else:
                        raise TypeError(f"Parsed string from LLM was not a dict: {type(llm_output_raw_parsed)}")
                except json.JSONDecodeError as json_e:
                    logging.error(f"Failed to parse LLM string output as JSON: {json_e}")
                    raise TypeError(f"LLM output was a string but not valid JSON. Raw: {llm_output_raw}") from json_e
            else:
                raise TypeError(f"Unexpected type returned from parser or direct LLM output: {type(llm_output_raw)}")
        assets = []
        if user_profile.get("suggested_assets"):
            assets.extend(user_profile["suggested_assets"])
        if assets:
            assets = sorted(list(set(ticker.upper().strip() for ticker in assets if isinstance(ticker, str))))
        if not assets:
            logging.error("No assets identified or generated by the initial parsing step.")
            return {"user_profile": user_profile, "asset_universe": [], "error_message": "No assets were identified or generated to proceed."}
        else:
            return {"user_profile": user_profile, "asset_universe": assets}
    except Exception as e:
        logging.error(f"Error parsing user request: {e}")
        return {"error_message": f"Failed to parse user request with LLM: {e}"}

#Node function implementations

def fetch_market_news(state):
    result = fetch_news(state)
    if not result:
        return {"market_news": None, "step": "fetch_market_news"}
    result["step"] = "fetch_market_news"
    return result

def fetch_data_node(state):
    result = data_node(state)
    if not result:
        return {"financial_data": None, "step": "fetch_data_node"}
    result["step"] = "fetch_data_node"
    return result

def calculate_metrics_node(state):
    result = metrics_node(state)
    if not result:
        return {"metrics": None, "step": "calculate_metrics_node"}
    result["step"] = "calculate_metrics_node"
    return result

# --- Propose Portfolio Node ---
class PortfolioAllocationSchema(BaseModel):
    portfolio_allocation: Dict[str, float] = Field(description="Dictionary mapping asset tickers to allocation weight (e.g., {'AAPL': 0.6, 'MSFT': 0.4}). Weights must sum to 1.0.")
    reasoning: Optional[str] = Field(description="Brief reasoning for the proposed allocation based on user profile and data.")

def propose_portfolio_node(state):
    user_profile = state.get('user_profile')
    metrics = state.get('metrics')
    news = state.get('market_news')
    asset_universe = state.get('asset_universe')
    if not user_profile or not metrics or not asset_universe:
        return {"error_message": "Missing required inputs (profile, metrics, or asset universe) to propose portfolio."}
    user_profile_json = json.dumps(user_profile)
    metrics_summary_json = json.dumps(metrics, indent=2)
    if len(metrics_summary_json) > 5000:
        metrics_summary_json = metrics_summary_json[:5000] + "\n... (truncated)"
    escaped_user_profile = user_profile_json.replace('{', '{{').replace('}', '}}')
    escaped_metrics_summary = metrics_summary_json.replace('{', '{{').replace('}', '}}')
    escaped_news = (news.replace('{', '{{').replace('}', '}}') if news else "N/A")
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert portfolio manager. Your task is to propose a portfolio allocation based on the user's profile, available asset data metrics (including historical performance, CAPM expected return, and SMA indicators), and recent market news.\nUser Profile: {escaped_user_profile}\nAvailable Assets with Data: {', '.join(asset_universe)}\nAsset Metrics Summary (Historical Return/Vol/Sharpe/Drawdown, Beta, CAPM Expected Return, SMA 50/200, Portfolio Momentum Outlook):\n{escaped_metrics_summary}\nRecent Market News Context:\n{escaped_news}\n\nConstraints:\n- Allocate ONLY among the 'Available Assets with Data'.\n- Proposed weights MUST sum to 1.0 (or very close to it).\n- Strive for a diverse range of allocation percentages, reflecting a detailed analysis. For instance, feel free to use precise values like 7.3%, 12.8%, 18.2%, etc., rather than rounding to simpler percentages, if the underlying data and user profile suggest such a nuanced distribution.\n- Consider the user's goal and risk tolerance foremost.\n- Also consider the CAPM expected return and the portfolio momentum outlook (SMA trend) when making allocations.\n- Provide brief reasoning.\n\nOutput ONLY the JSON object matching the required schema. Ensure ticker symbols in the output JSON match the available assets exactly."""),
        ("human", "Based on the provided information, propose a suitable portfolio allocation and provide reasoning, considering historical metrics, expected returns, and momentum.")
    ])
    parser = JsonOutputParser(pydantic_object=PortfolioAllocationSchema)
    chain = prompt_template | llm | parser
    try:
        proposal = chain.invoke({})
        proposed_portfolio = None
        llm_reasoning = None
        if isinstance(proposal, dict):
            if 'portfolio_allocation' in proposal:
                proposed_portfolio = proposal.get('portfolio_allocation')
                llm_reasoning = proposal.get('reasoning')
                if not isinstance(proposed_portfolio, dict):
                    raise ValueError("LLM output contained 'portfolio_allocation' key, but its value was not a dictionary.")
            elif all(isinstance(k, str) and isinstance(v, (float, int)) for k, v in proposal.items()):
                proposed_portfolio = proposal
            else:
                raise ValueError("LLM output is a dictionary but not in the expected portfolio structure (missing 'portfolio_allocation' key or invalid format).")
        else:
            raise TypeError(f"LLM output was not a dictionary, received type: {type(proposal)}")
        proposed_portfolio = {k.upper(): v for k,v in proposed_portfolio.items()}
        filtered_portfolio = { t:w for t,w in proposed_portfolio.items() if t in asset_universe }
        current_sum = sum(filtered_portfolio.values())
        if abs(current_sum) > 1e-6 and len(filtered_portfolio) > 0:
            renormalized_portfolio = {t: w / current_sum for t, w in filtered_portfolio.items()}
            if abs(current_sum - 1.0) > 0.05 :
                logging.info(f"Re-normalized portfolio weights from sum {current_sum:.3f} to 1.0")
            proposed_portfolio = renormalized_portfolio
        elif not filtered_portfolio:
            return {"error_message":"LLM proposed portfolio contained no valid/available assets."}
        else:
            proposed_portfolio = filtered_portfolio
        return {
            "proposed_portfolio": proposed_portfolio,
            "llm_commentary": llm_reasoning,
            "step": "propose_portfolio_node"
        }
    except Exception as e:
        logging.error(f"Error proposing portfolio: {e}")
        return {"error_message": f"Failed to propose portfolio with LLM: {e}", "step": "propose_portfolio_node"}

# --- Validate Portfolio Node ---
def validate_portfolio_node(state):
    portfolio = state.get('proposed_portfolio')
    metrics = state.get('metrics')
    financial_data = state.get('financial_data')
    recalculated_metrics = {}
    if portfolio and financial_data:
        portfolio_metrics_update = calculate_financial_metrics(data=financial_data, portfolio=portfolio)
        if 'portfolio' in portfolio_metrics_update:
            if metrics is None: metrics = {}
            metrics['portfolio'] = portfolio_metrics_update['portfolio']
            recalculated_metrics = {"metrics": metrics}
        elif 'error' in portfolio_metrics_update:
            if metrics is None: metrics = {}
            metrics['portfolio'] = {'error': portfolio_metrics_update['error']}
            recalculated_metrics = {"metrics": metrics}
    validation_result = validate_portfolio_calculations(portfolio=portfolio, metrics=metrics)
    final_update = recalculated_metrics
    final_update["validation_result"] = validation_result
    if not final_update:
        return {"validation_result": None, "step": "validate_portfolio_node"}
    final_update["step"] = "validate_portfolio_node"
    return final_update

# --- Generate Commentary Node ---
def generate_commentary_node(state):
    user_profile = state.get('user_profile')
    portfolio = state.get('proposed_portfolio')
    metrics = state.get('metrics')
    validation = state.get('validation_result')
    news = state.get('market_news')
    llm_reasoning = state.get('llm_commentary')

    user_profile_summary = json.dumps(user_profile if user_profile else {})
    metrics_summary = "Metrics calculation encountered an error or did not run."
    if isinstance(metrics, dict):
        metrics_summary = json.dumps(metrics, indent=2)
        if len(metrics_summary) > 4000:
            metrics_summary = metrics_summary[:4000] + "\n... (truncated)"
    validation_summary = "Validation did not run or failed."
    if isinstance(validation, dict):
        validation_summary = f"Status: {validation.get('status', 'N/A').upper()}"
        if validation.get('errors'):
            validation_summary += f", Issues: {'; '.join(validation['errors'])}"
    portfolio_summary = "No portfolio proposed or portfolio was invalid."
    if isinstance(portfolio, dict) and portfolio:
        portfolio_summary = json.dumps(portfolio)
    escaped_user_profile = user_profile_summary.replace('{', '{{').replace('}', '}}')
    escaped_metrics = metrics_summary.replace('{', '{{').replace('}', '}}')
    escaped_validation = validation_summary.replace('{', '{{').replace('}', '}}')
    escaped_portfolio = portfolio_summary.replace('{', '{{').replace('}', '}}')
    escaped_news = (news.replace('{', '{{').replace('}', '}}') if news else "N/A")
    escaped_llm_reasoning = (llm_reasoning.replace('{', '{{').replace('}', '}}') if llm_reasoning else "(No specific reasoning provided by the allocation model)")

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", f"""You are a financial advisor AI. Generate a clear and concise commentary explaining the proposed portfolio allocation, its key metrics (including historical performance, CAPM expected return, and momentum outlook), and validation results to the user. If the allocation model provided reasoning, incorporate it. If validation failed, explain the issues.\nContext:\n- User Profile: {escaped_user_profile}\n- Proposed Portfolio: {escaped_portfolio}\n- Portfolio & Asset Metrics (Includes Historical, CAPM Exp. Return, SMAs, Momentum): {escaped_metrics}\n- Validation Result: {escaped_validation}\n- Market News Context: {escaped_news}\n- Initial Allocation Reasoning (if provided): {escaped_llm_reasoning}\n\nInstructions:\n- Explain the reasoning behind the allocation in relation to the user's profile, historical performance, expected returns (CAPM), and the overall portfolio momentum outlook (SMA trend).\n- Briefly interpret the key portfolio metrics (Return, Volatility, Sharpe, Drawdown, CAPM Expected Return, Momentum Outlook).\n- Mention the validation outcome. If issues were found, briefly explain them clearly.\n- Keep the tone informative and objective.\n- **Include a disclaimer that this is not financial advice.**\n- Output only the commentary text.\n"""),
        ("human", "Please provide the commentary for the generated portfolio report based on the context, including interpretation of the new CAPM and momentum metrics.")
    ])
    parser = StrOutputParser()
    chain = prompt_template | llm | parser
    try:
        commentary = chain.invoke({})
        if "not financial advice" not in commentary.lower() and "disclaimer" not in commentary.lower():
            commentary += "\n\n**Disclaimer:** This is an AI-generated analysis and does not constitute financial advice. Consult a qualified professional before making investment decisions."
        return {"llm_commentary": commentary, "step": "generate_commentary_node"}
    except Exception as e:
        logging.error(f"Error generating commentary: {e}")
        return {"llm_commentary": f"Failed to generate commentary: {e}", "step": "generate_commentary_node"}

# --- Structure Output Node ---
def structure_output_node(state):
    out = {"final_report": structure_output_report(state), "step": "structure_output_node"}
    return out

# --- Handle Error Node ---
def handle_error_node(state):
    error = state.get("error_message", "An unspecified error occurred.")
    error_report = f"# Portfolio Generation Failed\n\nAn error occurred during the process:\n\n```\n{error}\n```\n\nPlease review the input or contact support."
    return {"final_report": error_report, "step": "handle_error_node"}

# --- Routing Functions for Conditional Edges ---
def should_proceed_after_parsing(state: PortfolioGenerationState) -> str:
    if state.get("error_message"):
        return "handle_error"
    if not state.get("asset_universe"):
        state["error_message"] = state.get("error_message", "No specific assets identified to proceed with analysis.")
        return "handle_error"
    return "fetch_market_news"

def should_proceed_after_data_fetch(state: PortfolioGenerationState) -> str:
    if state.get("error_message"):
        return "handle_error"
    if not state.get("financial_data"):
        state["error_message"] = state.get("error_message", "No financial data fetched.")
        return "handle_error"
    return "calculate_metrics"

def should_proceed_after_proposal(state: PortfolioGenerationState) -> str:
    if state.get("error_message"):
        return "handle_error"
    if not state.get("proposed_portfolio"):
        state["error_message"] = state.get("error_message", "No portfolio could be proposed.")
        return "handle_error"
    return "validate_portfolio"

def should_proceed_after_validation(state: PortfolioGenerationState) -> str:
    if state.get("error_message"):
        return "handle_error"
    validation = state.get("validation_result", {})
    if isinstance(validation, dict) and validation.get("status", "pass").lower() != "pass":
        state["error_message"] = f"Portfolio validation failed: {validation.get('errors', 'Unknown error')}"
        return "handle_error"
    return "generate_commentary"

# --- Graph wiring and workflow logic ---
def build_workflow():
    """Build and return the LangGraph workflow for the portfolio agent."""
    workflow = StateGraph(PortfolioGenerationState)
    # Add all nodes
    workflow.add_node("parse_user_request", parse_user_request)
    workflow.add_node("fetch_market_news", fetch_market_news)
    workflow.add_node("fetch_data", fetch_data_node)
    workflow.add_node("calculate_metrics", calculate_metrics_node)
    workflow.add_node("propose_portfolio", propose_portfolio_node)
    workflow.add_node("validate_portfolio", validate_portfolio_node)
    workflow.add_node("generate_commentary", generate_commentary_node)
    workflow.add_node("structure_output", structure_output_node)
    workflow.add_node("handle_error", handle_error_node)
    # Set entry point
    workflow.set_entry_point("parse_user_request")
    # Conditional edges for robust error handling and correct transitions
    workflow.add_conditional_edges(
        "parse_user_request",
        should_proceed_after_parsing,
        {
            "fetch_market_news": "fetch_market_news",
            "handle_error": "handle_error"
        }
    )
    workflow.add_edge("fetch_market_news", "fetch_data")
    workflow.add_conditional_edges(
        "fetch_data",
        should_proceed_after_data_fetch,
        {
            "calculate_metrics": "calculate_metrics",
            "handle_error": "handle_error"
        }
    )
    workflow.add_edge("calculate_metrics", "propose_portfolio")
    workflow.add_conditional_edges(
        "propose_portfolio",
        should_proceed_after_proposal,
        {
            "validate_portfolio": "validate_portfolio",
            "handle_error": "handle_error"
        }
    )
    workflow.add_conditional_edges(
        "validate_portfolio",
        should_proceed_after_validation,
        {
            "generate_commentary": "generate_commentary",
            "handle_error": "handle_error"
        }
    )
    workflow.add_edge("generate_commentary", "structure_output")
    workflow.add_edge("structure_output", END)
    workflow.add_edge("handle_error", END)
    return workflow.compile() 