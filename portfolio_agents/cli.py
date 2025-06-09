"""
Command-line interface for portfolio agents package. You can run the script directly from the command line and it will prompt you for the input. 
in the input you can specify the initial investment amount, the time horizon, the risk tolerance, and any specific preferences. 
it will then generate the portfolio and save the outputs in the output folder. 
"""
import logging
import argparse
import datetime
import os
import sys
import json
from .agents import build_workflow


def main():
    """Main CLI entry point for portfolio agent workflow."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Portfolio Agents CLI")
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode (default)')
    args = parser.parse_args()
    print("\nPlease provide your investment details:")
    capital_str = input("1. Initial investment amount (e.g., 10000): ")
    time_horizon_str = input("2. Investment time horizon (e.g., '5 years', '10-15 years', 'long-term'): ")
    risk_tolerance_str = input("3. Risk tolerance (e.g., 'low', 'medium', 'high', 'conservative', 'aggressive'): ")
    preferences_str = input("4. Any specific preferences? (e.g., 'focus on tech', 'avoid fossil fuels', 'include GOOGL', or leave blank): ")
    try:
        initial_capital = float(capital_str)
    except ValueError:
        logging.warning(f"Could not parse '{capital_str}' as an amount. Proceeding without initial capital.")
        initial_capital = None
    initial_user_request = f"I want to invest ${initial_capital if initial_capital else 'an amount'} for a time horizon of {time_horizon_str}. My risk tolerance is {risk_tolerance_str}."
    if preferences_str:
        initial_user_request += f" Specific preferences: {preferences_str}."
    else:
        initial_user_request += " No specific preferences mentioned."
    inputs = {"initial_request": initial_user_request}
    logging.info(f"Constructed Request: {initial_user_request}")
    workflow = build_workflow()
    final_state = None
    try:
        logging.info("Invoking graph...")
        final_state = workflow.invoke(inputs, {"recursion_limit": 20})
        logging.info("--- Graph Execution Finished ---")
    except Exception as e:
        logging.error(f"Graph Execution Failed: {e}")
    output_base_dir = "output"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(output_base_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    if final_state and isinstance(final_state, dict):
        report_content = final_state.get("final_report")
        error_message = final_state.get("error_message")
        proposed_portfolio = final_state.get('proposed_portfolio')
        metrics_data = final_state.get('metrics')
        if report_content:
            print("\n========= FINAL REPORT =========\n")
            print(report_content)
            print("\n==============================\n")
        elif error_message:
            error_report = f"# Portfolio Generation Failed\n\nAn error occurred:\n```\n{error_message}\n```"
            print("\n========= ERROR REPORT =========\n")
            print(error_report)
            print("\n==============================\n")
        else:
            print("\n--- No final report or error message found in state --- ")

        if report_content and not error_message:
            report_filename = os.path.join(output_dir, f"portfolio_report.md")
            try:
                with open(report_filename, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                print(f"\n--- Final report saved to {report_filename} --- ")
            except Exception as report_e:
                print(f"\n--- Error saving report to file: {report_e} ---")
        elif error_message:
            print(f"\n--- Skipping report file saving due to error: {error_message} --- ")
        else:
            print("\n--- Skipping report file saving as no report content was generated. ---")
        if proposed_portfolio and isinstance(proposed_portfolio, dict) and not error_message:
            json_filename = os.path.join(output_dir, "portfolio_allocation.json")
            try:
                with open(json_filename, 'w', encoding='utf-8') as f:
                    json.dump(proposed_portfolio, f, ensure_ascii=False, indent=4)
                print(f"--- Portfolio allocation saved to {json_filename} --- ")
            except Exception as json_e:
                print(f"--- Error saving portfolio to JSON: {json_e} ---")
        elif error_message:
            print(f"--- Skipping JSON save due to error: {error_message} --- ")
        else:
            print("--- Skipping JSON save: No valid proposed portfolio found or error occurred. ---")
        if metrics_data and isinstance(metrics_data, dict) and not error_message:
            metrics_filename = os.path.join(output_dir, "portfolio_metrics.json")
            def replace_nan_with_none(obj):
                if isinstance(obj, dict):
                    return {k: replace_nan_with_none(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [replace_nan_with_none(elem) for elem in obj]
                elif isinstance(obj, float) and hasattr(__import__('numpy'), 'isnan') and __import__('numpy').isnan(obj):
                    return None
                return obj
            try:
                cleaned_metrics_data = replace_nan_with_none(metrics_data)
                with open(metrics_filename, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_metrics_data, f, ensure_ascii=False, indent=4)
                print(f"--- Portfolio metrics saved to {metrics_filename} --- ")
            except Exception as metrics_e:
                print(f"--- Error saving metrics to JSON: {metrics_e} ---")
        elif error_message:
            print(f"--- Skipping Metrics JSON save due to error: {error_message} --- ")
        else:
            print("--- Skipping Metrics JSON save: No valid metrics data found or error occurred. ---")

        try:
            from portfolio_agents.visualization import main as run_visualization
            if proposed_portfolio and metrics_data:
                print("\nGenerating visualization dashboard...\n")
                run_visualization(output_dir=output_dir)
        except Exception as viz_e:
            print(f"Visualization step failed: {viz_e}")

        print(f"\nAll outputs for this run are saved in: {output_dir}\n")
    elif final_state:
        print("\n--- Could not process final state (unexpected format) --- ")
        print("Final State Value:", repr(final_state)[:500])
    else:
        print("\n--- No final state available due to execution error --- ")

if __name__ == "__main__":
    main() 