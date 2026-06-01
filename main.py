from pathlib import Path

import pandas as pd

from config import ground_truth, scenarios
from dt_ml.baseline_analytics import compute
from dt_ml.risk_scorer import score
from dt_ml.simulation_engine import run
from dt_ml.validation import validate_model
from dt_ml.validation.report_builder import build_validation_report
from memory_cleaner import clear_memory


PROJECT_ROOT = Path(__file__).resolve().parent      
def load_dataset(file_path):
    return pd.read_csv(file_path)


def price_model_wrapper(test_df, parameter, magnitude, magnitude_type):
    return run(
        df=test_df,
        decision_type="price_change",
        parameter=parameter,
        magnitude=magnitude,
        magnitude_type=magnitude_type,
    )


def main():
    data_path = PROJECT_ROOT / "data" / "sales_data.csv"
    df = load_dataset(data_path)

    print("Baseline Analytics:")
    baseline = compute(df)
    print(baseline)

    print("\nSimulation Result:")
    simulation_result = run(
        df=df,
        decision_type="price_change",
        parameter="price_per_unit",
        magnitude=10,
        magnitude_type="percentage",
    )
    print(simulation_result)

    print("\nValidation Results:")
    validation_results = validate_model(
        price_model_wrapper,
        test_df=df,
        scenarios=scenarios,
        ground_truth=ground_truth,
    )
    for metric, value in validation_results.items():
        print(f"{metric}: {value}")

    print("\nRisk Score:")
    risk = score(
        "price_change",
        10,
        simulation_result,
    )
    print(risk)

    report_path = PROJECT_ROOT / "validation_report.pdf"
    generated_report = build_validation_report(output_path=str(report_path))
    print(f"\nPDF report generated: {generated_report}")

    clear_memory()


if __name__ == "__main__":
    main()