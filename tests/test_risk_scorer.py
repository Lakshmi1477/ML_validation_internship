from pathlib import Path

import pandas as pd

from configure import ground_truth, scenarios
from dt_ml.simulation_engine import run
from dt_ml.validation import validate_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]         #searches where the mentioned file is present


def load_dataset(file_path):
    """Load a CSV file into a pandas DataFrame."""

    return pd.read_csv(file_path)


def price_model_wrapper(test_df, parameter, magnitude, magnitude_type):
    """Wrap the simulation engine with the price-change scenario inputs."""
#learns the mapping of price in the market and profit or lose for the company.
    return run(
        df=test_df,
        decision_type="price_change",
        parameter=parameter,
        magnitude=magnitude,
        magnitude_type=magnitude_type,
    )


def run_validation():
    """Run validation using the shared scenario and ground-truth config."""
#here validation takes place by comparing the predicted results with the actual results and then calculating the metrics like mape, directional accuracy and rmse.
    test_df = load_dataset(PROJECT_ROOT / "data" / "sales_data.csv")
    return validate_model(
        test_df=test_df,
        scenarios=scenarios,
        ground_truth=ground_truth,
        model_predict_fn=price_model_wrapper,
    )


if __name__ == "__main__":  # check whether the used file is imported or not. It is a dunder keyword.
    results = run_validation()
##for debugging purpose we are checking whether the results contains the required metrics or not.
    assert "mape" in results
    assert "directional_accuracy" in results
    assert "rmse" in results

    print(results)


