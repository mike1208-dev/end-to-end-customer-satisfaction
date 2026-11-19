"""Data and model tests for the customer satisfaction project.

Run from the project root with:  python -m pytest tests -v
"""
import numpy as np
import pandas as pd
import pytest

from model.data_cleaning import DataCleaning, DataDivideStrategy, DataPreprocessStrategy
from model.evaluation import MSE, RMSE, Evaluation, R2Score
from model.model_dev import LightGBMModel
from steps.ingest_data import IngestData

# The inputs streamlit_app.py sends to the production model, in this order.
FEATURE_COLUMNS = [
    "payment_sequential",
    "payment_installments",
    "payment_value",
    "price",
    "freight_value",
    "product_name_lenght",
    "product_description_lenght",
    "product_photos_qty",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
]
DEPLOYMENT_MAX_RMSE = 1.35  # same default as continuous_deployment_pipeline


@pytest.fixture(scope="module")
def raw_data():
    return IngestData().get_data()


@pytest.fixture(scope="module")
def clean_data(raw_data):
    return DataCleaning(raw_data, DataPreprocessStrategy()).handle_data()


@pytest.fixture(scope="module")
def split_data(clean_data):
    return DataCleaning(clean_data, DataDivideStrategy()).handle_data()


# ---------- data cleaning ----------

def test_cleaned_data_has_no_missing_values(clean_data):
    assert clean_data.isna().sum().sum() == 0


def test_cleaned_data_is_all_numeric(clean_data):
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in clean_data.dtypes)


def test_target_is_a_1_to_5_review_score(clean_data):
    assert clean_data["review_score"].between(1, 5).all()


# ---------- train/test split ----------

def test_feature_columns_match_serving_inputs(split_data):
    x_train, x_test, _, _ = split_data
    assert list(x_train.columns) == FEATURE_COLUMNS
    assert list(x_test.columns) == FEATURE_COLUMNS


def test_split_shapes(split_data):
    x_train, x_test, y_train, y_test = split_data
    assert x_train.shape == (92487, 12)
    assert x_test.shape == (23122, 12)
    assert y_train.shape == (92487,)
    assert y_test.shape == (23122,)


def test_no_row_is_in_both_train_and_test(split_data):
    x_train, x_test, _, _ = split_data
    assert x_train.index.intersection(x_test.index).empty


@pytest.mark.xfail(
    strict=True,
    reason="Known issue: orders span several rows, and a random row split puts "
    "~21% of test orders in train too. Fix: split by order_id (GroupShuffleSplit).",
)
def test_no_order_is_in_both_train_and_test(raw_data, split_data):
    x_train, x_test, _, _ = split_data
    train_orders = set(raw_data.loc[x_train.index, "order_id"])
    test_orders = set(raw_data.loc[x_test.index, "order_id"])
    assert train_orders.isdisjoint(test_orders)


# ---------- evaluation metrics ----------

def test_metrics_on_hand_computed_values():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 5.0])  # one error of 2 -> squared errors 0, 0, 4
    assert MSE().calculate_score(y_true, y_pred) == pytest.approx(4 / 3)
    assert RMSE().calculate_score(y_true, y_pred) == pytest.approx(np.sqrt(4 / 3))
    assert R2Score().calculate_score(y_true, y_pred) == pytest.approx(-1.0)  # 1 - 4/2


def test_perfect_predictions_give_r2_of_one():
    y = np.array([1.0, 3.0, 5.0])
    assert R2Score().calculate_score(y, y) == pytest.approx(1.0)


def test_evaluation_base_class_is_abstract():
    with pytest.raises(TypeError):
        Evaluation()


# ---------- model quality ----------

def test_lightgbm_passes_deployment_gate(split_data):
    x_train, x_test, y_train, y_test = split_data
    model = LightGBMModel().train(x_train, y_train, verbose=-1)
    rmse = RMSE().calculate_score(y_test, model.predict(x_test))
    assert rmse < DEPLOYMENT_MAX_RMSE
