from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


RAW_FEATURES = [
    "Category",
    "Sub-Category",
    "Quantity",
    "Amount",
    "PaymentMode",
    "State",
    "City",
]

LEAKAGE_COLUMNS = [
    "Order ID",
    "CustomerName",
    "Profit",
    "Profit_Margin",
    "Order Date",
    "Year-Month",
]

TARGET_COLUMN = "Risk_Label"
PROFIT_MARGIN_THRESHOLD = 0.25


def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw training data and create the binary target label."""
    cleaned = df.copy()

    for col in ["Amount", "Profit", "Quantity"]:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned = cleaned.dropna(subset=["Amount", "Profit", "Quantity"])
    cleaned["State"] = cleaned["State"].fillna("Unknown")
    cleaned["City"] = cleaned["City"].fillna("Unknown")
    cleaned["Category"] = cleaned["Category"].fillna("Unknown")
    cleaned["Sub-Category"] = cleaned["Sub-Category"].fillna("Unknown")
    cleaned["PaymentMode"] = cleaned["PaymentMode"].fillna("Unknown")
    cleaned = cleaned.drop_duplicates()

    cleaned["Profit_Margin"] = cleaned["Profit"] / cleaned["Amount"]
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna(subset=["Profit_Margin"])
    cleaned[TARGET_COLUMN] = (
        cleaned["Profit_Margin"] >= PROFIT_MARGIN_THRESHOLD
    ).astype(int)

    return cleaned


def _sorted_unique(values: Iterable[object]) -> List[str]:
    return sorted(pd.Series(values).dropna().astype(str).unique().tolist())


@dataclass
class OrderRiskPreprocessor:
    """Transform raw order rows into the 24 model-ready features."""

    onehot_columns: List[str] = field(
        default_factory=lambda: ["PaymentMode", "Category", "Sub-Category"]
    )
    label_columns: List[str] = field(default_factory=lambda: ["State", "City"])
    numeric_columns: List[str] = field(default_factory=lambda: ["Quantity", "Amount"])
    categories_: Dict[str, List[str]] = field(default_factory=dict)
    label_maps_: Dict[str, Dict[str, int]] = field(default_factory=dict)
    feature_columns_: List[str] = field(default_factory=list)
    numeric_medians_: Dict[str, float] = field(default_factory=dict)
    quantity_upper_bound_: float | None = None
    quantity_cap_: float | None = None
    scaler_: MinMaxScaler | None = None

    def fit(self, X: pd.DataFrame) -> "OrderRiskPreprocessor":
        for col in self.numeric_columns:
            numeric_values = pd.to_numeric(X[col], errors="coerce")
            median_value = numeric_values.median()
            self.numeric_medians_[col] = 0.0 if pd.isna(median_value) else float(median_value)

        prepared = self._basic_prepare(X)

        q1 = prepared["Quantity"].quantile(0.25)
        q3 = prepared["Quantity"].quantile(0.75)
        iqr = q3 - q1
        self.quantity_upper_bound_ = float(q3 + 1.5 * iqr)
        self.quantity_cap_ = float(prepared["Quantity"].quantile(0.95))

        for col in self.onehot_columns:
            self.categories_[col] = _sorted_unique(prepared[col])

        for col in self.label_columns:
            values = _sorted_unique(prepared[col])
            self.label_maps_[col] = {value: idx for idx, value in enumerate(values)}

        encoded = self._encode(prepared, fit_mode=True)
        self.feature_columns_ = encoded.columns.tolist()
        self.scaler_ = MinMaxScaler()
        self.scaler_.fit(encoded)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.scaler_ is None:
            raise RuntimeError("Preprocessor has not been fitted.")

        prepared = self._basic_prepare(X)
        encoded = self._encode(prepared, fit_mode=False)
        encoded = encoded.reindex(columns=self.feature_columns_, fill_value=0)
        scaled = self.scaler_.transform(encoded)
        return pd.DataFrame(scaled, columns=self.feature_columns_, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def _basic_prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        prepared = X.copy()

        for col in RAW_FEATURES:
            if col not in prepared.columns:
                prepared[col] = np.nan

        prepared = prepared[RAW_FEATURES].copy()
        prepared["Quantity"] = pd.to_numeric(prepared["Quantity"], errors="coerce")
        prepared["Amount"] = pd.to_numeric(prepared["Amount"], errors="coerce")

        for col in self.numeric_columns:
            fallback_value = self.numeric_medians_.get(col)
            if fallback_value is None:
                fallback_value = prepared[col].median()
                fallback_value = 0.0 if pd.isna(fallback_value) else float(fallback_value)
            prepared[col] = prepared[col].fillna(fallback_value)

        for col in ["Category", "Sub-Category", "PaymentMode", "State", "City"]:
            prepared[col] = prepared[col].fillna("Unknown").astype(str)

        if self.quantity_upper_bound_ is not None and self.quantity_cap_ is not None:
            prepared["Quantity"] = np.where(
                prepared["Quantity"] > self.quantity_upper_bound_,
                self.quantity_cap_,
                prepared["Quantity"],
            )

        prepared["Amount"] = np.log1p(prepared["Amount"].clip(lower=0))
        return prepared

    def _encode(self, prepared: pd.DataFrame, fit_mode: bool) -> pd.DataFrame:
        encoded = pd.DataFrame(index=prepared.index)
        encoded["Quantity"] = prepared["Quantity"].astype(float)
        encoded["Amount"] = prepared["Amount"].astype(float)

        for col in self.onehot_columns:
            categories = self.categories_[col] if not fit_mode else self.categories_[col]
            for category in categories:
                encoded[f"{col}_{category}"] = (prepared[col] == category).astype(int)

        for col in self.label_columns:
            mapping = self.label_maps_[col]
            encoded[f"{col}_LabelEncoded"] = (
                prepared[col].map(mapping).fillna(-1).astype(int)
            )

        return encoded
