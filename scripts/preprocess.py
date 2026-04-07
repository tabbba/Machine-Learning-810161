from pathlib import Path

import numpy as np
import pandas as pd

TEAM_MAPPING = {
    "content": "Content",
    "CONTENT": "Content",
    "Contennt": "Content",
    "media": "Media",
    "MEDIA": "Media",
    "Paid Media": "Media",
    "seo": "SEO",
    "SEO ": "SEO",
    "design": "Design",
    "DESIGN": "Design",
    "Desgn": "Design",
}

def consolidate_task_type(value: object) -> str:
    """Apply the task-type normalization"""
    if pd.isna(value):
        return "unknown"

    normalized = str(value).lower().strip().replace("_task", "")

    if "article" in normalized or "artcle" in normalized or "blog" in normalized:
        return "article"
    if "design" in normalized or "creative" in normalized:
        return "design"
    if "ticket" in normalized or "support" in normalized:
        return "ticket"
    if "report" in normalized or "repport" in normalized:
        return "report"
    if "ad" in normalized:
        return "ad"
    if "dev" in normalized:
        return "dev"
    if "rele" in normalized:
        return "release"

    return normalized


def normalize_legacy_ai_flag(series: pd.Series) -> pd.Series:
    """Map legacy flag values to booleans"""
    normalized = series.replace("unknown", np.nan)

    if pd.api.types.is_bool_dtype(normalized):
        return normalized

    return normalized.map(
        {
            "true": True,
            "false": False,
            True: True,
            False: False,
        }
    )


# IMPORTANT
# Findings not modified to clarify:
# - `rework_hours > hours_spent`, because the metric definition is ambiguous.
# - `ai_assisted` / `ai_usage_pct` mismatches, because no correction rule was chosen.
def preprocess(
    df: pd.DataFrame,
    copy: bool = False,
) -> pd.DataFrame:
    """
    Apply the post EDA preprocessing decisions from the exploratory notebook`.

    """
    if copy:
        df = df.copy()

    df["billable_hours"] = df["billable_hours"].clip(lower=0)
    df["is_loss"] = (df["profit"] < 0).astype(int)

    df["team"] = df["team"].replace(TEAM_MAPPING)
    df["task_type"] = df["task_type"].apply(consolidate_task_type)
    df["legacy_ai_flag"] = normalize_legacy_ai_flag(df["legacy_ai_flag"])

    df["updated_at"] = pd.to_datetime(df["updated_at"])
    df = (
        df.sort_values("updated_at", ascending=False)
        .drop_duplicates(subset="task_id", keep="first")
        .reset_index(drop=True)
    )

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["delivered_at"] = pd.to_datetime(df["delivered_at"], errors="coerce")

    valid_date_order = df["delivered_at"].isna() | (df["delivered_at"] >= df["created_at"])
    df = df.loc[valid_date_order].reset_index(drop=True)

    return df


def load_preprocessed_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the CSV and return the cleaned dataframe."""
    df = pd.read_csv(csv_path)
    return preprocess(df)