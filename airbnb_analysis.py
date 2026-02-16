"""Airbnb NYC 2024 exploratory data analysis.

This script recreates a complete analysis workflow using:
- numpy
- pandas
- matplotlib
- seaborn

It loads `datasets.csv`, cleans the data, prints key insights, and saves
visualizations to the `outputs/` folder.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DATASET_PATH = Path("datasets.csv")
OUTPUT_DIR = Path("outputs")



def clean_airbnb_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the Airbnb dataframe."""
    cleaned = df.copy()

    # Normalize column names.
    cleaned.columns = [col.strip().lower() for col in cleaned.columns]

    # Convert numeric columns.
    numeric_cols = [
        "price",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "number_of_reviews_ltm",
        "rating",
        "bedrooms",
        "beds",
        "baths",
    ]
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # Convert dates.
    cleaned["last_review"] = pd.to_datetime(cleaned["last_review"], errors="coerce", dayfirst=True)

    # Fill missing values in selected columns.
    cleaned["reviews_per_month"] = cleaned["reviews_per_month"].fillna(0)
    cleaned["rating"] = cleaned["rating"].fillna(cleaned["rating"].median())

    # Remove impossible prices and duplicated IDs.
    cleaned = cleaned[cleaned["price"].notna() & (cleaned["price"] > 0)]
    cleaned = cleaned.drop_duplicates(subset=["id"])

    return cleaned



def print_key_insights(df: pd.DataFrame) -> None:
    """Print concise EDA insights in terminal."""
    print("\n=== Airbnb NYC EDA Summary ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.shape[1]}")
    print(f"Average price: ${df['price'].mean():.2f}")
    print(f"Median price: ${df['price'].median():.2f}")
    print(f"Average rating: {df['rating'].mean():.2f}")
    print("\nTop neighbourhood groups by listing count:")
    print(df["neighbourhood_group"].value_counts().head(5).to_string())
    print("\nRoom type distribution:")
    print(df["room_type"].value_counts(normalize=True).mul(100).round(2).to_string())



def save_visualizations(df: pd.DataFrame, output_dir: Path) -> None:
    """Create and save EDA plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # 1) Price distribution (capped at 99th percentile for readability).
    upper_price = np.nanpercentile(df["price"], 99)
    plt.figure(figsize=(10, 5))
    sns.histplot(df.loc[df["price"] <= upper_price, "price"], bins=50, kde=True)
    plt.title("Price Distribution (up to 99th percentile)")
    plt.xlabel("Price")
    plt.ylabel("Listings")
    plt.tight_layout()
    plt.savefig(output_dir / "price_distribution.png", dpi=150)
    plt.close()

    # 2) Average price by neighbourhood group.
    plt.figure(figsize=(8, 5))
    order = df.groupby("neighbourhood_group")["price"].mean().sort_values(ascending=False).index
    sns.barplot(data=df, x="neighbourhood_group", y="price", order=order, estimator=np.mean, errorbar=None)
    plt.title("Average Price by Neighbourhood Group")
    plt.xlabel("Neighbourhood Group")
    plt.ylabel("Average Price")
    plt.tight_layout()
    plt.savefig(output_dir / "avg_price_by_neighbourhood_group.png", dpi=150)
    plt.close()

    # 3) Room type count.
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="room_type", order=df["room_type"].value_counts().index)
    plt.title("Listing Count by Room Type")
    plt.xlabel("Room Type")
    plt.ylabel("Count")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_dir / "room_type_count.png", dpi=150)
    plt.close()

    # 4) Rating vs Price scatter (sampled for readability).
    sample = df[["rating", "price"]].dropna()
    if len(sample) > 10_000:
        sample = sample.sample(10_000, random_state=42)

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=sample, x="rating", y="price", alpha=0.3)
    plt.title("Rating vs Price")
    plt.xlabel("Rating")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(output_dir / "rating_vs_price.png", dpi=150)
    plt.close()



def main() -> None:
    """Run complete EDA pipeline."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    raw_df = pd.read_csv(DATASET_PATH, encoding_errors="ignore")
    cleaned_df = clean_airbnb_data(raw_df)

    print_key_insights(cleaned_df)
    save_visualizations(cleaned_df, OUTPUT_DIR)

    cleaned_df.to_csv(OUTPUT_DIR / "cleaned_airbnb_data.csv", index=False)
    print(f"\nSaved cleaned data and charts in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
