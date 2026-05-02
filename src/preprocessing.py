import pandas as pd
import csv
import os
import re


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data(path: str) -> pd.DataFrame | None:
    """
    Load and validate the mental health dataset with robust CSV parsing.
    """

    print("Loading dataset...")

    try:
        df = pd.read_csv(
            path,
            encoding="utf-8",
            engine="python",
            quoting=csv.QUOTE_MINIMAL,
            on_bad_lines="skip"
        )
    except FileNotFoundError:
        print(f"Error: File not found at path -> {path}")
        return None
    except Exception as e:
        print(f"Error while loading dataset: {e}")
        return None

    if df is None or df.empty:
        print("Error: Loaded dataset is empty.")
        return None

    # Standardize column names
    df.columns = df.columns.str.strip()

    print("Dataset loaded successfully.")
    print("\nColumns:", df.columns.tolist())

    # Validate expected structure
    expected_columns = {"Context", "Response"}
    if not expected_columns.issubset(set(df.columns)):
        print("Error: Expected columns 'Context' and 'Response' not found.")
        return None

    # Keep only required columns
    df = df[["Context", "Response"]].copy()

    df = df.dropna(subset=["Context", "Response"])

    df["Context"] = df["Context"].astype(str)
    df["Response"] = df["Response"].astype(str)

    # Remove empty strings
    df = df[
        (df["Context"].str.strip() != "") &
        (df["Response"].str.strip() != "")
    ]

    print("\nInitial dataset shape:", df.shape)

    return df


# -----------------------------
# TEXT CLEANING (LIGHT — IMPORTANT DESIGN CHOICE)
# -----------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    # normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text


# -----------------------------
# PREPROCESS PIPELINE
# -----------------------------
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing pipeline optimized for semantic models.
    """

    print("\nStarting preprocessing...")

    df = df.copy()

    # Clean text
    df["clean_context"] = df["Context"].apply(clean_text)
    df["clean_response"] = df["Response"].apply(clean_text)

    # Remove duplicates (after normalization)
    before = len(df)
    df = df.drop_duplicates(subset=["clean_context", "clean_response"])
    after = len(df)

    print(f"Removed {before - after} duplicate rows")

    # Remove low-information samples
    df = df[
        (df["clean_context"].str.len() > 10) &
        (df["clean_response"].str.len() > 10)
    ]

    # Reset index
    df = df.reset_index(drop=True)

    print("Final dataset shape after preprocessing:", df.shape)

    return df


# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "data", "mental_health.csv")

    df = load_data(data_path)

    if df is not None:
        df = preprocess_dataframe(df)

        print("\nSample cleaned data:")
        print(df[["clean_context", "clean_response"]].head())

## “Because this is a mental health conversational dataset, where emotional meaning is critical. Aggressive preprocessing like stopword removal or stemming can distort meaning. Since we use semantic similarity models, preserving original structure improves performance.”
