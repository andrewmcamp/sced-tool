import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer

# --- Configuration ---
CSV_PATH = "00_resources/sced_v13.csv"
DB_OUTPUT_PATH = "./sced_lancedb"
TABLE_NAME = "sced_courses"


def create_database():
    print(f"1. Loading data from {CSV_PATH}...")
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"   Loaded {len(df)} rows.")
    except FileNotFoundError:
        print("Error: CSV file not found.")
        return

    print("2. Loading embedding model (all-MiniLM-L6-v2)...")
    # We use this model to turn text into numbers
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("3. Generating embeddings...")

    # Helper to clean and encode text
    def generate_vector(text):
        # Handle NaN/None values safely
        clean_text = str(text).replace("\n", " ").strip() if text else ""
        return model.encode(clean_text).tolist()

    # We combine Title and Description for the embedding to give the search better context
    # (e.g. searching for "stats" will match the title "Statistics" even if description is vague)
    df["combined_text"] = (
        df["sced_course_title"].fillna("")
        + ": "
        + df["sced_course_description"].fillna("")
    )

    # Create the vector column
    df["vector"] = df["combined_text"].apply(generate_vector)

    print(f"4. Saving database to {DB_OUTPUT_PATH}...")
    db = lancedb.connect(DB_OUTPUT_PATH)

    # Overwrite if exists to ensure a fresh start
    db.create_table(TABLE_NAME, data=df, mode="overwrite")

    print("Success! Database created.")
    print(f"Action Item: Copy the '{DB_OUTPUT_PATH}' folder to your server.")


if __name__ == "__main__":
    create_database()
