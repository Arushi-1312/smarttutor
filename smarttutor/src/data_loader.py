# src/data_loader.py
import argparse, pandas as pd, numpy as np
from pathlib import Path
DATA_PATH = Path(__file__).resolve().parents[1] / "data"
DATA_PATH.mkdir(parents=True, exist_ok=True)
OUTFILE = DATA_PATH / "synthetic_students.csv"
def generate_synthetic_data(n=1000, seed=42):
    np.random.seed(seed)
    study_hours = np.clip(np.random.normal(10, 4, n), 0, 40)
    assignments = np.random.binomial(10, p=0.7, size=n)
    attendance = np.clip(np.random.normal(80, 10, n), 40, 100)
    math_grade = np.clip(np.random.normal(8, 1.2, n), 4, 10)
    ai_grade = np.clip(np.random.normal(8.5, 1.0, n), 4, 10)
    cgpa = np.clip((math_grade*0.25 + ai_grade*0.35 + (study_hours/40)*1.5 + (assignments/10)*1.5), 4, 10)
    high_performer = ((cgpa >= 8.5) & (math_grade >= 8) & (ai_grade >= 8)).astype(int)
    return pd.DataFrame({
        "study_hours": study_hours, "assignments": assignments, "attendance": attendance,
        "math_grade": math_grade, "ai_grade": ai_grade, "cgpa": cgpa, "high_performer": high_performer})
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    if args.generate:
        df = generate_synthetic_data(1500)
        df.to_csv(OUTFILE, index=False)
        print(f"Wrote {OUTFILE}")
    else:
        print("Run with --generate to create data.")
