import pandas as pd
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["study_hours_norm"] = (df["study_hours"] - df["study_hours"].mean()) / (df["study_hours"].std()+1e-9)
    df["assignment_rate"] = df["assignments"] / 10.0
    df["grade_mean"] = df[["math_grade","ai_grade"]].mean(axis=1)
    df["attend_x_study"] = df["attendance"] * df["study_hours_norm"]
    return df
