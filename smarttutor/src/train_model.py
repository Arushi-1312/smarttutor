import joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from pathlib import Path
from features import add_features
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
def train_and_save(data_path, model_path=MODEL_DIR/"model.joblib"):
    df = pd.read_csv(data_path)
    df = add_features(df)
    X = df[["study_hours_norm","assignment_rate","attendance","grade_mean","attend_x_study"]]
    y = df["high_performer"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train,y_train)
    preds = clf.predict(X_test)
    print(classification_report(y_test,preds))
    print("AUC:", roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))
    joblib.dump({"model": clf, "features": X.columns.tolist()}, model_path)
if __name__ == "__main__":
    datafile = Path(__file__).resolve().parents[1] / "data" / "synthetic_students.csv"
    train_and_save(str(datafile))
