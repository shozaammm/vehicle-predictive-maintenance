import os
import glob
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score

class VehiclePredictiveMaintenanceModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.rf_classifier = None
        self.isolation_forest = None
        self.feature_names = None
        self.is_trained = False

    def load_from_datasets_folder(self, folder_path="Datasets"):
        """
        Load CSVs from the Datasets folder and pick a suitable dataset.
        """
        if not os.path.exists(folder_path):
            print(f"[WARN] Datasets folder not found at: {folder_path}")
            return None
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        print(f"[INFO] Found {len(csv_files)} CSV file(s) in {folder_path}")
        if not csv_files:
            return None
        dataframes = []
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                non_empty_cols = df.columns[df.notnull().any()].tolist()
                df = df[non_empty_cols]
                print(f"  - {os.path.basename(f)} | shape={df.shape}")
                dataframes.append(df)
            except Exception as e:
                print(f"    [ERROR] Could not read {f}: {e}")
        if not dataframes:
            return None
        best_df = max(dataframes, key=lambda d: d.shape[0])
        print(f"[INFO] Selected dataset: rows={best_df.shape[0]}, cols={best_df.shape[1]}")
        return best_df

    def prepare_data(self, df: pd.DataFrame):
        """
        Prepare numeric features and determine/create a binary target column.
        """
        print("[INFO] Preparing features and target")
        df = df.loc[:, ~df.columns.duplicated()].copy()
        df.columns = [str(c).strip() for c in df.columns]
        target_candidates = [
            c for c in df.columns
            if any(k in c.lower() for k in ["failure", "fail", "target", "label", "class", "y"])
        ]
        if target_candidates:
            target_col = target_candidates[0]
            print(f"[INFO] Using target column: {target_col}")
            y_raw = df[target_col]
            if not np.issubdtype(y_raw.dtype, np.number):
                y = y_raw.astype("category").cat.codes
            else:
                y = y_raw.astype(float)
                if y.nunique() > 2:
                    y = (y > y.median()).astype(int)
                else:
                    y = y.astype(int)
            X = df.drop(columns=[target_col])
        else:
            print("[WARN] No explicit target found. Creating a proxy target from sensor patterns.")
            X = df.copy()
            temp_cols = [c for c in X.columns if "temp" in c.lower()]
            rpm_cols  = [c for c in X.columns if "rpm"  in c.lower()]
            load_cols = [c for c in X.columns if "load" in c.lower()]
            risk = np.zeros(len(X), dtype=float)
            if temp_cols:
                t = pd.to_numeric(X[temp_cols], errors="coerce")
                thr = np.nanpercentile(t, 85)
                risk += np.nan_to_num((t > thr).astype(float) * 0.6)
            if rpm_cols:
                r = pd.to_numeric(X[rpm_cols], errors="coerce")
                thr = np.nanpercentile(r, 85)
                risk += np.nan_to_num((r > thr).astype(float) * 0.2)
            if load_cols:
                l = pd.to_numeric(X[load_cols], errors="coerce")
                thr = np.nanpercentile(l, 85)
                risk += np.nan_to_num((l > thr).astype(float) * 0.2)
            y = (risk >= 0.6).astype(int)
            print(f"[INFO] Created proxy target. Failure rate: {y.mean():.2%}")
        X_num = X.select_dtypes(include=[np.number]).copy()
        if X_num.shape[1] == 0:
            X_num = X.apply(pd.to_numeric, errors="coerce").select_dtypes(include=[np.number]).copy()
        X_num = X_num.dropna(axis=1, how="all")
        X_num = X_num.fillna(X_num.mean())
        X_num = X_num.clip(lower=X_num.quantile(0.001), upper=X_num.quantile(0.999), axis=1)
        if X_num.shape[1] > 12:
            variances = X_num.var().sort_values(ascending=False)
            keep_cols = variances.head(12).index.tolist()
            X_num = X_num[keep_cols]
        self.feature_names = list(X_num.columns)
        print(f"[INFO] Using {len(self.feature_names)} feature(s): {self.feature_names}")
        return X_num, y

    def train(self, X: pd.DataFrame, y: pd.Series):
        print("[INFO] Training models")
        X_scaled = self.scaler.fit_transform(X)
        strat = y if y.nunique() > 1 else None
        Xtr, Xte, ytr, yte = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=strat
        )
        self.rf_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=14,
            min_samples_split=5,
            class_weight="balanced" if y.nunique() > 1 else None,
            random_state=42
        )
        self.rf_classifier.fit(Xtr, ytr)
        ypred = self.rf_classifier.predict(Xte)
        acc = accuracy_score(yte, ypred)
        print(f"[RESULT] RandomForest accuracy: {acc:.3f}")
        normal = Xtr[ytr == 0] if y.nunique() > 1 else Xtr
        contam = float(np.clip(y.mean() if y.nunique() > 1 else 0.1, 0.01, 0.5))
        self.isolation_forest = IsolationForest(
            n_estimators=200, contamination=contam, random_state=42
        )
        self.isolation_forest.fit(normal)
        if hasattr(self.rf_classifier, "feature_importances_"):
            fi = pd.Series(self.rf_classifier.feature_importances_, index=self.feature_names)
            print("[INFO] Top feature importances:")
            print(fi.sort_values(ascending=False).head(8))
        self.is_trained = True
        return acc

    def predict_maintenance(self, input_row: dict | pd.DataFrame):
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")
        if isinstance(input_row, dict):
            df = pd.DataFrame([input_row])
        else:
            df = input_row.copy()
        for f in self.feature_names:
            if f not in df.columns:
                df[f] = 0.0
        df = df[self.feature_names]
        Xs = self.scaler.transform(df)
        if hasattr(self.rf_classifier, "predict_proba"):
            proba = self.rf_classifier.predict_proba(Xs)
            fail_prob = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        else:
            fail_prob = self.rf_classifier.predict(Xs).astype(float)
        anomaly = self.isolation_forest.decision_function(Xs)
        anomaly_risk = np.clip((0.5 - anomaly) / 0.5, 0, 1) * 0.2
        total_risk = np.clip(fail_prob + anomaly_risk, 0, 1)
        recs = []
        for r in total_risk:
            if r > 0.7:
                recs.append("URGENT: Schedule immediate maintenance!")
            elif r > 0.5:
                recs.append("HIGH RISK: Schedule maintenance within 1 week")
            elif r > 0.3:
                recs.append("MODERATE RISK: Schedule maintenance within 1 month")
            else:
                recs.append("LOW RISK: Continue regular maintenance schedule")
        return {
            "risk_scores": total_risk.tolist(),
            "failure_probabilities": fail_prob.tolist(),
            "recommendations": recs
        }

    def save(self, model_path="vehicle_maintenance_model.pkl",
             info_path="vehicle_maintenance_model_info.json"):
        with open(model_path, "wb") as f:
            pickle.dump(self, f)
        info = {
            "feature_names": self.feature_names,
            "is_trained": self.is_trained,
            "num_features": len(self.feature_names)
        }
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"[INFO] Saved model -> {model_path}")
        print(f"[INFO] Saved info  -> {info_path}")

def main():
    print("=== Vehicle Predictive Maintenance: Training ===")
    model = VehiclePredictiveMaintenanceModel()
    df = model.load_from_datasets_folder("Datasets")
    if df is None:
        print("[ERROR] No datasets loaded. Put CSVs in the Datasets folder and retry.")
        return
    X, y = model.prepare_data(df)
    acc = model.train(X, y)
    model.save()
    sample = {f: float(np.nan_to_num(X[f].median())) for f in model.feature_names}
    out = model.predict_maintenance(sample)
    print("\n=== Sample Prediction ===")
    print(f"Risk Score: {out['risk_scores'][0]:.2f}")
    print(f"Failure Probability: {out['failure_probabilities'][0]:.2f}")
    print(f"Recommendation: {out['recommendations'][0]}")

if __name__ == "__main__":
    main()
