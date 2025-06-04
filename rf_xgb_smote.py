import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer to clip outliers using IQR
class OutlierClipper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.lower = {}
        self.upper = {}
        for col in X.columns:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.lower[col] = q1 - 1.5 * iqr
            self.upper[col] = q3 + 1.5 * iqr
        return self

    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            X[col] = X[col].clip(self.lower[col], self.upper[col])
        return X

# Feature engineering: add loan_to_income_ratio
class FeatureAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'loan_to_income_ratio' not in X.columns:
            X['loan_to_income_ratio'] = X['Total_EMI_per_month'] / (X['Monthly_Inhand_Salary'] + 1)
        return X


def load_data():
    X_train = pd.read_csv('ML_x_train.csv')
    y_train = pd.read_csv('ML_y_train.csv')['Credit_Score']
    X_test = pd.read_csv('ML_x_test.csv')
    return X_train, y_train, X_test


def build_pipeline(model):
    steps = [
        ('feature', FeatureAdder()),
        ('clip', OutlierClipper()),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ]
    return Pipeline(steps)


def main():
    X_train, y_train, X_test = load_data()

    scoring = make_scorer(f1_score, average='macro')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # RandomForest setup
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight='balanced'
    )
    rf_pipe = build_pipeline(rf)
    rf_score = cross_val_score(rf_pipe, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1).mean()
    print("RandomForest CV F1:", rf_score)

    # XGBoost setup
    xgb = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        num_class=3,
        tree_method='hist',
        random_state=42,
        use_label_encoder=False
    )
    xgb_pipe = build_pipeline(xgb)
    xgb_score = cross_val_score(xgb_pipe, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1).mean()
    print("XGBoost CV F1:", xgb_score)

    # Choose best model
    if xgb_score >= rf_score:
        best_model = xgb_pipe
        best_name = 'XGBoost'
        best_score = xgb_score
    else:
        best_model = rf_pipe
        best_name = 'RandomForest'
        best_score = rf_score
    print("Selected model:", best_name, 'F1:', best_score)

    best_model.fit(X_train, y_train)
    preds = best_model.predict(X_test)

    submission = pd.DataFrame({'Id': range(len(preds)), 'Credit_Score': preds})
    submission.to_csv('submission.csv', index=False)
    print('Saved submission.csv')

    # Provide a short Korean summary
    print('\n\n결과 요약')
    print('선택된 모델:', best_name)
    print('교차검증 Macro F1 점수:', round(best_score, 4))

if __name__ == '__main__':
    main()
