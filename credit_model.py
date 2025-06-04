import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip numeric features using the interquartile range."""

    def fit(self, X, y=None):
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        self.lower_ = q1 - 1.5 * (q3 - q1)
        self.upper_ = q3 + 1.5 * (q3 - q1)
        return self

    def transform(self, X):
        return np.clip(X, self.lower_, self.upper_)

# Load data
X_train = pd.read_csv('ML_x_train.csv')
y_train = pd.read_csv('ML_y_train.csv')['Credit_Score']
X_test = pd.read_csv('ML_x_test.csv')

categorical_features = ['Occupation', 'Payment_of_Min_Amount']
numeric_features = [c for c in X_train.columns if c not in categorical_features]

numeric_preprocess = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('clip', IQRClipper()),
    ('scale', StandardScaler()),
])

categorical_preprocess = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

preprocess = ColumnTransformer([
    ('num', numeric_preprocess, numeric_features),
    ('cat', categorical_preprocess, categorical_features),
])

models = {
    'log_reg': ('sk', LogisticRegression(max_iter=500, multi_class='multinomial', solver='lbfgs')),
    'random_forest': ('sk', RandomForestClassifier(n_estimators=100, random_state=42)),
    'xgboost': ('sk', XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        tree_method='hist',
        random_state=42,
    )),
    'lightgbm': ('sk', LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )),
    'catboost': ('cat', CatBoostClassifier(
        iterations=100,
        learning_rate=0.05,
        depth=8,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        verbose=False,
        random_seed=42,
    )),
}

results = {}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cat_idx = [X_train.columns.get_loc(c) for c in categorical_features]

for name, (kind, model) in models.items():
    if kind == 'sk':
        pipe = Pipeline([
            ('preprocess', preprocess),
            ('model', model),
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=1)
    else:
        # CatBoost handles categorical features natively
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', fit_params={'cat_features': cat_idx}, n_jobs=1)
    results[name] = scores.mean()
    print(f"{name} accuracy: {scores.mean():.4f} +- {scores.std():.4f}")

best_model_name = max(results, key=results.get)
print(f"Best model: {best_model_name} (accuracy={results[best_model_name]:.4f})")

kind, estimator = models[best_model_name]
if kind == 'sk':
    best_model = Pipeline([
        ('preprocess', preprocess),
        ('model', estimator),
    ])
    best_model.fit(X_train, y_train)
    preds = best_model.predict(X_test)
else:
    estimator.fit(X_train, y_train, cat_features=cat_idx)
    preds = estimator.predict(X_test)

# Prepare submission
submission = pd.DataFrame({
    'Id': range(len(preds)),
    'Credit_Score': preds
})
submission.to_csv('submission.csv', index=False)
print("Saved predictions to submission.csv")
