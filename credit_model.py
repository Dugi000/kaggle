import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Load data
X_train = pd.read_csv('ML_x_train.csv')
y_train = pd.read_csv('ML_y_train.csv')['Credit_Score']
X_test = pd.read_csv('ML_x_test.csv')

# Preprocessing pipelines
scale_preprocess = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

tree_preprocess = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

models = {
    'log_reg': (scale_preprocess, LogisticRegression(max_iter=500, multi_class='auto')),
    'random_forest': (tree_preprocess, RandomForestClassifier(n_estimators=200, random_state=42)),
    'xgboost': (tree_preprocess, XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        tree_method='hist',
        random_state=42
    )),
    'catboost': (None, CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=8,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        verbose=False,
        random_seed=42
    ))
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cat_features = ['Occupation', 'Payment_of_Min_Amount']
cat_idx = [X_train.columns.get_loc(c) for c in cat_features]

for name, (prep, model) in models.items():
    if prep is not None:
        pipe = Pipeline([
            ('preprocess', prep),
            ('model', model)
        ])
        fit_params = {}
    else:
        pipe = model
        fit_params = {'cat_features': cat_idx}
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy', fit_params=fit_params, n_jobs=-1)
    results[name] = scores.mean()
    print(f"{name} accuracy: {scores.mean():.4f} +- {scores.std():.4f}")

best_model_name = max(results, key=results.get)
print(f"Best model: {best_model_name} (accuracy={results[best_model_name]:.4f})")

best_prep, best_estimator = models[best_model_name]
if best_prep is not None:
    best_model = Pipeline([
        ('preprocess', best_prep),
        ('model', best_estimator)
    ])
    fit_params = {}
else:
    best_model = best_estimator
    fit_params = {'cat_features': cat_idx}

best_model.fit(X_train, y_train, **fit_params)

preds = best_model.predict(X_test)

# Prepare submission
submission = pd.DataFrame({
    'Id': range(len(preds)),
    'Credit_Score': preds
})
submission.to_csv('submission.csv', index=False)
print("Saved predictions to submission.csv")
