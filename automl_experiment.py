import pandas as pd
import numpy as np
import shap
from scipy.stats import zscore
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import itertools


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Delayed_Payment_Ratio'] = df['Num_of_Delayed_Payment'] / (df['Credit_History_Months'] + 1)
    df['Loan_to_Income'] = df['Num_of_Loan'] / (df['Annual_Income'] + 1)
    df['Balance_to_EMI'] = df['Monthly_Balance'] / (df['Total_EMI_per_month'] + 1)
    df['Salary_Ratio'] = df['Monthly_Inhand_Salary'] / (df['Monthly_Balance'] + 1)
    df['EMI_to_Income'] = df['Total_EMI_per_month'] / (df['Annual_Income'] + 1)
    df['Debt_per_Loan'] = df['Outstanding_Debt'] / (df['Num_of_Loan'] + 1)
    return df


def shap_select(X: pd.DataFrame, y: pd.Series, top: int = 15) -> list:
    """Select top features using SHAP values computed on a sample."""
    imp = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss",
        n_jobs=-1,
    )
    model.fit(X_imp, y)

    # Use a random subset to speed up SHAP computation
    x_sample = X_imp.sample(n=min(500, len(X_imp)), random_state=42)
    explainer = shap.TreeExplainer(model, data=x_sample, approximate=True)
    shap_values = explainer.shap_values(x_sample)
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)
    scores = np.abs(shap_values).mean(axis=(0, 2))
    idx = np.argsort(scores)[-top:]
    cols = X.columns.to_numpy()[idx]
    return list(cols)


def remove_outliers(X: pd.DataFrame, y: pd.Series):
    imp = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
    z = np.abs(zscore(X_imp))
    mask = (z < 3).all(axis=1)
    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)


def build_model(name: str):
    if name == 'rf':
        return RandomForestClassifier(n_estimators=300, max_depth=15,
                                      min_samples_split=2, min_samples_leaf=1,
                                      class_weight='balanced', random_state=42,
                                      n_jobs=-1)
    elif name == 'xgb':
        return XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=42, eval_metric='mlogloss',
                             n_jobs=-1)
    else:
        return GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                          max_depth=5, subsample=0.8,
                                          random_state=42)


def evaluate_combo(X, y, X_test, feature_selection, scaler_type, outlier,
                    model_name, postprocess):
    X_use = X.copy()
    y_use = y.copy()

    if outlier:
        X_use, y_use = remove_outliers(X_use, y_use)

    if feature_selection:
        selected = shap_select(X_use, y_use, top=15)
    else:
        selected = [c for c in X_use.columns if c != 'Annual_Income']

    X_use = X_use[selected]
    X_test_use = X_test[selected]

    scaler = StandardScaler() if scaler_type == 'standard' else RobustScaler()
    model = build_model(model_name)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    scores = []
    for tr_idx, val_idx in skf.split(X_use, y_use):
        X_tr, X_val = X_use.iloc[tr_idx], X_use.iloc[val_idx]
        y_tr, y_val = y_use.iloc[tr_idx], y_use.iloc[val_idx]

        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', scaler),
            ('model', model)
        ])
        pipe.fit(X_tr, y_tr)

        if postprocess:
            proba = pipe.predict_proba(X_val)
            preds = np.argmax(proba, axis=1)
            for i in range(len(preds)):
                if preds[i] == 2 and proba[i, 2] < 0.4:
                    preds[i] = 1
        else:
            preds = pipe.predict(X_val)
        scores.append(f1_score(y_val, preds, average='macro'))
    return np.mean(scores), X_use, y_use, X_test_use, scaler, model


def final_train_predict(X, y, X_test, scaler, model, postprocess):
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', scaler),
        ('model', model)
    ])
    pipe.fit(X, y)
    if postprocess:
        proba = pipe.predict_proba(X_test)
        preds = np.argmax(proba, axis=1)
        for i in range(len(preds)):
            if preds[i] == 2 and proba[i, 2] < 0.4:
                preds[i] = 1
        return preds
    else:
        return pipe.predict(X_test)


def main():
    x_train = pd.read_csv('ML_x_train.csv')
    y_train = pd.read_csv('ML_y_train.csv')['Credit_Score']
    x_test = pd.read_csv('ML_x_test.csv')
    sample = pd.read_csv('ML_sample_submission.csv')

    x_train = add_features(x_train)
    x_test = add_features(x_test)

    options = list(itertools.product(
        [True, False],
        ['standard', 'robust'],
        [True, False],
        ['rf', 'xgb', 'gb'],
        [True, False]
    ))

    results = []
    combos = {}
    for feat_sel, scaler, outlier, model, post in options:
        score, X_proc, y_proc, X_test_proc, scaler_obj, model_obj = evaluate_combo(
            x_train, y_train, x_test, feat_sel, scaler, outlier, model, post)
        results.append({
            'feature_selection': feat_sel,
            'scaler': scaler,
            'outlier_removal': outlier,
            'model': model,
            'postprocess': post,
            'f1_macro': score
        })
        combos[(feat_sel, scaler, outlier, model, post)] = (
            X_proc, y_proc, X_test_proc, scaler_obj, model_obj)
        print(f'Combo {(feat_sel, scaler, outlier, model, post)} -> {score:.4f}')

    result_df = pd.DataFrame(results)
    print(result_df.sort_values('f1_macro', ascending=False))

    best = result_df.sort_values('f1_macro', ascending=False).iloc[0]
    best_key = (best.feature_selection, best.scaler, best.outlier_removal,
                best.model, best.postprocess)
    print(f'Best Combo: {best_key} with F1 {best.f1_macro:.4f}')

    X_best, y_best, X_test_best, scaler_best, model_best = combos[best_key]
    preds = final_train_predict(X_best, y_best, X_test_best,
                                scaler_best, model_best, best_key[4])

    sample['Credit_Score'] = preds
    sample.to_csv('submission.csv', index=False)
    print('Saved submission.csv')


if __name__ == '__main__':
    main()
