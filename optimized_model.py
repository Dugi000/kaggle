import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Load datasets with fallback to alternative file names
def load_datasets():
    try:
        X_train = pd.read_csv('x_train.csv')
        y_train = pd.read_csv('y_train.csv')['Credit_Score']
        X_test = pd.read_csv('x_test.csv')
    except FileNotFoundError:
        X_train = pd.read_csv('ML_x_train.csv')
        y_train = pd.read_csv('ML_y_train.csv')['Credit_Score']
        X_test = pd.read_csv('ML_x_test.csv')
    return X_train, y_train, X_test

# Derived ratio features
def add_ratios(df):
    df = df.copy()
    df['Delayed_Payment_Ratio'] = df['Num_of_Delayed_Payment'] / (df['Credit_History_Months'] + 1)
    df['Loan_to_Income'] = df['Num_of_Loan'] / (df['Annual_Income'] + 1)
    df['Balance_to_EMI'] = df['Monthly_Balance'] / (df['Total_EMI_per_month'] + 1)
    df['Salary_Ratio'] = df['Monthly_Inhand_Salary'] / (df['Monthly_Balance'] + 1)
    df['EMI_to_Income'] = df['Total_EMI_per_month'] / (df['Annual_Income'] + 1)
    df['Debt_per_Loan'] = df['Outstanding_Debt'] / (df['Num_of_Loan'] + 1)
    return df

# Build preprocessing pipeline

def build_preprocessor(cols):
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, cols)
    ])
    return preprocessor


def evaluate_model(pipe, X, y, cv, scoring):
    return cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1).mean()


def main():
    X_train, y_train, X_test = load_datasets()

    # Print basic info
    print('Train shape:', X_train.shape)
    print('Class distribution:', y_train.value_counts().to_dict())

    # Feature engineering
    X_train = add_ratios(X_train)
    X_test = add_ratios(X_test)

    features = X_train.columns
    preprocessor = build_preprocessor(features)

    scoring = make_scorer(f1_score, average='macro')
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    results = {}

    # Logistic Regression
    log_pipe = ImbPipeline([
        ('pre', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])
    results['logreg'] = evaluate_model(log_pipe, X_train, y_train, cv, scoring)
    print('LogisticRegression F1:', results['logreg'])

    # RandomForest
    rf_pipe = ImbPipeline([
        ('pre', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    results['rf'] = evaluate_model(rf_pipe, X_train, y_train, cv, scoring)
    print('RandomForest F1:', results['rf'])

    # XGBoost
    xgb_pipe = ImbPipeline([
        ('pre', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=42,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.8
        ))
    ])
    results['xgb'] = evaluate_model(xgb_pipe, X_train, y_train, cv, scoring)
    print('XGBoost F1:', results['xgb'])

    # LightGBM (simple version)
    lgb_pipe = ImbPipeline([
        ('pre', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', LGBMClassifier(random_state=42, n_estimators=100))
    ])
    results['lgbm'] = evaluate_model(lgb_pipe, X_train, y_train, cv, scoring)
    print('LightGBM F1:', results['lgbm'])

    # Voting ensemble
    voting = VotingClassifier(
        estimators=[('rf', rf_pipe), ('xgb', xgb_pipe), ('lgbm', lgb_pipe)],
        voting='soft',
        n_jobs=-1
    )
    results['voting'] = evaluate_model(voting, X_train, y_train, cv, scoring)
    print('Voting ensemble F1:', results['voting'])

    best_model_name = max(results, key=results.get)
    best_score = results[best_model_name]
    print('Best model:', best_model_name, best_score)

    if best_model_name == 'logreg':
        best_model = log_pipe
    elif best_model_name == 'rf':
        best_model = rf_pipe
    elif best_model_name == 'xgb':
        best_model = xgb_pipe
    elif best_model_name == 'lgbm':
        best_model = lgb_pipe
    else:
        best_model = voting

    # Train best model on full data
    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)

    submission = pd.DataFrame({'Id': range(len(predictions)), 'Credit_Score': predictions})
    submission.to_csv('submission.csv', index=False)
    print('Saved submission.csv')

    print('\n최적 모델:', best_model_name)
    print('교차검증 평균 Macro F1:', round(best_score, 4))

if __name__ == '__main__':
    main()
