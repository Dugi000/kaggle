# Improved model script with preprocessing and validation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.utils.class_weight import compute_class_weight


def load_data():
    x_train = pd.read_csv('ML_x_train.csv')
    y_train = pd.read_csv('ML_y_train.csv')['Credit_Score']
    x_test = pd.read_csv('ML_x_test.csv')
    return x_train, y_train, x_test


def add_features(df):
    df['loan_to_income_ratio'] = df['Total_EMI_per_month'] / (df['Monthly_Inhand_Salary'] + 1)
    return df


def build_preprocessor(x):
    float_cols = x.select_dtypes(include='float').columns.tolist()
    int_cols = [c for c in x.columns if c not in float_cols]

    float_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    int_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])

    preprocessor = ColumnTransformer([
        ('float', float_pipe, float_cols),
        ('int', int_pipe, int_cols)
    ])
    return preprocessor


def get_class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    return {cls: w for cls, w in zip(classes, weights)}


def evaluate_model(model, X_val, y_val):
    pred = model.predict(X_val)
    return f1_score(y_val, pred, average='macro')


def main():
    x_train, y_train, x_test = load_data()

    # Feature engineering
    x_train = add_features(x_train)
    x_test = add_features(x_test)

    # Train/validation split
    X_trn, X_val, y_trn, y_val = train_test_split(
        x_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    preprocessor = build_preprocessor(X_trn)
    class_weights = get_class_weights(y_trn)

    scoring = make_scorer(f1_score, average='macro')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Logistic Regression
    log_pipe = Pipeline([
        ('prep', preprocessor),
        ('model', LogisticRegression(max_iter=1000, class_weight=class_weights, C=1.0))
    ])
    log_f1 = evaluate_model(log_pipe.fit(X_trn, y_trn), X_val, y_val)
    print('Logistic F1:', log_f1)

    # Random Forest
    rf_pipe = Pipeline([
        ('prep', preprocessor),
        ('model', RandomForestClassifier(random_state=42, class_weight=class_weights, n_estimators=120, max_depth=None))
    ])
    rf_f1 = evaluate_model(rf_pipe.fit(X_trn, y_trn), X_val, y_val)
    print('RF F1:', rf_f1)

    # XGBoost
    xgb_pipe = Pipeline([
        ('prep', preprocessor),
        ('model', XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=42,
            use_label_encoder=False,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9
        ))
    ])
    xgb_f1 = evaluate_model(xgb_pipe.fit(X_trn, y_trn), X_val, y_val)
    print('XGB F1:', xgb_f1)

    # LightGBM
    lgb_pipe = Pipeline([
        ('prep', preprocessor),
        ('model', LGBMClassifier(objective='multiclass', random_state=42, n_estimators=150, learning_rate=0.1, num_leaves=31))
    ])
    lgb_f1 = evaluate_model(lgb_pipe.fit(X_trn, y_trn), X_val, y_val)
    print('LGBM F1:', lgb_f1)

    # SVM
    svm_pipe = Pipeline([
        ('prep', preprocessor),
        ('model', SVC(class_weight=class_weights, probability=True, C=1.0, gamma='scale'))
    ])
    svm_f1 = evaluate_model(svm_pipe.fit(X_trn, y_trn), X_val, y_val)
    print('SVM F1:', svm_f1)

    # Stacking ensemble using best base models
    estimators = [
        ('rf', rf_pipe),
        ('xgb', xgb_pipe),
        ('lgb', lgb_pipe)
    ]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        n_jobs=-1,
        passthrough=False
    )
    stack.fit(X_trn, y_trn)
    stack_f1 = evaluate_model(stack, X_val, y_val)
    print('Stacking F1:', stack_f1)

    scores = {
        'logistic': (log_pipe, log_f1),
        'rf': (rf_pipe, rf_f1),
        'xgb': (xgb_pipe, xgb_f1),
        'lgbm': (lgb_pipe, lgb_f1),
        'svm': (svm_pipe, svm_f1),
        'stacking': (stack, stack_f1)
    }

    best_name = max(scores, key=lambda k: scores[k][1])
    best_model = scores[best_name][0]
    print('Best model:', best_name, 'F1:', scores[best_name][1])

    # Retrain best model on full training data
    best_model.fit(x_train, y_train)
    pred = best_model.predict(x_test)
    sub = pd.DataFrame({'Id': range(len(pred)), 'Credit_Score': pred})
    sub.to_csv('submission.csv', index=False)
    print('Saved submission.csv')


if __name__ == '__main__':
    main()
