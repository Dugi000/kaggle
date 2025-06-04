import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier


def main():
    x_train = pd.read_csv('ML_x_train.csv')
    y_train = pd.read_csv('ML_y_train.csv')['Credit_Score']
    x_test = pd.read_csv('ML_x_test.csv')

    num_features = x_train.columns

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_features)
    ])

    scoring = make_scorer(f1_score, average='macro')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {}

    pipe_log = Pipeline([('preprocess', preprocessor), ('model', LogisticRegression(max_iter=1000))])
    score_log = cross_val_score(pipe_log, x_train, y_train, cv=cv, scoring=scoring).mean()
    models['logreg'] = (pipe_log, score_log)
    print('LogReg CV Macro F1:', score_log)

    pipe_rf = Pipeline([('preprocess', preprocessor), ('model', RandomForestClassifier(n_estimators=100, random_state=42))])
    score_rf = cross_val_score(pipe_rf, x_train, y_train, cv=cv, scoring=scoring, n_jobs=-1).mean()
    models['rf'] = (pipe_rf, score_rf)
    print('RF CV Macro F1:', score_rf)

    pipe_xgb = Pipeline([('preprocess', preprocessor), ('model', XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, eval_metric='mlogloss', random_state=42, verbosity=0))])
    score_xgb = cross_val_score(pipe_xgb, x_train, y_train, cv=cv, scoring=scoring, n_jobs=-1).mean()
    models['xgb'] = (pipe_xgb, score_xgb)
    print('XGB CV Macro F1:', score_xgb)

    pipe_lgb = Pipeline([('preprocess', preprocessor), ('model', LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=-1, random_state=42, verbose=-1))])
    score_lgb = cross_val_score(pipe_lgb, x_train, y_train, cv=cv, scoring=scoring, n_jobs=-1).mean()
    models['lgbm'] = (pipe_lgb, score_lgb)
    print('LGBM CV Macro F1:', score_lgb)

    pipe_svm = Pipeline([('preprocess', preprocessor), ('model', SVC(C=2, probability=True))])
    score_svm = cross_val_score(pipe_svm, x_train, y_train, cv=cv, scoring=scoring, n_jobs=-1).mean()
    models['svm'] = (pipe_svm, score_svm)
    print('SVM CV Macro F1:', score_svm)

    voting_clf = VotingClassifier(estimators=[('rf', pipe_rf), ('xgb', pipe_xgb), ('lgbm', pipe_lgb)], voting='soft')
    score_voting = cross_val_score(voting_clf, x_train, y_train, cv=cv, scoring=scoring, n_jobs=-1).mean()
    models['ensemble'] = (voting_clf, score_voting)
    print('Ensemble CV Macro F1:', score_voting)

    best_name = max(models, key=lambda k: models[k][1])
    best_score = models[best_name][1]
    print('Best model:', best_name, best_score)

    best_model = models[best_name][0]
    best_model.fit(x_train, y_train)
    pred = best_model.predict(x_test)
    sub = pd.DataFrame({'Id': range(len(pred)), 'Credit_Score': pred})
    sub.to_csv('submission.csv', index=False)
    print('Saved submission.csv')

if __name__ == '__main__':
    main()
