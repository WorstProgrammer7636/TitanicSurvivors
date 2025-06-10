from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_logistic_model(df):
    X = df.drop(columns=['Survived'], axis=1)
    y = df['Survived']

    model = LogisticRegressionCV(cv=5, scoring='roc_auc', max_iter=1000)
    model.fit(X, y)

    y_pred_prob = model.predict_proba(X)[:, 1]
    print("Training AUC: ", roc_auc_score(y, y_pred_prob))
    return model

def train_random_forest_model(df):
    X = df.drop(columns=['Survived'], axis=1)
    y = df['Survived']
    random_forest_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=5,
    min_samples_split=10,
    random_state=207
    )

    random_forest_model.fit(X, y)

    return random_forest_model

def train_xgboost_model(df):
    X = df.drop(columns=['Survived'], axis=1)
    y = df['Survived']
    xgb_model = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, use_label_encoder=False, eval_metric='logloss', random_state=207)
    xgb_model.fit(X, y)
    return xgb_model

