import pandas as pd
from titanic import preprocessing, feature_engineering, modeling, utils, prep_test
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

df_train = pd.read_csv(r"C:\Users\5kyle\OneDrive\Desktop\titanic\data\train.csv")
df_test = pd.read_csv(r"C:\Users\5kyle\OneDrive\Desktop\titanic\data\test.csv")
df_test_raw = pd.read_csv(r"C:\Users\5kyle\OneDrive\Desktop\titanic\data\test.csv")
pd.set_option('display.max_columns', None)

#Prepare Training Dataset

df_train = feature_engineering.add_rare_titles(df_train)
df_train = feature_engineering.add_cabin(df_train)
df_train = feature_engineering.add_family_size(df_train)


df_train = preprocessing.drop_columns(df_train)
df_train = preprocessing.one_hot_encode(df_train)

imputer = preprocessing.fit_imputer(df_train)
scaler = preprocessing.fit_scaler(df_train)


df_train = preprocessing.apply_imputer(df_train, imputer)
df_train = preprocessing.apply_scaler(df_train, scaler)

print(df_train.head())

logisticModel = modeling.train_logistic_model(df_train)

#Prepare test dataset

df_test = feature_engineering.add_rare_titles(df_test)
df_test = feature_engineering.add_cabin(df_test)
df_test = feature_engineering.add_family_size(df_test)

df_test = preprocessing.drop_columns(df_test)
df_test = preprocessing.one_hot_encode(df_test)

df_test = df_test.reindex(columns=df_train.drop('Survived', axis=1).columns, fill_value=0)
assert all(df_test.columns == df_train.drop(columns=['Survived']).columns)

df_test = preprocessing.apply_imputer(df_test, imputer)
df_test = preprocessing.apply_scaler(df_test, scaler)



print(df_test)

#Predict test dataset using model

y_pred_prob = logisticModel.predict_proba(df_test)[:, 1]
y_pred = (y_pred_prob >= 0.5).astype(int)

df_submission = pd.DataFrame({
    'PassengerId': df_test_raw['PassengerId'],
    'Survived': y_pred
})

df_submission.to_csv('submission.csv', index=False)

#Random Forest Model

rf_model = modeling.train_random_forest_model(df_train)
y_pred_rf = rf_model.predict(df_test)
df_submission_rf = pd.DataFrame({
    'PassengerId': df_test_raw['PassengerId'],  
    'Survived': y_pred_rf
})

df_submission_rf.to_csv('submission_rf.csv', index=False)

#XGBoost






