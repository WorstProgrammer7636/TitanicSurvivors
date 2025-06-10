from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.preprocessing import StandardScaler


def drop_columns_test(df):
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    return df


def impute_test(df_train, df):
    age_imputer = SimpleImputer(strategy='median')
    df[['Age']] = age_imputer.fit_transform(df[['Age']])

    fare_imputer = SimpleImputer(strategy='median')
    fare_imputer.fit(df_train[['Fare']])
    df[['Fare']] = fare_imputer.transform(df[['Fare']])


    return df

def scale_test(df):
    scaler = StandardScaler()
    numeric_explanatory = ['Age', 'SibSp', 'Parch', 'Fare']
    df[numeric_explanatory] = scaler.fit_transform(df[numeric_explanatory])
    return df