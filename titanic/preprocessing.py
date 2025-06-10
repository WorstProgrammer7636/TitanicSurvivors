import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def drop_columns(df):
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    return df

    
def one_hot_encode(df):
    columns_to_encode = ['Sex', 'Embarked', 'Pclass', 'Title', 'Deck']
    df = pd.get_dummies(df, columns = columns_to_encode, drop_first=True, dtype=int)
    return df

def fit_imputer(df):
    imputer = SimpleImputer(strategy='median')
    imputer.fit(df[['Age', 'Fare']])
    return imputer

def apply_imputer(df, imputer):
    df[['Age', 'Fare']] = imputer.transform(df[['Age', 'Fare']])
    return df

def fit_scaler(df):
    scaler = StandardScaler()
    numeric_explanatory = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']
    scaler.fit(df[numeric_explanatory])
    return scaler

def apply_scaler(df, scaler):
    numeric_explanatory = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']
    df[numeric_explanatory] = scaler.transform(df[numeric_explanatory])
    return df


