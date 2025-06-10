import pandas as pd

def extract_title(name):
    return name.split(",")[1].split(".")[0].strip()


def simplify_title(title):
    if title in ['Mr', 'Miss', 'Mrs', 'Master']:
        return title
    else:
        return 'Rare'

def add_rare_titles(df):
    df['Title'] = df['Name'].apply(extract_title)
    df['Title'] = df['Title'].apply(simplify_title)
    return df

def add_cabin(df):
    df['Deck'] = df['Cabin'].str[0]
    return df

def add_family_size(df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    return df