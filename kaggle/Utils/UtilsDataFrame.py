import pandas as pd


def delete_column(df, name):
    df.drop([name], axis=1, inplace=True, errors='ignore')


def print_na_count(df):
    for column in df:
        total_na = df[column].isnull().sum()
        if total_na > 0:
            print("%s -> %s" % (column, total_na))
