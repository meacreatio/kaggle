import pandas as pd


def delete_column(df, name):
    df.drop([name], axis=1, inplace=True, errors='ignore')
