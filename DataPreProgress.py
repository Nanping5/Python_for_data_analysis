import pandas as pd
import numpy as np

def transform_to_binary(df, col_name):
    # yes->1, no->0, unknown->NaN
    df.loc[:, col_name] = df[col_name].map({'yes': 1, 'no': 0, 'True': 1, 'False': 0})
    df.loc[:, col_name] = df[col_name].replace('unknown', np.nan)
    return df

def dummy_variables(df, columns):
    for col in columns:
        if col in df.columns:
            # 将unknown替换为NaN
            df.loc[:, col] = df[col].replace('unknown', np.nan)
            df.loc[:, col] = df[col].replace('True', '1')
            df.loc[:, col] = df[col].replace('False', '0')
            #哑编码
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
            #将编码列添加到DataFrame 中
            df = df.drop(col, axis=1)
            df = pd.concat([df, dummies], axis=1)

    return df

def numberical_category_data(df):
    #数值化数据
    numericalcols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    for col in numericalcols:
        if col in df.columns:
            df.loc[:, col] = df[col].replace('unknown', np.nan)
            df.loc[:, col] = pd.to_numeric(df[col])
    #二元分类数据
    categoricaltobinarycols = ['default', 'housing', 'loan', 'y']
    for col in categoricaltobinarycols:
        df = transform_to_binary(df, col)
   
    return df

def remove_duplicate_values(df):
    return df.drop_duplicates()

def drop_unknown_values(df, columns):
    for col in columns:
        df = df[df[col] != 'unknown']
    return df

def modify_marital_column(df):
    if 'marital' in df.columns:
        df.loc[:, 'marital'] = df['marital'].map({'single': 0, 'married': 1, 'divorced': 2})
    else:
        print("Column 'marital' not found in DataFrame")
    return df

def data_processing(df):
    columns_to_check = ['poutcome', 'contact', 'education', 'marital']
    df = drop_unknown_values(df, columns_to_check)
    df = modify_marital_column(df)
    df = numberical_category_data(df)
    df = remove_duplicate_values(df)
    return df

def write_to_csv(df, filename):
    data_processing(df)
    df.to_csv(filename, sep=';', index=False)
