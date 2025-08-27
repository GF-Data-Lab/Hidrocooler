import pandas as pd

def process_estimation(data):
    data_new = data.drop_duplicates(subset=['IX_LOT','FECHA'])
    df_grouped = (
        data_new
        .groupby(['CAMPO', 'FECHA'], as_index=False)['RECEPCIÓN']
        .sum()
    )
    return df_grouped


def process_estimation_gruesa(data):
    df_grouped = (
        data
        .groupby(['CAMPO', 'FECHA'], as_index=False)['CANTIDAD']
        .sum()
    )
    return df_grouped
