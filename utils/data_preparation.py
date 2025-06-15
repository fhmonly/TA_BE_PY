import pandas as pd
import numpy as np
from io import StringIO

def read_csv_string_to_df(csv_string):
    return pd.read_csv(StringIO(csv_string))

def df_group_by_interval(df, date_col, value_col, freq):
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    start = df[date_col].min().to_period(freq)
    # end = pd.Timestamp.today().to_period(freq)
    end = df[date_col].max().to_period(freq) #end date using last date on record

    df['period'] = df[date_col].dt.to_period(freq)
    grouped = df.groupby('period')[value_col].sum()
    full_index = pd.period_range(start, end, freq=freq)
    grouped_full = grouped.reindex(full_index, fill_value=0)

    # Ubah PeriodIndex jadi DatetimeIndex
    grouped_full.index = grouped_full.index.to_timestamp()

    # Convert Series jadi DataFrame biar bisa akses kolom
    return grouped_full.to_frame(name=value_col) 
    # return grouped.to_frame(name=value_col) 

def df_group_by_interval_interpolate(df, date_col, value_col, freq):
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    start = df[date_col].min().to_period(freq)
    end = pd.Timestamp.today().to_period(freq)

    df['period'] = df[date_col].dt.to_period(freq)
    grouped = df.groupby('period')[value_col].sum()
    full_index = pd.period_range(start, end, freq=freq)
    grouped_full = grouped.reindex(full_index, fill_value=0)

    # Convert PeriodIndex ke DatetimeIndex
    grouped_full.index = grouped_full.index.to_timestamp()

    # Interpolasi linear untuk mengisi 0 yang ada di data
    # Pertama, ubah 0 jadi NaN supaya interpolasi bisa jalan
    grouped_full_replaced = grouped_full.replace(0, np.nan)

    # Lakukan interpolasi berdasarkan waktu index
    grouped_interpolated = grouped_full_replaced.interpolate(method='time')

    # Optional: kalau mau fill sisa NaN di ujung dengan 0 lagi (atau pakai forward/backward fill)
    grouped_interpolated = grouped_interpolated.fillna(0)

    return grouped_interpolated.to_frame(name=value_col)
