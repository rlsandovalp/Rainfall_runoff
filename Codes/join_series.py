import pandas as pd
import os

station = 'Nibionno/'
code = '5882_4.csv'

df = pd.DataFrame(columns = ['Id Sensore','Data-Ora', 'value'])

for file in os.listdir('../processed_data/'+station):
    if file.endswith(code):
        df1 = pd.read_csv('../processed_data/'+station+file)
        df = pd.merge(df, df1, how = 'outer')

df['Data-Ora'] =pd.to_datetime(df['Data-Ora'], infer_datetime_format=True)
df.sort_values(by='Data-Ora', inplace=True)
df.to_csv('../joined_data/'+station+code, index=False)
