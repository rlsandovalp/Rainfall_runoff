import pandas as pd
import os

station = 'Molteno/'
variables = ['9084_1', '9106_4', '9017_1', '11020_1']


folder = '../processed_data/'+station

for variable in variables:
    df = pd.DataFrame(columns = ['Id Sensore','Data-Ora', 'value'])
    for file in os.listdir(folder):
        if file.endswith(variable+'.csv'):
            df1 = pd.read_csv('../processed_data/'+station+file)
            df = pd.merge(df, df1, how = 'outer')

    df['Data-Ora'] =pd.to_datetime(df['Data-Ora'], infer_datetime_format=True)
    df.sort_values(by='Data-Ora', inplace=True)
    df.to_csv('../joined_data/'+station+variable+'.csv', index=False)
