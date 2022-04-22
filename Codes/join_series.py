import pandas as pd
import os

variables = [['Costa', '8148_1', 'h [cm]'],
                  ['Molteno', '9084_1', 'h [cm]'],
                  ['Molteno', '9017_1', 'T [C]'],
                  ['Molteno', '9106_4', 'P [mm]'],
                  ['Molteno', '11020_1', 'HR [%]'],
                  ['Caslino', '8122_4', 'P [mm]'],
                  ['Caslino', '8123_1', 'T [C]'],
                  ['Caslino', '8124_1', 'h [cm]'],
                  ['Canzo', '2614_4', 'P [mm]'],
                  ['Erba', '6163_1', 'RH [%]'],
                  ['Erba', '5871_1', 'T [C]'],
                  ['Erba', '5870_4', 'P [mm]'],
                  ['Lambrugo', '8198_1', 'T [C]'],
                  ['Lambrugo', '8197_4', 'P [mm]']]

for variable in variables:
    folder = '../processed_data/'+variable[0]
    df = pd.DataFrame(columns = ['Id Sensore','Data-Ora', 'value'])
    for file in os.listdir(folder):
        if file.endswith(variable[1]+'.csv'):
            df1 = pd.read_csv('../processed_data/'+variable[0]+'/'+file)
            df = pd.merge(df, df1, how = 'outer')

    df['Data-Ora'] =pd.to_datetime(df['Data-Ora'], infer_datetime_format=True)
    df.sort_values(by='Data-Ora', inplace=True)
    df.to_csv('../joined_data/'+variable[0]+'/'+variable[1]+'.csv', index=False)
