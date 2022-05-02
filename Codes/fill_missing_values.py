import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import os

variables = [['Lesmo', '8120_1', 'h [cm]'],
                  ['Costa', '8148_1', 'h [cm]'],
                  ['Molteno', '9084_1', 'h [cm]'],
                #   ['Molteno', '9017_1', 'T [C]'],
                  ['Molteno', '9106_4', 'P [mm]'],
                #   ['Molteno', '11020_1', 'HR [%]'],
                  ['Caslino', '8124_1', 'h [cm]'],
                #   ['Caslino', '8123_1', 'T [C]'],
                  ['Caslino', '8122_4', 'P [mm]'],
                  ['Canzo', '2614_4', 'P [mm]'],
                #   ['Erba', '5871_1', 'T [C]'],
                  ['Erba', '5870_4', 'P [mm]'],
                #   ['Erba', '6163_1', 'RH [%]'],
                #   ['Lambrugo', '8198_1', 'T [C]'],
                  ['Lambrugo', '8197_4', 'P [mm]'],
                  ['Casatenovo', '2385_4', 'P [mm]']]

imputer = KNNImputer(n_neighbors=2, weights="uniform", missing_values=-999)

for variable in variables:
    folder = '../raw_data/'+variable[0]+'/'
    code_variable = variable[1]
    for file in os.listdir(folder):
        if file.endswith(code_variable+'.csv'):
            print('-'*50)
            source = folder+file
            output = '../processed_data/'+variable[0]+'/'+file
            print(file)

            Y = pd.read_csv(source)
            X_raw = pd.read_csv(source).values[:,2]

            X = X_raw.reshape(-1,1)
            X_fill = imputer.fit_transform(X)

            Y['value'] = X_fill
            Y.drop(Y.keys()[2], axis = 1, inplace = True)
            Y.to_csv(output, index=False)

            print('Mean: ', X_fill.mean())
            print('Std: ', np.std(X_fill))
            print('Min: ', X_fill.min())
            print('Max: ', X_fill.max())