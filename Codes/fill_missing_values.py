import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import os

station = 'Molteno/'
variables = ['9084_1', '9106_4', '9017_1', '11020_1']

folder = '../raw_data/'+station

imputer = KNNImputer(n_neighbors=2, weights="uniform", missing_values=-999)

for variable in variables:
    for file in os.listdir(folder):
        if file.endswith(variable):
            print('-'*50)
            source = folder+file
            output = '../processed_data/'+station+file
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