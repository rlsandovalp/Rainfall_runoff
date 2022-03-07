import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import os


def flowrate(X_raw):
    X_new = np.zeros(np.size(X_raw))
    for i in range(X_raw.size):
        if X_raw[i] == -999:
            X_new[i] = -999
        elif X_raw[i] < -30:
            X_new[i] = -999
        elif (X_raw[i] > -20) and (X_raw[i] < -10):
            X_raw[i] = -10
            X_new[i] = 27.032*(X_raw[i]/100+0.212)**1.633
        else:
            X_new[i] = 27.032*(X_raw[i]/100+0.212)**1.633
    return X_new            

place = 'Lesmo'

for file in os.listdir('../raw_data/'+place+'/'):
    if file.endswith("8120_1.csv"):

        source = '../raw_data/'+place+'/'+file
        output = '../processed_data/'+place+'/'+file
        threshold = 15
        print(file)
        Y = pd.read_csv(source)
        X_raw = pd.read_csv(source).values[:,2]

        X = flowrate(X_raw).reshape(-1,1)
        X = np.where(X<0, -999, X)
        imputer = KNNImputer(n_neighbors=2, weights="uniform", missing_values=-999)
        X1 = imputer.fit_transform(X)

        for i in range(1,X1.size-1):
            if (abs(X1[i]-X1[i-1])>threshold) and (abs(X1[i]-X1[i+1])>threshold):
                X1[i] = (X1[i-1]+X1[i+1])/2

        Y['value'] = X1
        Y.drop(' Medio', axis = 1, inplace = True)
        Y.to_csv(output, index=False)

        print('Mean: ', X1.mean())
        print('Std: ', np.std(X1))
        print('Min: ', X1.min())
        print('Max: ', X1.max())
        print('-'*50)
