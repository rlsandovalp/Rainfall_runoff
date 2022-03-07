import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import os          

place = 'Nibionno'

for file in os.listdir('../raw_data/'+place+'/'):
    if file.endswith("5882_4.csv"):

        source = '../raw_data/'+place+'/'+file
        output = '../processed_data/'+place+'/'+file
        threshold = 15

        Y = pd.read_csv(source)
        X_raw = pd.read_csv(source).values[:,2]

        X = X_raw.reshape(-1,1)
        X = np.where(X<-100, -999, X)
        imputer = KNNImputer(n_neighbors=2, weights="uniform", missing_values=-999)
        X1 = imputer.fit_transform(X)

        for i in range(1,X1.size-1):
            if (abs(X1[i]-X1[i-1])>threshold) and (abs(X1[i]-X1[i+1])>threshold):
                X1[i] = (X1[i-1]+X1[i+1])/2

        Y['value'] = X1
        # Y.drop(' Medio', axis = 1, inplace = True)
        Y.drop('Valore Cumulato', axis = 1, inplace = True)
        
        Y.to_csv(output, index=False)

        print(file)
        print('Mean: ', X1.mean())
        print('Std: ', np.std(X1))
        print('Min: ', X1.min())
        print('Max: ', X1.max())
        print('-'*50)
