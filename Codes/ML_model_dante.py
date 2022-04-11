import numpy as np
import pandas as pd
from sklearn.svm import SVR
from functions_ML import *

anticipation = 2
target_variable = ['Costa', '8148_1']

precipitation_list = [1,2,3,4,5,6]
for i in range(anticipation-1): precipitation_list.remove(i+1)

train_variables = [['Costa', '8148_1', [0+anticipation,1+anticipation]],
                  ['Molteno', '9084_1', [0+anticipation,1+anticipation]],
                  ['Caslino', '8124_1', [0+anticipation,1+anticipation]],
                  ['Molteno', '9106_4', precipitation_list],
                  ['Caslino', '8122_4', precipitation_list],
                  ['Canzo', '2614_4', precipitation_list],
                  ['Erba', '5870_4', precipitation_list],
                  ['Lambrugo', '8197_4', precipitation_list]]

max_dt = max(precipitation_list)+1
X_train, X_test, Y_train, Y_test = organize_input_data(train_variables, target_variable, max_dt, starting_point = 1000, training_size=10000, testing_size =1000)

svr_lin = SVR(kernel='linear', C=1, epsilon=0.1)
svr_lin.fit(X_train, Y_train)
Y_predicted = svr_lin.predict(X_test)

np.savetxt('../Results/coefficients_dante_'+str(anticipation)+'hr.txt', svr_lin.coef_)
np.savetxt('../Results/intercept_dante_'+str(anticipation)+'hr.txt', svr_lin.intercept_)

plot_hydrograph(Y_predicted, Y_test)
plot_scatter(Y_predicted, Y_test, 1, 100)