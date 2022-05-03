import numpy as np
import pandas as pd
import tensorflow as tf
from functions_ML import *


window = 24
anticipation = 1

# #############################################
# ##########  MOLTENO  ########################
# #############################################

# training_variables = [['Molteno', '9084_1', 'h [cm]'],
#                   ['Molteno', '9106_4', 'P [mm]'],
#                   ['Molteno', '9017_1', 'T [C]'],
#                   ['Molteno', '11020_1', 'RH [%]']]

# target_variable = ['Molteno', '9084_1', 'Q [cms]']

#############################################
##########  DANTE  ########################
#############################################

target_variable = ['Lesmo', '8120_1']

training_variables = [['Lesmo', '8120_1', 'h [cm]'],
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

model = model = tf.keras.models.load_model('../Models/rnn_model_wf_ant'+str(anticipation)+'_'+target_variable[0]+'.h5')

starting_point = 0
training_size = 57000
testing_size = 1000

X_test = np.zeros((testing_size, len(training_variables)))

limits_normalization = np.loadtxt('../Models/X_lim_model_wf_ant'+str(anticipation)+'_'+target_variable[0]+'.txt')
X_train_min = limits_normalization[0,:]
X_train_max = limits_normalization[1,:]

limits_normalization = np.loadtxt('../Models/Y_lim_model_wf_ant'+str(anticipation)+'_'+target_variable[0]+'.txt')
Y_train_min = limits_normalization[0]
Y_train_max = limits_normalization[1]

for num, variable in enumerate(training_variables):
    X_test[:,num] = pd.read_csv('../joined_data/'+variable[0]+'/'+variable[1]+'.csv').values[starting_point+training_size:starting_point+training_size+testing_size,-1]

Y_test = pd.read_csv('../joined_data/'+target_variable[0]+'/'+target_variable[1]+'.csv').values[starting_point+training_size:starting_point+training_size+testing_size,-1]

X_test_2 = X_test
Y_test_2 = Y_test

X_test = (X_test-X_train_min)/(X_train_max-X_train_min)-0.5
Y_test = (Y_test-Y_train_min)/(Y_train_max-Y_train_min)-0.5

X = []
Y = []

for i in range (window+anticipation,testing_size+1):
    X.append(X_test[i-window-anticipation:i-anticipation,:])
    Y.append(Y_test[i-1])

## TUTTO E UN CASINO!!
X_test, Y_test = np.array(X), np.array(Y)

Yp = model.predict(X_test)
Yp = (Yp+0.5)*(Y_train_max-Y_train_min)+Y_train_min
Yp = np.concatenate((np.zeros(window),np.reshape(Yp, -1)))
Y_test = (Y_test+0.5)*(Y_train_max-Y_train_min)+Y_train_min
Y_test = np.concatenate((np.zeros(window),Y_test))

plt.figure(figsize = (15,10))

total = len(training_variables)+2

if total < 10:
    a = 3
    b = 3
elif total < 12:
    a = 3
    b = 4
else:
    a = 4
    b = 4

plt.subplot(a,b,1)
plt.plot(Yp, '.', label = 'LSTM')
plt.plot(Y_test, '.', label = 'Measured')
plt.ylabel('h level [cm]')
plt.legend()
# plt.text(0.03, 10, 'MAPE = ' + str('%.2f' % mape(Yp, Y_test)) + '%')
# plt.text(0.03, 5, 'NSE = ' + str('%.2f' % nse(Yp, Y_test)))

A = (Y_test-Yp)/Y_test
A[Y_test<60] = 0

plt.subplot(a,b,2)
plt.plot(A, 'r-', label = 'Y_pr - Y_obs')
plt.ylabel('h [cm]')
plt.xlabel('Time [hours]')
plt.legend()

for n_var, variable in enumerate(training_variables):
    plt.subplot(a,b,n_var+3)
    plt.plot(X_test_2[:,n_var], label = training_variables[n_var][0] + ' ' + training_variables[n_var][1])
    plt.xlabel('Time [hours]')
    plt.ylabel(training_variables[n_var][2])
    plt.legend()

plt.show()
