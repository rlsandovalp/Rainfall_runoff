import numpy as np
import pandas as pd
import tensorflow as tf
from functions_ML import *


window = 8
anticipation = 1



training_variables = [['Molteno', '9084_1', 'Q [cms]'],
                  ['Molteno', '9106_4', 'P [mm]'],
                  ['Molteno', '9017_1', 'T [C]'],
                  ['Molteno', '11020_1', 'RH [%]']]

target_variable = ['Molteno', '9084_1', 'Q [cms]']

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

plt.figure(figsize = (10,6))

plt.subplot(3,3,1)
plt.plot(Yp, 'r-', label = 'LSTM')
plt.plot(Y_test, 'b-', label = 'Measured')
plt.ylabel('Flowrate [cms]')
plt.legend()
plt.text(0.03, 10, 'MAPE = ' + str('%.2f' % mape(Yp, Y_test)) + '%')
plt.text(0.03, 5, 'NSE = ' + str('%.2f' % nse(Yp, Y_test)))

plt.subplot(3,3,2)
plt.plot(X_test_2[:,0], label = 'Level [cm]')
plt.xlabel('Time [hours]')
plt.ylabel('q [cm]')
plt.legend()

plt.subplot(3,3,3)
plt.bar(np.linspace(0,1000,1000), X_test_2[:,1], width = 0.8)
plt.xlabel('Time [hours]')
plt.ylabel('P [mm]')

plt.subplot(3,3,4)
plt.plot(X_test_2[:,2])
plt.xlabel('Time [hours]')
plt.ylabel('T [C]')

plt.subplot(3,3,5)
plt.plot(X_test_2[:,3])
plt.xlabel('Time [hours]')
plt.ylabel('RH [%]')

plt.subplot(3,3,6)
plt.plot(Y_test-Yp, 'r-', label = 'LSTM')
plt.ylabel('Flowrate difference [cms]')
plt.xlabel('Time [hours]')
plt.legend()

plt.show()
