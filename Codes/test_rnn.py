import numpy as np
import pandas as pd
import tensorflow as tf
from functions_ML import *


window = 8
anticipation = 1

training_variables = [['Costa', '8148_1'],
                    ['Molteno', '9084_1'],
                    ['Caslino', '8124_1'],
                    ['Molteno', '9106_4'],
                    ['Caslino', '8122_4'],
                    ['Canzo', '2614_4'],
                    ['Erba', '5870_4'],
                    ['Lambrugo', '8197_4']]

target_variable = ['Costa', '8148_1']

model = model = tf.keras.models.load_model('rnn_model_wf_ant1.h5')

starting_point = 0
training_size = 57000
testing_size = 1000

X_test = np.zeros((testing_size, len(training_variables)))
Y_test = np.zeros(testing_size)

for num, variable in enumerate(training_variables):
    X_test[:,num] = pd.read_csv('../joined_data/'+variable[0]+'/'+variable[1]+'.csv').values[starting_point+training_size:starting_point+training_size+testing_size,-1]

Y_test = pd.read_csv('../joined_data/'+target_variable[0]+'/'+target_variable[1]+'.csv').values[starting_point+training_size:starting_point+training_size+testing_size,-1]

X = []
Y = []

for i in range (window+anticipation,testing_size):
    X.append(X_test[i-window-anticipation:i-anticipation,:])
    Y.append(Y_test[i-1])


## TUTTO E UN CASINO!!
X_test, Y_test = np.array(X), np.array(Y)

Yp = model.predict(X_test)
Yp = np.reshape(Yp, -1)

plt.figure(figsize = (10,6))
plt.plot(Y_test-Yp, 'r-', label = 'LSTM')
plt.ylabel('Flowrate difference [cms]')
plt.xlabel('Time [hours]')
plt.legend()
plt.show()


plt.figure(figsize = (10,6))
plt.subplot(3,3,1)
plt.plot(Yp, 'r-', label = 'LSTM')
plt.plot(Y_test, 'b-', label = 'Measured')
plt.ylabel('Flowrate [cms]')
plt.legend()
plt.text(0.03, 10, 'MAPE = ' + str('%.2f' % mape(Yp, Y_test)) + '%')
plt.text(0.03, 5, 'NSE = ' + str('%.2f' % nse(Yp, Y_test)))
plt.subplot(3,3,2)
plt.plot(X_test[:,-1,0], label = 'Precipitation [mm]')
plt.xlabel('Time [hours]')
plt.ylabel('q [mm]')
plt.legend()
plt.subplot(3,3,3)
plt.plot(X_test[:,-1,1], label = 'Precipitation [mm]')
plt.xlabel('Time [hours]')
plt.ylabel('q [mm]')
plt.legend()
plt.subplot(3,3,4)
plt.plot(X_test[:,-1,2], label = 'Precipitation [mm]')
plt.xlabel('Time [hours]')
plt.ylabel('q [mm]')
plt.legend()
plt.subplot(3,3,5)
plt.plot(X_test[:,-1,3], label = 'Precipitation [mm]')
plt.xlabel('Time [hours]')
plt.ylabel('Precipitation [mm]')
plt.legend()
plt.subplot(3,3,6)
plt.plot(X_test[:,-1,4], label = 'Precipitation [mm]')
plt.xlabel('Time [hours]')
plt.ylabel('Precipitation [mm]')
plt.legend()
plt.subplot(3,3,7)
plt.plot(X_test[:,-1,5], label = 'Precipitation [mm]')
plt.xlabel('Time [hours]')
plt.ylabel('Precipitation [mm]')
plt.legend()
plt.subplot(3,3,8)
plt.plot(X_test[:,-1,6], label = 'Precipitation [mm]')
plt.xlabel('Time [hours]')
plt.ylabel('Precipitation [mm]')
plt.legend()
plt.subplot(3,3,9)
plt.plot(X_test[:,-1,7], label = 'Precipitation [mm]')
plt.xlabel('Time [hours]')
plt.ylabel('Precipitation [mm]')
plt.legend()
plt.show()
