import numpy as np
import pandas as pd
import tensorflow as tf
import time
from functions_ML import *

########################################
            # DEFINE THE VARIABLES
########################################

training_variables = [['Molteno', '9106_4'],
                      ['Molteno', '9084_1']]

target_variable = ['Molteno', '9084_1']

########################################
            # DEFINE THE MODEL
########################################

days_corr = 5
window = days_corr*24

def make_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(30, return_sequences = True, input_shape = (window, len(training_variables))))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(12, return_sequences = True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1))
    return model

########################################
            # READ THE DATA
########################################

starting_point = 0
training_size = 10000
testing_size = 1000

X_train = np.zeros((training_size, len(training_variables)))
Y_train = np.zeros(training_size)
X_test = np.zeros((testing_size, len(training_variables)))
Y_test = np.zeros(testing_size)

for num, variable in enumerate(training_variables):
    X_train[:,num] = pd.read_csv('../joined_data/'+variable[0]+'/'+variable[1]+'.csv').values[starting_point:starting_point+training_size,-1]

Y_train = pd.read_csv('../joined_data/'+target_variable[0]+'/'+target_variable[1]+'.csv').values[starting_point:starting_point+training_size,-1]

for num, variable in enumerate(training_variables):
    X_test[:,num] = pd.read_csv('../joined_data/'+variable[0]+'/'+variable[1]+'.csv').values[starting_point+training_size:starting_point+training_size+testing_size,-1]

Y_test = pd.read_csv('../joined_data/'+target_variable[0]+'/'+target_variable[1]+'.csv').values[starting_point+training_size:starting_point+training_size+testing_size,-1]

X = []
Y = []

for i in range (window,training_size):
    X.append(X_train[i-window:i,:])
    Y.append(Y_train[i])

X_train, Y_train = np.array(X), np.array(Y)

########################################
            # CREATE THE MODEL
########################################

s = tf.keras.backend.clear_session()
model = make_model()
model.summary()

# ########################################
#             # COMPILE MODEL
# ########################################

model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])
callback = tf.keras.callbacks.EarlyStopping(monitor = 'loss', mode = 'min', verbose = 1, patience = 10)

########################################
            # FIT THE MODEL
########################################
t0 = time.time()
history = model.fit(X_train, Y_train, batch_size = 500, epochs = 10, callbacks = [callback])
t1 = time.time()
print('Runtime: %.2f s' % (t1-t0))

plt.plot(history.history['loss']) 
plt.title('Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.show()

model.save('rnn_model.h5')