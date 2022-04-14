import numpy as np
import pandas as pd
import tensorflow as tf
import time
from functions_ML import *

########################################
            # DEFINE THE VARIABLES
########################################

training_variables = [['Costa', '8148_1'],
                    ['Molteno', '9084_1'],
                    ['Caslino', '8124_1'],
                    ['Molteno', '9106_4'],
                    ['Caslino', '8122_4'],
                    ['Canzo', '2614_4'],
                    ['Erba', '5870_4'],
                    ['Lambrugo', '8197_4']]

target_variable = ['Costa', '8148_1']

########################################
            # DEFINE THE MODEL
########################################

window = 8
anticipation = 2

def make_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(50, return_sequences = True, input_shape = (window, len(training_variables))))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(50))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1))
    return model

########################################
            # READ THE DATA
########################################

starting_point = 0
training_size = 50000

X_train = np.zeros((training_size, len(training_variables)))
Y_train = np.zeros(training_size)

for num, variable in enumerate(training_variables):
    X_train[:,num] = pd.read_csv('../joined_data/'+variable[0]+'/'+variable[1]+'.csv').values[starting_point:starting_point+training_size,-1]

Y_train = pd.read_csv('../joined_data/'+target_variable[0]+'/'+target_variable[1]+'.csv').values[starting_point:starting_point+training_size,-1]

X = []
Y = []

for i in range (window+anticipation-1,training_size):
    X.append(X_train[i-window-anticipation+1:i-anticipation+1,:])
    Y.append(Y_train[i])

X_train, Y_train = np.array(X), np.array(Y)

Y_train = np.reshape(Y_train, (len(Y_train), 1))
########################################
            # CREATE THE MODEL
########################################

s = tf.keras.backend.clear_session()
model = make_model()
model.summary()

# ########################################
#             # COMPILE MODEL
# ########################################

model.compile(loss = 'mean_absolute_error', optimizer = 'adam', metrics = ['accuracy'])
callback = tf.keras.callbacks.EarlyStopping(monitor = 'loss', mode = 'min', verbose = 1, patience = 20)

########################################
            # FIT THE MODEL
########################################
t0 = time.time()
history = model.fit(X_train, Y_train, batch_size = 1000, epochs = 100, callbacks = [callback])
t1 = time.time()
print('Runtime: %.2f s' % (t1-t0))

plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

model.save('rnn_model_wf_ant2.h5')