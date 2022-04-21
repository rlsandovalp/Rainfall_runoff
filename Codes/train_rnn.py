import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time
from functions_ML import *

#############################################
##########  DANTE  ########################
#############################################

training_variables = [['Costa', '8148_1', 'h [cm]'],
                  ['Molteno', '9084_1', 'h [cm]'],
                  ['Caslino', '8124_1', 'h [cm]'],
                  ['Molteno', '9106_4', 'P [mm]'],
                  ['Caslino', '8122_4', 'P [mm]'],
                  ['Canzo', '2614_4', 'P [mm]'],
                  ['Erba', '5870_4', 'P [mm]'],
                  ['Lambrugo', '8197_4', 'P [mm]']]

target_variable =  ['Costa', '8148_1', 'h [cm]']

########################################
            # DEFINE THE MODEL
########################################

window = 80
anticipation = 1

def make_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(50, input_shape = (window, len(training_variables))))
    model.add(tf.keras.layers.Dropout(0.2))
    # model.add(tf.keras.layers.LSTM(30))
    # model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1))
    return model

########################################
            # READ THE DATA
########################################

starting_point = 0
training_size = 55000

X_train = np.zeros((training_size, len(training_variables)))
Y_train = np.zeros(training_size)

for num, variable in enumerate(training_variables):
    X_train[:,num] = pd.read_csv('../joined_data/'+variable[0]+'/'+variable[1]+'.csv').values[starting_point:starting_point+training_size,-1]

Y_train = pd.read_csv('../joined_data/'+target_variable[0]+'/'+target_variable[1]+'.csv').values[starting_point:starting_point+training_size,-1]

X_train_min = np.min(X_train, axis = 0)
X_train_max = np.max(X_train, axis = 0)
X_train = (X_train-X_train_min)/(X_train_max-X_train_min)-0.5

Y_train_min = np.min(Y_train, axis = 0)
Y_train_max = np.max(Y_train, axis = 0)
Y_train = (Y_train-Y_train_min)/(Y_train_max-Y_train_min)-0.5

np.savetxt('../Models/X_lim_model_wf_ant'+str(anticipation)+'_'+target_variable[0]+'.txt', [X_train_min, X_train_max])
np.savetxt('../Models/Y_lim_model_wf_ant'+str(anticipation)+'_'+target_variable[0]+'.txt', [Y_train_min, Y_train_max])

X = []
Y = []

for i in range (window+anticipation,training_size):
    if X_train[i,0]> -0.15:
        X.append(X_train[i-window-anticipation:i-anticipation,:])
        Y.append(Y_train[i-1])

X_train, Y_train = np.array(X), np.array(Y)

Y_train = np.reshape(Y_train, (len(Y_train), 1))

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.25)
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
callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 0, patience = 10)

########################################
            # FIT THE MODEL
########################################
t0 = time.time()
history = model.fit(X_train, Y_train, batch_size = 3000, epochs = 200, validation_data=(X_test, Y_test), callbacks = [callback])
t1 = time.time()
print('Runtime: %.2f s' % (t1-t0))

plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'validation')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

model.save('../Models/rnn_model_wf_ant'+str(anticipation)+'_'+target_variable[0]+'.h5')