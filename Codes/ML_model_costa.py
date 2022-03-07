import numpy as np
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt

##################### READ DATA

# plt.style.use(['science','nature'])

target_variable = ['Costa', '8148_1']

train_variables = [['Costa', '8148_1', [1,2]],
                  ['Molteno', '9084_1', [1,2]],
                  ['Caslino', '8124_1', [1,2]],
                  ['Molteno', '9106_4', [2,3,4,5,6]],
                  ['Caslino', '8122_4', [2,3,4,5,6]],
                  ['Canzo', '2614_4', [2,3,4,5,6]],
                  ['Erba', '5870_4', [2,3,4,5,6]],
                  ['Lambrugo', '8197_4', [2,3,4,5,6]]]

starting_point = 1000
training_size = 1000
testing_size = 100
maxdt = 7

X_train = []
Y_train = []
X_test = []
Y_test = []

for station in train_variables:
    for dt in station[2]:
        X_train.append(pd.read_csv('../joined_data/'+station[0]+'/'+station[1]+'.csv').values[starting_point+(maxdt-dt):starting_point+training_size+(maxdt-dt),-1])
        X_test.append(pd.read_csv('../joined_data/'+station[0]+'/'+station[1]+'.csv').values[starting_point+training_size+(maxdt-dt):starting_point+training_size+testing_size+(maxdt-dt),-1])

X_train = np.transpose(np.array(X_train))
X_test = np.transpose(np.array(X_test))


Y_train = pd.read_csv('../joined_data/'+target_variable[0]+'/'+target_variable[1]+'.csv').values[starting_point+maxdt:starting_point+training_size+maxdt,-1]
Y_test =  pd.read_csv('../joined_data/'+target_variable[0]+'/'+target_variable[1]+'.csv').values[starting_point+training_size+maxdt:starting_point+training_size+testing_size+maxdt,-1]

svr_lin = SVR(kernel='linear', C=100)
svr_lin.fit(X_train, Y_train)

Y_predicted = svr_lin.predict(X_test)



ind_Y_test_10 = np.argpartition(Y_test, -int(testing_size*0.1))[-int(testing_size*0.1):]
Y_test_10 = Y_test[ind_Y_test_10]
Y_predicted_10 = Y_predicted[ind_Y_test_10]

print('MAPE all:', (abs(Y_test-Y_predicted)/Y_test).mean())
print('MAPE 10:', (abs(Y_test_10-Y_predicted_10)/Y_test_10).mean())

fig = plt.figure(dpi = 300)
ax = fig.add_subplot(111)
ax.plot(Y_predicted, color = 'red', linewidth = 1, label = 'Y predicted')
ax.plot(Y_test, '.', label = 'Y observed', markersize = 3)
ax.set_title('Flowrate predicted and observed')
ax.set_xlabel('Time [hours]')
ax.set_ylabel('Flowrate [cms]')
plt.legend()
plt.show()


x = np.linspace(Y_test.min(),Y_test.max(),100)
y = x

fig = plt.figure(dpi = 300)
ax = fig.add_subplot(111)
ax.scatter(Y_test, Y_predicted, s = 1, color = 'black')
ax.plot(x, y, 'g')
ax.set_title('Scatterplot observed vs predicted flowrates')
ax.set_xlabel('Observed [cms]')
ax.set_ylabel('Predicted [cms]')
plt.xscale('log')
plt.yscale('log')
plt.show()

