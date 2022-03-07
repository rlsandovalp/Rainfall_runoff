import numpy as np
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt

##################### READ DATA

plt.style.use(['science','nature'])

station_X = 'Molteno'
variable_X = '9106_4'
station_Y = 'Molteno'
variable_Y = '9084_1'

starting_point = 1000
training_size = 10000
testing_size = 1000

q_1_train = pd.read_csv('../joined_data/'+station_Y+'/'+variable_Y+'.csv').values[starting_point+5:starting_point+training_size+5,-1]
q_2_train = pd.read_csv('../joined_data/'+station_Y+'/'+variable_Y+'.csv').values[starting_point+4:starting_point+training_size+4,-1]
p_1_train = pd.read_csv('../joined_data/'+station_X+'/'+variable_X+'.csv').values[starting_point+1:starting_point+training_size+1,-1]
p_2_train = pd.read_csv('../joined_data/'+station_X+'/'+variable_X+'.csv').values[starting_point+2:starting_point+training_size+2,-1]
p_3_train = pd.read_csv('../joined_data/'+station_X+'/'+variable_X+'.csv').values[starting_point+3:starting_point+training_size+3,-1]
p_4_train = pd.read_csv('../joined_data/'+station_X+'/'+variable_X+'.csv').values[starting_point+4:starting_point+training_size+4,-1]
p_5_train = pd.read_csv('../joined_data/'+station_X+'/'+variable_X+'.csv').values[starting_point+5:starting_point+training_size+5,-1]
X_train = np.transpose([q_1_train, q_2_train, p_1_train, p_2_train, p_3_train, p_4_train, p_5_train])
Y_train = pd.read_csv('../joined_data/'+station_Y+'/'+variable_Y+'.csv').values[starting_point+6:starting_point+training_size+6,-1]

q_1_test = pd.read_csv('../joined_data/'+station_Y+'/'+variable_Y+'.csv').values[starting_point+training_size+5:starting_point+training_size+testing_size+5,-1]
q_2_test = pd.read_csv('../joined_data/'+station_Y+'/'+variable_Y+'.csv').values[starting_point+training_size+4:starting_point+training_size+testing_size+4,-1]
p_1_test = pd.read_csv('../joined_data/'+station_X+'/'+variable_X+'.csv').values[starting_point+training_size+1:starting_point+training_size+testing_size+1,-1]
p_2_test = pd.read_csv('../joined_data/'+station_X+'/'+variable_X+'.csv').values[starting_point+training_size+2:starting_point+training_size+testing_size+2,-1]
p_3_test = pd.read_csv('../joined_data/'+station_X+'/'+variable_X+'.csv').values[starting_point+training_size+3:starting_point+training_size+testing_size+3,-1]
p_4_test = pd.read_csv('../joined_data/'+station_X+'/'+variable_X+'.csv').values[starting_point+training_size+4:starting_point+training_size+testing_size+4,-1]
p_5_test = pd.read_csv('../joined_data/'+station_X+'/'+variable_X+'.csv').values[starting_point+training_size+5:starting_point+training_size+testing_size+5,-1]
X_test = np.transpose([q_1_test, q_2_test, p_1_test, p_2_test, p_3_test, p_4_test, p_5_test])
Y_test = pd.read_csv('../joined_data/'+station_Y+'/'+variable_Y+'.csv').values[starting_point+training_size+6:starting_point+training_size+testing_size+6,-1]

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

