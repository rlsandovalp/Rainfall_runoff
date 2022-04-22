from functions_ML import *

anticipation = 1
window = 5
cycles_train = 500
realizations = 500

target_variable = ['Costa', '8148_1']

training_variables = [['Costa', '8148_1', 'h [cm]'],
                  ['Molteno', '9084_1', 'h [cm]'],
                  ['Molteno', '9017_1', 'T [C]'],
                  ['Molteno', '9106_4', 'P [mm]'],
                  ['Molteno', '11020_1', 'HR [%]'],
                  ['Caslino', '8122_4', 'P [mm]'],
                  ['Caslino', '8123_1', 'T [C]'],
                  ['Caslino', '8124_1', 'h [cm]'],
                  ['Canzo', '2614_4', 'P [mm]'],
                  ['Erba', '6163_1', 'RH [%]'],
                  ['Erba', '5871_1', 'T [C]'],
                  ['Erba', '5870_4', 'P [mm]'],
                  ['Lambrugo', '8198_1', 'T [C]'],
                  ['Lambrugo', '8197_4', 'P [mm]']]

max_dt = window + anticipation

starting_point = 0
training_size = 57000
testing_size = 1000

X, X_test, Y, Y_test = organize_input_data(training_variables, target_variable, max_dt, starting_point, training_size, testing_size, anticipation, window)

Y_predicted = pred_bootstrapping(cycles_train, realizations, X, Y, X_test)
# Y_predicted = pred_max_lik(realizations, X, Y, X_test)
Y_predicted_mean = np.mean(Y_predicted, axis = 1)
Y_predicted_std = np.std(Y_predicted, axis = 1)

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
plt.plot(Y_predicted_mean, '.', label = 'LSTM')
plt.plot(Y_test, '.', label = 'Measured')
plt.plot(Y_predicted_mean + 2*Y_predicted_std, color = 'black', linewidth = 1)
plt.plot(Y_predicted_mean - 2*Y_predicted_std, color = 'black', linewidth = 1)
plt.ylabel('h level [cm]')
plt.legend()

A = np.zeros(1000)
for i in range(1000):
    if (Y_test[i] < Y_predicted_mean[i] - 2*Y_predicted_std[i]):
        A[i] = (Y_test[i] - (Y_predicted_mean[i] - 2*Y_predicted_std[i]))/Y_test[i]
    elif (Y_test[i] > Y_predicted_mean[i] + 2*Y_predicted_std[i]):
        A[i] = (Y_test[i] - (Y_predicted_mean[i] + 2*Y_predicted_std[i]))/Y_test[i]
    else:
        A[i] = 0

plt.subplot(a,b,2)
plt.plot(A, 'r-', label = 'Y_pr - Y_obs')
plt.ylabel('h [cm]')
plt.xlabel('Time [hours]')
plt.legend()

for n_var, variable in enumerate(training_variables):
    plt.subplot(a,b,n_var+3)
    plt.plot(X_test[:,n_var], label = training_variables[n_var][0] + ' ' + training_variables[n_var][1])
    plt.xlabel('Time [hours]')
    plt.ylabel(training_variables[n_var][2])
    plt.legend()
plt.show()

# fig = plt.figure(dpi = 200)
# ax = fig.add_subplot(111)
# for i in range (realizations): ax.plot(Y_predicted[:,i], color = 'gray', linewidth = 1)
# ax.plot(Y_test, '.', color = 'red', label = 'Y observed', markersize = 3)
# ax.plot(Y_predicted_mean, '.', color = 'blue', label = 'Mean Y predicted', markersize = 3)
# ax.plot(Y_predicted_mean + 3*Y_predicted_std, color = 'black', linewidth = 1)
# ax.plot(Y_predicted_mean - 3*Y_predicted_std, color = 'black', linewidth = 1)
# ax.set_xlabel('Time [hours]')
# ax.set_ylabel('Flowrate ['+r'$m^3/s$'+']')
# plt.legend(frameon=False)
# plt.show()

# plot_scatter(Y_predicted_mean, Y_test, 10, 1000)