from functions_ML import *

anticipation = 2
cycles_train = 500
realizations = 500

target_variable = ['Costa', '8148_1']

precipitation_list = [1,2,3,4,5]
temperature_list = [1,2,3,4,5]

for i in range(anticipation-1): 
    if i+1 in precipitation_list:
        precipitation_list.remove(i+1)
    if i+1 in temperature_list:
        temperature_list.remove(i+1)        


train_variables = [['Costa', '8148_1', [0+anticipation, 1+anticipation, 2+anticipation]],
                  ['Molteno', '9084_1', [0+anticipation, 1+anticipation, 2+anticipation]],
                  ['Caslino', '8124_1', [0+anticipation,1+anticipation, 2+anticipation]],
                  ['Molteno', '9106_4', precipitation_list],
                  ['Molteno', '9017_1', temperature_list],
                  ['Caslino', '8122_4', precipitation_list],
                  ['Caslino', '8123_1', temperature_list],
                  ['Canzo', '2614_4', precipitation_list],
                  ['Canzo', '2613_1', temperature_list],
                  ['Erba', '5870_4', precipitation_list],
                  ['Erba', '5871_1', temperature_list],
                  ['Lambrugo', '8197_4', precipitation_list],
                  ['Lambrugo', '8198_1', temperature_list]]

max_dt = max(precipitation_list)+1

starting_point = 2000
training_size = 10000
testing_size = 1000

X, X_test, Y, Y_test = organize_input_data(train_variables, target_variable, max_dt, starting_point, training_size, testing_size)

# Y_predicted = pred_bootstrapping(cycles_train, realizations, X, Y, X_test)
Y_predicted = pred_max_lik(realizations, X, Y, X_test)
Y_predicted_mean = np.mean(Y_predicted, axis = 1)
Y_predicted_std = np.std(Y_predicted, axis = 1)

fig = plt.figure(dpi = 200)
ax = fig.add_subplot(111)
for i in range (realizations): ax.plot(Y_predicted[:,i], color = 'gray', linewidth = 1)
ax.plot(Y_test, '.', color = 'red', label = 'Y observed', markersize = 3)
ax.plot(Y_predicted_mean, color = 'blue', label = 'Mean Y predicted', linewidth = 1)
ax.plot(Y_predicted_mean + 3*Y_predicted_std, color = 'black', linewidth = 1)
ax.plot(Y_predicted_mean - 3*Y_predicted_std, color = 'black', linewidth = 1)
ax.set_xlabel('Time [hours]')
ax.set_ylabel('Flowrate ['+r'$m^3/s$'+']')
plt.legend(frameon=False)
plt.show()

plot_scatter(Y_predicted_mean, Y_test, 1, 100)