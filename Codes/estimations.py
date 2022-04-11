from functions_ML import *


anticipation = 1
cycles_train = 500
realizations = 100

target_variable = ['Molteno', '9084_1']

precipitation_list = [1,2,3,4,5]
for i in range(anticipation-1): precipitation_list.remove(i+1)

train_variables = [['Molteno', '9084_1', [0+anticipation,1+anticipation]],
                  ['Molteno', '9106_4', precipitation_list]]

max_dt = max(precipitation_list)+1

starting_point = 2000
training_size = 10000
testing_size = 1000

X, X_test, Y, Y_test = organize_input_data(train_variables, target_variable, max_dt, starting_point, training_size, testing_size)

# Y_predicted = pred_bootstrapping(cycles_train, realizations, X, Y, X_test)
Y_predicted = pred_max_lik(realizations, X, Y, X_test)


fig = plt.figure(dpi = 200)
ax = fig.add_subplot(111)
for i in range (realizations): ax.plot(Y_predicted[:,i], color = 'gray', linewidth = 1)
ax.plot(Y_test, '.', color = 'red', label = 'Y observed', markersize = 3)
ax.set_xlabel('Time [hours]')
ax.set_ylabel('Flowrate ['+r'$m^3/s$'+']')
plt.legend(frameon=False)
plt.show()

Y_predicted_mean = np.mean(Y_predicted, axis = 1)
plot_scatter(Y_predicted_mean, Y_test, 0.1, 10)