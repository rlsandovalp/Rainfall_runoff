import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def nse(Y_predicted, Y_test):
    return 1-(np.sum((Y_predicted-Y_test)**2)/np.sum((Y_test-np.mean(Y_test))**2))

def mape(Y_predicted, Y_test):
    return 100*(abs(Y_test-Y_predicted)/Y_test).mean()

def plot_hydrograph(Y_predicted, Y_test):
    fig = plt.figure(dpi = 200)
    ax = fig.add_subplot(111)
    ax.plot(Y_predicted, color = 'red', linewidth = 1, label = 'Y predicted')
    ax.plot(Y_test, '.', label = 'Y observed', markersize = 3)
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Flowrate ['+r'$m^3/s$'+']')
    ax.text(0.03, 0.8, 'MAPE = ' + str('%.2f' % mape(Y_predicted, Y_test)) + '%', transform=ax.transAxes)
    ax.text(0.03, 0.74, 'NSE = ' + str('%.2f' % nse(Y_predicted, Y_test)), transform=ax.transAxes)
    plt.legend(frameon=False)
    plt.show()

def plot_scatter(Y_predicted, Y_test, pl_min, pl_max):

    x = np.linspace(pl_min, pl_max)
    fig = plt.figure(dpi = 200)
    ax = fig.add_subplot(111)
    ax.scatter(Y_test, Y_predicted, s = 1, color = 'black')
    ax.plot(x, x, 'g')
    ax.set_xlabel('Observed ['+r'$m^3/s$'+']')
    ax.set_xlim(pl_min, pl_max)
    ax.set_ylim(pl_min, pl_max)
    ax.set_ylabel('Predicted ['+r'$m^3/s$'+']')
    ax.text(0.03, 0.9, 'MAPE = ' + str('%.2f' % mape(Y_predicted, Y_test)) + '%', transform=ax.transAxes)
    ax.text(0.03, 0.84, 'NSE = ' + str('%.2f' % nse(Y_predicted, Y_test)), transform=ax.transAxes)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

def organize_input_data(train_variables, target_variable, max_dt, starting_point, training_size, testing_size):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for station in train_variables:
        for dt in station[2]:
            X_train.append(pd.read_csv('../joined_data/'+station[0]+'/'+station[1]+'.csv').values[starting_point+(max_dt-dt):starting_point+training_size+(max_dt-dt),-1])
            X_test.append(pd.read_csv('../joined_data/'+station[0]+'/'+station[1]+'.csv').values[starting_point+training_size+(max_dt-dt):starting_point+training_size+testing_size+(max_dt-dt),-1])

    X_train = np.transpose(np.array(X_train))
    X_test = np.transpose(np.array(X_test))


    Y_train = pd.read_csv('../joined_data/'+target_variable[0]+'/'+target_variable[1]+'.csv').values[starting_point+max_dt:starting_point+training_size+max_dt,-1]
    Y_test =  pd.read_csv('../joined_data/'+target_variable[0]+'/'+target_variable[1]+'.csv').values[starting_point+training_size+max_dt:starting_point+training_size+testing_size+max_dt,-1]

    return X_train, X_test, Y_train, Y_test

def pred_bootstrapping(cycles_train, realizations, X, Y, X_test):
    w = []
    b = []

    lin_reg = LinearRegression(n_jobs = 1)
    for i in range(cycles_train):
        X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.25)
        reg = lin_reg.fit(X_train, Y_train)
        w.append(reg.coef_)
        b.append(reg.intercept_)

    predictions = np.zeros((np.shape(X_test)[0], realizations))
    pars_means = np.mean(np.array(w), axis = 0)
    pars2_means = np.mean(np.array(b))
    pars_std = np.std(np.array(w), axis = 0)
    pars2_std = np.std(np.array(b))

    a = []
    b = np.random.normal(pars2_means, pars2_std, realizations)
    for i in range(X_train.shape[1]): a.append(np.random.normal(pars_means[i], pars_std[i], realizations))

    a = np.transpose(np.array(a))
    for i in range (realizations): predictions[:,i] = X_test @ a[i,:] + b[i]

    return (predictions)

def pred_max_lik(realizations, X, Y, X_test):

    lin_reg = LinearRegression(n_jobs = 1)
    reg = lin_reg.fit(X, Y)
    w = reg.coef_
    b = reg.intercept_

    sig2 = (np.transpose(lin_reg.predict(X)-Y) @ (lin_reg.predict(X)-Y))/np.size(Y)
    ones = np.reshape(np.ones(np.shape(X)[0]),[np.size(Y),1])

    H = np.array(np.concatenate((ones, X), axis = 1), dtype='float')
    HH = np.transpose(H) @ H
    J = np.linalg.inv(HH) * sig2

    predictions = np.zeros((np.shape(X_test)[0], realizations))

    a = []
    b = np.random.normal(b, np.sqrt(J[0,0]), realizations)
    for i in range(1,np.shape(J)[0]): a.append(np.random.normal(w[i-1], np.sqrt(J[i,i]), realizations))
    a = np.transpose(np.array(a))

    for i in range (realizations): predictions[:,i] = X_test @ a[i,:] + b[i]
    return (predictions)