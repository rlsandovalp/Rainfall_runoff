import pandas as pd
import numpy as np
import scipy as sp
from scipy import signal
from matplotlib import pyplot as plt
from statsmodels.tsa import stattools

# plt.style.use(['science','nature'])

# https://towardsdatascience.com/computing-cross-correlation-between-geophysical-time-series-488642be7bf0

# Have some time series data (via pandas)
station1 = 'Molteno'
variable1 = '9084_1'
station2 = 'Molteno'
variable2 = '9106_4'
values = 200

data1 = pd.read_csv('../joined_data/'+station1+'/'+variable1+'.csv').values[0:values,-1]
data2 = pd.read_csv('../joined_data/'+station2+'/'+variable2+'.csv').values[0:values,-1]
# Select relevant data, index by Date
corr1 = stattools.ccf(data1, data2)

fig, (ax_qoi, ax_var, ax_corr) = plt.subplots(3, 1, figsize = (8, 10), dpi = 100)
ax_qoi.plot(data1)
ax_qoi.set_title('Flowrate [cms] '+station1+' '+variable1)
ax_var.plot(data2)
ax_var.set_title('Flowrate [cms] '+station2+' '+variable2)
ax_var.set_xlabel('Time [hours]')
ax_corr.plot(corr1[0:24])
ax_corr.set_title('Cross-correlated signal')
ax_corr.set_xlabel('Lag')
ax_qoi.margins(0, 0.1)
ax_var.margins(0, 0.1)
ax_corr.margins(0, 0.1)
fig.tight_layout()
plt.show()
