import pandas as pd
import numpy as np
import scipy as sp
from scipy import signal
from matplotlib import pyplot as plt
from statsmodels.tsa import stattools

# https://towardsdatascience.com/computing-cross-correlation-between-geophysical-time-series-488642be7bf0

# Have some time series data (via pandas)
station = 'Lesmo'
variable = '8120_1'
data1 = pd.read_csv('../joined_data/'+station+'/'+variable+'.csv').values[0:20000,-1]

# Select relevant data, index by Date
corr1, conf_int = stattools.pacf(data1, 24, alpha=0.05)

fig, (ax_orig, ax_corr) = plt.subplots(2, 1, figsize=(8, 8))
ax_orig.plot(data1)
ax_orig.set_title('Flowrate [cms]')
ax_orig.set_xlabel('Time [hours]')
ax_corr.plot(corr1)
ax_corr.set_title('Partial autocorrelation function')
ax_corr.set_xlabel('Lag')
ax_orig.margins(0, 0.1)
ax_corr.margins(0, 0.1)
fig.tight_layout()

print(conf_int)
plt.show()
