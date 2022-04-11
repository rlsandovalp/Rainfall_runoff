import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

## https://www.alpharithms.com/autocorrelation-time-series-python-432909/

station = 'Lesmo/'
variable = '8120_1'

# Have some time series data (via pandas)
data = pd.read_csv('../joined_data/'+station+variable+'.csv')
# Select relevant data, index by Date
data = data[['Data-Ora', 'value']].set_index(['Data-Ora'])
# Calculate the ACF (via statsmodel)
plot_acf(data, lags=100)
# Show the data as a plot (via matplotlib)
plt.show()