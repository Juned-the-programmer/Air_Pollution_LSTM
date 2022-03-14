import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set()

## For statistical insights
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats.distributions import chi2 

## For ARCH Modelling
from arch import arch_model
# import arch
print("Necessary packages imported successfully!")

url = 'Dataset/LSTM-Multivariate_pollution.csv'
raw_csv_data = pd.read_csv(url,error_bad_lines=False)
print(raw_csv_data.head(10))

df_comp=raw_csv_data.copy()
df_comp.date = pd.to_datetime(df_comp.date, dayfirst = True)
df_comp.set_index("date", inplace=True)
df_comp=df_comp.asfreq('b')
df_comp=df_comp.fillna(method='ffill')

del df_comp['dew']
del df_comp['temp']
del df_comp['press']
del df_comp['wnd_spd']
size = int(len(df_comp)*0.8)
df, df_test = df_comp.iloc[:size], df_comp.iloc[size:]

def LLR_test(mod_1, mod_2, DF = 1):
    L1 = mod_1.fit(start_ar_lags = 11).llf
    L2 = mod_2.fit(start_ar_lags = 11).llf
    LR = (2*(L2-L1))    
    p = chi2.sf(LR, DF).round(3)
    return p

df['press'] = df.pollution.pct_change(1)*100

df['wnd_spd'] = df.pollution.mul(df.pollution)

df.pollution.plot(figsize=(20,5))
plt.title("Pollution Data", size = 24)
plt.show()

df.wnd_spd.plot(figsize=(20,5))
plt.title("Wind Speed", size = 24)
plt.show()

sgt.plot_pacf(df.pollution[1:], lags = 40, alpha = 0.05, zero = False , method = ('ols'))
plt.title("PACF of Pollution", size = 20)
plt.show()

sgt.plot_pacf(df.wnd_spd[1:], lags = 40, alpha = 0.05, zero = False , method = ('ols'))
plt.title("PACF of Wind Speed", size = 20)
plt.show()

model_arch_1 = arch_model(df.pollution[1:])
results_arch_1 = model_arch_1.fit(update_freq = 5)
results_arch_1.summary()