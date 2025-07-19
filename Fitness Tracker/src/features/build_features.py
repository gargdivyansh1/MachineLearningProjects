import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.features.DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from src.features.TemporalAbstraction import NumericalAbstraction
from src.features.FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle(f'data/interim/02_outliers_removed_cahuventes.pkl')

prediction_col = list(df.columns[:6])
prediction_col

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['lines.linewidth'] = 2


# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

for col in prediction_col:
    df[col] = df[col].interpolate()

df.info()


# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------
df[df['set'] == 20]['acc_y'].plot()
plt.show()

duration = df[df['set'] == 20].index[-1] - df[df['set'] == 20].index[0]
duration


for s in df['set'].unique():

    start = df[df['set'] == s].index[0]
    end = df[df['set'] == s].index[-1]

    duration = end - start
    df.loc[(df['set'] == s), 'duration'] = duration.seconds 

duration_df = df.groupby(['category'])['duration'].mean()
duration_df


duration_df.iloc[0]

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowPass = df.copy()
LowPass = LowPassFilter()

fs = 1000 / 200 
cutoff = 1.3

df_lowPass = LowPass.low_pass_filter(df_lowPass, 'acc_y', fs, cutoff, order = 5 )

subset = df_lowPass[df_lowPass['set'] == 45]
print(subset)

fig, ax = plt.subplots(nrows= 2, sharex = True, figsize = (20, 10))
ax[0].plot(subset['acc_y'].reset_index(drop = True), label = 'raw data')
ax[1].plot(subset['acc_y_lowpass'].reset_index(drop= True), label = 'blutterworth filter')
ax[0].legend(loc = 'upper center', bbox_to_anchor = (0.5, 1.15), fancybox = True, shadow = True)
ax[1].legend(loc = 'upper center', bbox_to_anchor = (0.5, 1.15), fancybox = True, shadow = True)
plt.show()

# for col in prediction_col:
#     df_lowPass = LowPass.low_pass_filter(df, col, fs, cutoff, order = 5)
#     df_lowPass[col] = df_lowPass[col + '_lowpass']
#     del df_lowPass[col + '_lowpass']

df_lowPass = LowPass.low_pass_filter(df_lowPass, 'acc_x', fs, cutoff, order = 5 )
df_lowPass['acc_x'] = df_lowPass['acc_x' + '_lowpass']

df_lowPass = LowPass.low_pass_filter(df_lowPass, 'acc_z', fs, cutoff, order = 5 )
df_lowPass['acc_z'] = df_lowPass['acc_z' + '_lowpass']

df_lowPass = LowPass.low_pass_filter(df_lowPass, 'acc_y', fs, cutoff, order = 5 )
df_lowPass['acc_y'] = df_lowPass['acc_y' + '_lowpass']

df_lowPass = LowPass.low_pass_filter(df_lowPass, 'gyr_x', fs, cutoff, order = 5 )
df_lowPass['gyr_x'] = df_lowPass['gyr_x' + '_lowpass']

df_lowPass = LowPass.low_pass_filter(df_lowPass, 'gyr_z', fs, cutoff, order = 5 )
df_lowPass['gyr_z'] = df_lowPass['gyr_z' + '_lowpass']

df_lowPass = LowPass.low_pass_filter(df_lowPass, 'gyr_y', fs, cutoff, order = 5 )
df_lowPass['gyr_y'] = df_lowPass['gyr_y' + '_lowpass']

df_lowPass


del df_lowPass['acc_x_lowpass']
del df_lowPass['acc_y_lowpass']
del df_lowPass['acc_z_lowpass']
del df_lowPass['gyr_x_lowpass']
del df_lowPass['gyr_y_lowpass']
del df_lowPass['gyr_z_lowpass']




# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df_lowPass.copy()
PCA_analyser = PrincipalComponentAnalysis()

pc_values = PCA_analyser.determine_pc_explained_variance(df_pca, prediction_col)

plt.figure(figsize=(20,10))
plt.plot(range(1, len(prediction_col) + 1), pc_values)
plt.xlabel("priciple component number")
plt.ylabel("explained variance")
plt.show()

df_pca = PCA_analyser.apply_pca(df_pca, prediction_col, 3)

subset = df_pca[df_pca['set'] == 35]

subset[['pca_1', 'pca_2', 'pca_3']].plot()
plt.show()


# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_squared = df_pca.copy()

acc_r = df_squared['acc_x'] ** 2 + df_squared['acc_y'] ** 2 + df_squared['acc_z']
gyr_r = df_squared['gyr_x'] ** 2 + df_squared['gyr_y'] ** 2 + df_squared['gyr_z']

df_squared['acc_r'] = np.sqrt(acc_r)
df_squared['gyr_r'] = np.sqrt(gyr_r)

subset = df_squared[df_squared['set'] == 35]

subset[['acc_r', 'gyr_r']].plot(subplots = True)
plt.show()

df_squared
# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

import os
print(os.getcwd())


df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

prediction_col =  prediction_col + ['acc_r', 'gyr_r']

prediction_col

ws = int(1000 / 200)

for col in prediction_col:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, 'mean')
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "std")

df_temporal.columns
# in this the data is wrong as the previous 5 values are not like only of squat for squat .. hence it has taken the values of othes set which is not correct ..hence we will be doing it by column by column 

df_temporal_new = df_squared.copy()

df_temporal_new.info()

df_temporal_list = []
for s in df_temporal_new['set'].unique():
    subset = df_temporal_new[df_temporal_new['set'] == s].copy()
    for col in prediction_col:
        subset = NumAbs.abstract_numerical(subset, [col], ws, 'mean')
        subset = NumAbs.abstract_numerical(subset, [col], ws, 'std')

    df_temporal_list.append(subset)

df_temporal_new = pd.concat(df_temporal_list)

df_temporal_new.info()

subset.set

subset[['acc_y', 'acc_y_temp_mean_ws_5', 'acc_y_temp_std_ws_5']].plot()
plt.show()

subset[['gyr_y', 'gyr_y_temp_mean_ws_5', 'gyr_y_temp_std_ws_5']].plot()


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

df_freq = df_temporal_new.copy().reset_index()
df_freq
Freq_Abs = FourierTransformation()

fs = int(1000 / 200)
ws = int(2800 / 200)

df_freq = Freq_Abs.abstract_frequency(df_freq, ['acc_y'], ws, fs)
df_freq

subset = df_freq[df_freq['set'] == 15]
subset[['acc_y']].plot()

subset[
    [
        'acc_y_max_freq',
        'acc_y_freq_weighted',
        'acc_y_pse',
        'acc_y_freq_1.429_Hz_ws_14',
        'acc_y_freq_2.5_Hz_ws_14'
    ]
].plot()
plt.show()

df_freq_new = df_temporal_new.copy()

df_freq_list = []

for s in df_freq_new['set'].unique():
    print(f"Applying fourier transformations to set {s}")
    subset = df_freq_new[df_freq_new['set'] == s].reset_index(drop = True).copy()
    subset = Freq_Abs.abstract_frequency(subset, prediction_col, ws, fs)
    df_freq_list.append(subset)

df_freq_list[0]
subset

df_freq_new = pd.concat(df_freq_list)

df_freq_new

df_freq_new.columns.value_counts()

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

df_freq_new = df_freq_new.dropna()

df_freq_new

df_freq_new = df_freq_new.iloc[::2]
# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

df_cluster = df_freq_new.copy()

cluster_columns = ['acc_x', 'acc_y', 'acc_z']
k_values = range(2, 10)

inertias = []

for k in k_values :
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias)
plt.xlabel('k')
plt.ylabel('Sum of squared distances')
plt.show()

## hence by this the correct amount of clusters is 5
kmeans = KMeans(n_clusters=5, n_init = 20, random_state= 0)
subset = df_cluster[cluster_columns]
df_cluster['cluster']= kmeans.fit_predict(subset)

fig = plt.figure(figsize = (15, 15))
ax = fig.add_subplot(projection='3d')
for c in df_cluster['cluster'].unique():
    subset = df_cluster[df_cluster['cluster'] == c]
    ax.scatter(subset['acc_x'], subset['acc_y'], subset['acc_z'], label = c)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.legend()
plt.show()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle('03_data_features.pkl')