import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
single_file_acc = pd.read_csv(r'Fitness Tracker\data\raw\MetaMotion\MetaMotion\A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv')

print(single_file_acc)
# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------
files = glob('Fitness Tracker/data/raw/MetaMotion/MetaMotion/*.csv')
len(files)

f = files[0]
f
# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------

data_path = 'Fitness Tracker/data/raw/MetaMotion/MetaMotion\\'

# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

def reading_data(files):

    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    for f in files:
    
        participtant = f.split('-')[0].replace(data_path, '')
        label = f.split('-')[1]
        category = f.split('-')[2].rstrip('_MetaWear_2019').rstrip('12345')

        # print(participtant + ' ' + label + ' ' + category)

        df = pd.read_csv(f)

        df['participant'] = participtant
        df['label'] = label
        df['category'] = category

        if 'Accelerometer' in f:
            df['set'] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])

        if 'Gyroscope' in f:
            df['set'] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])

    acc_df.index = pd.to_datetime(acc_df['epoch (ms)'], unit= 'ms') # type: ignore
    gyr_df.index = pd.to_datetime(gyr_df['epoch (ms)'], unit='ms') # type: ignore

    del acc_df['epoch (ms)']
    del acc_df['time (01:00)']
    del acc_df['elapsed (s)']

    del gyr_df['epoch (ms)']
    del gyr_df['time (01:00)']
    del gyr_df['elapsed (s)']

    return acc_df, gyr_df

acceleration_df, gyroscope_df = reading_data(files)

acceleration_df.head()
# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------


# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------


# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------
data_merged = pd.concat([acceleration_df.iloc[:,:3], gyroscope_df], axis = 1)

data_merged.columns = [
    'acc_x',
    'acc_y',
    'acc_z',
    'gyr_x',
    'gyr_y',
    'gyr_z',
    'participtant',
    'label',
    'category',
    'set'
]

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

sampling =  {
    'acc_x': 'mean',
    'acc_y': 'mean',
    'acc_z': 'mean',
    'gyr_x': 'mean',
    'gyr_y': 'mean',
    'gyr_z': 'mean',
    'participtant': 'last',
    'label': 'last',
    'category': 'last',
    'set': 'last'
}
data_merged[:1000].resample(rule='200ms').apply(sampling) # type: ignore
# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

days = [g for n, g in data_merged.groupby(pd.Grouper(freq='D'))]
data_reshampled = pd.concat([df.resample(rule = '200ms').apply(sampling).dropna() for df in days]) # type: ignore

data_reshampled.info()

data_reshampled['set'] = data_reshampled['set'].astype(int)
# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

data_reshampled['set'] = data_reshampled['set'].astype(int)

data_reshampled.info()