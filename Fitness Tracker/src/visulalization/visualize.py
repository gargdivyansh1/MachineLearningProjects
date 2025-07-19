import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle('Fitness Tracker/data/interim/01_data_processed.pkl')
print(df)
# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------

set_df = df[df['set'] == 1]
print(set_df)
plt.plot(set_df['acc_y'])
plt.show()

# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

mpl.style.use("seaborn-v0_8-deep") # type: ignore
mpl.rcParams['figure.figsize'] = (20, 5)
mpl.rcParams['figure.dpi'] = 100 

for label in df['label'].unique():
    subset = df[df['label'] == label]
    fig, ax = plt.subplots()
    plt.plot(subset['acc_y'].reset_index(drop=True), label = label)
    plt.legend()
    plt.show()


# now if we want for the smaller time duration 
for label in df['label'].unique():
    subset = df[df['label'] == label]
    fig, ax = plt.subplots()
    # this will only take first 100 rows of data and plot them 
    plt.plot(subset[:100]['acc_y'].reset_index(drop=True), label = label)
    plt.legend()
    plt.show()


# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------
mpl.style.use("seaborn-v0_8-deep") # type: ignore
mpl.rcParams['figure.figsize'] = (20, 5)
mpl.rcParams['figure.dpi'] = 100 

# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------

# here we are again going to make the subsets 
# so we are making by using different technique
# example only
category_df = df.query("label == 'squat'").query("participtant == 'A'").reset_index()
print(category_df)

fig, ax = plt.subplots()
category_df.groupby(['category'])['acc_y'].plot()
ax.set_ylabel('acc_y')
ax.set_xlabel('samples')
plt.legend()
plt.show()

fig, ax = plt.subplots()
category_df.groupby(['category'])['acc_x'].plot()
ax.set_ylabel('acc_x')
ax.set_xlabel('samples')
plt.legend()
plt.show()




# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------
participtant_df = df.query("label == 'squat'").sort_values('participtant').reset_index()
print(participtant_df)

fig, ax = plt.subplots()
participtant_df.groupby(['participtant'])['acc_y'].plot()
plt.legend()
plt.show()

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------

label = 'squat'
participtant = 'A'
all_axis_data_frame = df.query(f"label == '{label}'").query(f"participtant == '{participtant}'").reset_index()
print(all_axis_data_frame)

fig, ax = plt.subplots()
all_axis_data_frame[['acc_y', 'acc_x', 'acc_z']].plot()
ax.set_xlabel('samples')
ax.set_ylabel('acc_y acc_x acc_z')
plt.legend()
plt.show()

# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------

labels = df['label'].unique()
labels = sorted(labels)
participtants = df['participtant'].unique()
participtants = sorted(participtants)

for label in labels :
    for participtant in participtants:

        all_axis_df = (
            df.query(f"label == '{label}'").
            query(f"participtant == '{participtant}'").
            reset_index()
        )

        if(len(all_axis_df) > 0):

            fig, ax = plt.subplots()
            all_axis_df[['acc_y', 'acc_x', 'acc_z']].plot(ax = ax)
            ax.set_xlabel('samples')
            ax.set_ylabel('acc_y acc_x acc_z')
            plt.title(f"{participtant} doing {label}")
            plt.legend()
            plt.show()

for label in labels :
    for participtant in participtants:

        all_axis_df = (
            df.query(f"label == '{label}'").
            query(f"participtant == '{participtant}'").
            reset_index()
        )

        if(len(all_axis_df) > 0):

            fig, ax = plt.subplots()
            all_axis_df[['gyr_y', 'gyr_x', 'gyr_z']].plot(ax = ax)
            ax.set_xlabel('samples')
            ax.set_ylabel('gyr_y gyr_x gyr_z')
            plt.title(f"{participtant} doing {label}")
            plt.legend()
            plt.show()
        

# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------

label = 'row'
participtant = 'A'
combined_plot_df = (
    df.query(f"label == '{label}'")
    .query(f"participtant == '{participtant}'")
    .reset_index(drop = True)
)

fig, ax = plt.subplots(nrows = 2, sharex = True, figsize = (20, 10))
combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax = ax[0])
combined_plot_df[['gyr_x', 'gyr_y', 'gyr_z']].plot(ax = ax[1])

ax[0].legend(
    loc = "upper center", bbox_to_anchor = (0.5, 1.15), ncol= 3, fancybox = True, shadow = True
)
ax[1].legend(
    loc = "upper center", bbox_to_anchor = (0.5, 1.15), ncol= 3, fancybox = True, shadow = True
)
ax[1].set_xlabel('samples')

# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------

labels = df['label'].unique()
labels = sorted(labels)
participtants = df['participtant'].unique()
participtants = sorted(participtants)

for label in labels :
    for participtant in participtants:

        combined_plot_df = (
            df.query(f"label == '{label}'").
            query(f"participtant == '{participtant}'").
            reset_index()
        )

        if(len(combined_plot_df) > 0):

            fig, ax = plt.subplots(nrows = 2, sharex = True, figsize = (20, 10))
            combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax = ax[0])
            combined_plot_df[['gyr_x', 'gyr_y', 'gyr_z']].plot(ax = ax[1])

            ax[0].legend(
                loc = "upper center", bbox_to_anchor = (0.5, 1.15), ncol= 3, fancybox = True, shadow = True
            )
            ax[1].legend(
                loc = "upper center", bbox_to_anchor = (0.5, 1.15), ncol= 3, fancybox = True, shadow = True
            )
            ax[1].set_xlabel('samples')

            plt.savefig(f'../../reports/figures/{label.title()} ({participtant}).png')

            plt.show()