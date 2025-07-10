import pandas as pd
import numpy as np
#import ace_tools as tools
def extract_features():
    # Fixing column name typo and re-running the transformation
    df = pd.read_csv('qualifying_data_with_weather.csv')
    # Correct sort values by removing leading spaces in column names
    df.columns = df.columns.str.strip()  # Strip whitespace from column names

    # Re-initialize calculated columns
    df['driver_q_form'] = np.nan
    df['team_q_form'] = np.nan
    df['track_history'] = np.nan
    df['dry_perf'] = np.nan
    df['wet_perf'] = np.nan

    # Sort for rolling calculations
    df = df.sort_values(by=['driver', 'year', 'round'])

    # Calculate rolling features
    driver_groups = df.groupby('driver')
    team_groups = df.groupby('team')

    # A rolling window calculation (also known as a moving window) is a data analysis technique
    # where you compute statistics or other metrics over a specified, fixed-size range (the "window") of data,
    # and then this window "rolls" or slides through the dataset, performing the same calculation for each new window.
    for driver, group in driver_groups:
        df.loc[group.index, 'driver_q_form'] = group['best_q_time'].rolling(window=3, min_periods=1).mean().shift(1)
        dry_times = group[group['weather_condition'] == 'Dry']['best_q_time'].rolling(window=3, min_periods=1).mean().shift(1)
        wet_times = group[group['weather_condition'] == 'Wet']['best_q_time'].rolling(window=3, min_periods=1).mean().shift(1)
        df.loc[dry_times.index, 'dry_perf'] = dry_times
        df.loc[wet_times.index, 'wet_perf'] = wet_times

    for team, group in team_groups:
        df.loc[group.index, 'team_q_form'] = group['best_q_time'].rolling(window=3, min_periods=1).mean().shift(1)

    # Calculate track history
    df['track_key'] = df['driver'] + "_" + df['circuit']
    track_groups = df.groupby('track_key')
    for key, group in track_groups:
        df.loc[group.index, 'track_history'] = group['best_q_time'].rolling(window=3, min_periods=1).mean().shift(1)

    # Create target column
    df['pole'] = (df['position'] == 1).astype(int)

    # Drop temporary columns
    df.drop(columns=['q1_sec', 'q2_sec', 'q3_sec', 'track_key'], inplace=True)

    #tools.display_dataframe_to_user(name="Training Dataset with Features", dataframe=df)
