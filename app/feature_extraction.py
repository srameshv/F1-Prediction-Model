import pandas as pd
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Convert Q1, Q2, Q3 to seconds
def convert_to_seconds(time_str):
    if pd.isna(time_str):
        return None
    return time_str.total_seconds()


def to_timedelta_safe(val):
    try:
        return pd.to_timedelta(val)
    except Exception:
        return pd.NaT

def timedelta_to_seconds(td):
    if pd.isna(td):
        return None
    return td.total_seconds()

# Normalize weather condition
def normalize_weather(w):
    if pd.isna(w):
        return "Unknown"
    w = w.lower()
    if 'dry' in w:
        return 'Dry'
    elif 'wet' in w or 'rain' in w:
        return 'Wet'
    elif 'cloud' in w:
        return 'Cloudy'
    else:
        return 'Unknown'


import pandas as pd


def evaluate_ranking_per_race(clf, X_test_imputed, y_test, X_test_original_df, full_df):
    """
    Evaluates how well the model ranks drivers by pole probability per race.

    Args:
        clf: Trained classifier (must support predict_proba)
        X_test_imputed: Test features after imputation (used for prediction)
        y_test: Ground truth labels
        X_test_original_df: Original X_test DataFrame with indices matching full_df
        full_df: The full qualifying dataframe with columns: year, round, driver
    """
    # Predict pole probabilities
    probas = clf.predict_proba(X_test_imputed)[:, 1]

    # Build enriched test DataFrame
    X_test_with_meta = X_test_original_df.copy()
    X_test_with_meta['year'] = full_df.loc[X_test_original_df.index, 'year'].values
    X_test_with_meta['round'] = full_df.loc[X_test_original_df.index, 'round'].values
    X_test_with_meta['driver'] = full_df.loc[X_test_original_df.index, 'driver'].values
    X_test_with_meta['true_pole'] = y_test.values
    X_test_with_meta['pole_proba'] = probas

    top1_correct = 0
    top3_correct = 0
    total_races = 0

    # Evaluate per race
    for (year, rnd), group in X_test_with_meta.groupby(['year', 'round']):
        total_races += 1
        sorted_group = group.sort_values(by='pole_proba', ascending=False).reset_index(drop=True)

        # Top-1 accuracy
        if sorted_group.loc[0, 'true_pole'] == 1:
            top1_correct += 1

        # Top-3 accuracy
        if 1 in sorted_group.loc[:2, 'true_pole'].values:
            top3_correct += 1

        # Optional: print top candidates
        print(f"\n--- Race: {year} Round {rnd} ---")
        for i, row in sorted_group.head(3).iterrows():
            marker = "🏁" if row['true_pole'] == 1 else ""
            print(f"Rank {i + 1}: {row['driver']} - Proba={row['pole_proba']:.3f} {marker}")

    print(f"\n Total Races Evaluated: {total_races}")
    print(f" Top-1 Accuracy: {top1_correct}/{total_races} = {top1_correct / total_races:.2%}")
    print(f" Top-3 Accuracy: {top3_correct}/{total_races} = {top3_correct / total_races:.2%}")


def extract_features():
    # Fixing column name typo and re-running the transformation
    df = pd.read_csv('qualifying_data_with_weather.csv')
    # Correct sort values by removing leading spaces in column names
    df.columns = df.columns.str.strip()  # Strip whitespace from column names

    # Step 1: Convert q1/q2/q3 columns to timedelta safely
    for col in ['q1', 'q2', 'q3']:
        df[col] = df[col].apply(to_timedelta_safe)
        df[col + '_sec'] = df[col].apply(timedelta_to_seconds)

    # Compute best qualifying time from Q3 > Q2 > Q1
    df['best_q_time'] = df[['q3_sec', 'q2_sec', 'q1_sec']].apply(
        lambda row: next((x for x in row if pd.notna(x)), None), axis=1
    )

    df['weather_condition'] = df['weather_condition'].apply(normalize_weather)

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
        dry_times = group[group['weather_condition'] == 'Dry']['best_q_time'].rolling(window=3,
                                                                                      min_periods=1).mean().shift(1)
        wet_times = group[group['weather_condition'] == 'Wet']['best_q_time'].rolling(window=3,
                                                                                      min_periods=1).mean().shift(1)
        df.loc[dry_times.index, 'dry_perf'] = dry_times
        df.loc[wet_times.index, 'wet_perf'] = wet_times

    # This code calculates a 3-period backward-looking rolling average of the 'best_q_time' for each team,
    # and then assigns this calculated average to the 'team_q_form' column in the main DataFrame (df)
    # for the corresponding rows, effectively representing a "form" metric based on past performance.
    for team, group in team_groups:
        # df.loc[...] selects specific rows and columns in the DataFrame (df) and assigns values to them.
        # selects the rows corresponding to the current group being processed.
        # 'team_q_form' is the column where the calculated values are stored.
        # group['best_q_time']: This selects the "best qualifying time" column for the current group
        # 'rolling(window=3, min_periods=1).mean().shift(1)': This calculates the rolling mean of the "best qualifying time"
        # window=3: Sets the window size to 3. Each calculation considers the current and the two previous observations within that group
        df.loc[group.index, 'team_q_form'] = group['best_q_time'].rolling(window=3, min_periods=1).mean().shift(1)

    # Calculate track history
    df['track_key'] = df['driver'] + "_" + df['circuit']
    track_groups = df.groupby('track_key')
    for key, group in track_groups:
        df.loc[group.index, 'track_history'] = group['best_q_time'].rolling(window=3, min_periods=1).mean().shift(1)

    # Create target column
    df['pole'] = (df['position'] == 1).astype(int)
    print("----------- POLE POSITION COUNTS --------")
    print(df['pole'].value_counts())

    # Drop temporary columns
    df.drop(columns=['q1_sec', 'q2_sec', 'q3_sec', 'track_key'], inplace=True)
    print("--------- WAITING OUTSIDE --------")
    print(df)
    # Drop rows with missing values
    required_cols = ['best_q_time', 'driver_q_form', 'team_q_form', 'track_history']
    print(f"Before dropna: {df.shape}")
    df = df.dropna(subset=['best_q_time'])
    print(f"After dropna: {df.shape}")
    df = df.dropna(subset=required_cols)

    # One-hot encoding
    '''
    When you have categorical columns (like team names, weather types, circuit names), most ML models (like XGBoost, Logistic Regression, etc.) 
    can’t use them directly because they expect numerical input. One-hot encoding solves this by: Creating a new column for each unique category
    Filling it with 1s and 0s to represent presence or absence of that category
    pd.get_dummies(...) => This is a Pandas function that converts categorical columns into dummy/indicator variables
    drop_first=True => This avoids multicollinearity by dropping the first category from each column
    Example: if team has ["Red Bull", "Ferrari", "Mercedes"], it will create:
    team_Ferrari
    team_Mercedes
    and drop team_Red Bull
    This is safe for ML models because the dropped column is implied

    '''
    df = pd.get_dummies(df, columns=['team', 'weather_condition', 'circuit'], drop_first=True)

    # Feature and target
    feature_cols = ['best_q_time', 'driver_q_form', 'team_q_form', 'track_history', 'dry_perf', 'wet_perf']
    feature_cols += [col for col in df.columns if
                     col.startswith('team_') or col.startswith('weather_condition_') or col.startswith('circuit_')]
    X = df[feature_cols]
    X = X.dropna(axis=1, how='all')  # This removes 'dry_perf' and 'wet_perf' safely
    feature_cols = X.columns.tolist()

    y = df['pole']

    # Train/test split
    print("Splitting data into train and test sets...")
    print(X)
    print("TARGET DATA")
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    X_test_original = X_test.copy()
    from sklearn.impute import SimpleImputer

    # Impute before SMOTE
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)

    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_imputed, y_train)
    print("After SMOTE:", Counter(y_train_resampled))

    # Train model
    # random_state is a parameter to control randomness so results are reproducible. Results are stable, consistent across runs and environments
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    #model.fit(X_train, y_train)
    model.fit(X_train_resampled, y_train_resampled)

    probas = model.predict_proba(X_test_imputed)
    top_preds = probas[:, 1].argsort()[::-1]  # Highest pole probability first

    for i in top_preds[:5]:
        print(f"Rank {i}: Proba={probas[i, 1]:.3f}, True Label={y_test.iloc[i]}")

    evaluate_ranking_per_race(
        clf=model,
        X_test_imputed=X_test_imputed,
        y_test=y_test,
        X_test_original_df=X_test_original,
        full_df=df
    )

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Display evaluation
    results_df = pd.DataFrame(report).transpose()
    import ace_tools_open as tools
    tools.display_dataframe_to_user(name="Random Forest Evaluation", dataframe=results_df)
