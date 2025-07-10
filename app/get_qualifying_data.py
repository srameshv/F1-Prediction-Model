from typing import List, Dict, Union, Any

import fastf1
from fastf1.ergast import Ergast
import pandas as pd


# 2022 is when the current set of F1 rules and restriction started
# so we will get always past 3 years of data.
def get_qualifying_results(year_range):
    all_data: list[dict[str, Union[int, Any]]] = []
    for year in year_range:
        for round_number in range(1, 23):  # most seasons have up to 22 races
            try:
                # Q here indicated qualifying session.
                session = fastf1.get_session(year, round_number, 'Q')
                session.load()
                for row in session.results.itertuples():
                    all_data.append({
                        'year': year,
                        'round': round_number,
                        'circuit': session.event['Location'],
                        'driver': row.FullName,
                        'team': row.TeamName,
                        'position': row.Position,
                        'q1': row.Q1,
                        'q2': row.Q2,
                        'q3': row.Q3,
                    })
            except Exception as e:
                print('Skipping {year} round {round_number}: {e}')
    return pd.DataFrame(all_data)


# Enable caching for FastF1
fastf1.Cache.enable_cache('cache')


def get_weather_for_sessions(df):
    weather_map = {}

    # Go through each (year, round) combo
    for year, rnd in df[['year', 'round']].drop_duplicates().values:
        try:
            session = fastf1.get_session(int(year), int(rnd), 'Q')
            session.load()
            weather_series = session.weather_data['Weather']
            if not weather_series.empty:
                weather = weather_series.iloc[0]  # Take weather at session start
            else:
                weather = "Unknown"
            weather_map[(year, rnd)] = weather
        except Exception as e:
            print(f"Failed to load {year} R{rnd}: {e}")
            weather_map[(year, rnd)] = "Unknown"

    # Apply weather to DataFrame
    df['weather_condition'] = df.apply(lambda row: weather_map.get((row['year'], row['round']), "Unknown"), axis=1)

    return df


