from typing import List, Dict, Union, Any

import fastf1
from fastf1.ergast import Ergast
import pandas as pd

def normalize_time(time_str):
    if pd.isna(time_str):
        return None
    try:
        m, s = time_str.split(":")
        return float(m) * 60 + float(s)
    except:
        return None

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
    return 'Unknown'
# 2022 is when the current set of F1 rules and restriction started
# so we will get always past 3 years of data.
def get_qualifying_results(max_rounds_per_year):
    all_data: list[dict[str, Union[int, Any]]] = []
    completed_rounds_2025 = 12
    for year, max_rounds in max_rounds_per_year.items():
        for round_number in range(1, 23):  # most seasons have up to 22 races
            try:
                # Q here indicated qualifying session.
                print(f"  Attempting {year} round {round_number}")
                session = fastf1.get_session(year, round_number, 'Q')
                session.load()
                if session.results is None:
                    print("  No results found for {year} round {round_number}")
                    continue
                # Get session-level weather
                if session.weather_data is not None and 'Weather' in session.weather_data.columns:
                    weather_series = session.weather_data['Weather']
                    weather_str = weather_series.iloc[0] if not weather_series.empty else "Unknown"
                else:
                    weather_str = "Unknown"
                weather_str = normalize_weather(weather_str)
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
                        'weather_condition': weather_str
                    })
            except Exception as e:
                print(f" ❌ Skipping {year} round {round_number} due to: {str(e)}")
    return pd.DataFrame(all_data)




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


