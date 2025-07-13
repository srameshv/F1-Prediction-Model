import fastf1
import pandas as pd

from get_qualifying_data import get_qualifying_results, get_weather_for_sessions
import feature_extraction as fe

if __name__ == "__main__":
    max_rounds_per_year = {2022: 22, 2023: 22, 2024: 22, 2025: 12}
   # df = get_qualifying_results(max_rounds_per_year)

    #if df.empty:
    #    print("❌ No data retrieved! Check logs above.")
    #else:
    #    df.to_csv("qualifying_data_with_weather.csv", index=False)


    fastf1.Cache.enable_cache('cache')  # stores session data locally
   # df = get_qualifying_results(max_rounds_per_year={
    #2022: 22,
    #2023: 22,
    #2024: 22,
    #2025: 12})
    #df.to_csv("qualifying_data_with_weather.csv", index=False)
    print("Saved to qualifying_data_with_weather.csv")

    #df = get_qualifying_results(range(2024, 2025))
  
    # Load your uploaded qualifying data
    df = pd.read_csv('qualifying_data_with_weather.csv')
    #df_with_weather = get_weather_for_sessions(df)
    #df_with_weather.to_csv('qualifying_data_with_weather.csv',index=False)  # appends weather data to the existing DataFrame

    # print("Weather data added and saved as qualifying_data_with_weather.csv")
    
    #df.to_csv('qualifying_data.csv', index=False)  # index is false since we don't need it
    #print("Saved f1 data to qualifying_data.csv")

    fe.extract_features()  # Extract features from the qualifying data
