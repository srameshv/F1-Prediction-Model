import pandas as pd
import fastf1 as ff1
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class F1FastF1DataCollector:
    """
    Modern F1 data collection using FastF1 library (recommended approach)
    FastF1 provides official F1 timing data, telemetry, and session information
    """

    def __init__(self):
        # Enable FastF1 cache for faster subsequent requests
        ff1.Cache.enable_cache('cache')  # Creates a cache directory
        print("FastF1 Data Collector initialized")
        print("Cache enabled for faster data loading")

    def get_qualifying_data_season(self, year, max_races=None):
        """
        Get qualifying data for an entire season using FastF1

        Args:
            year (int): Season year (2018-2025)
            max_races (int): Limit number of races (for testing)
        """
        print(f"Loading {year} season qualifying data...")

        # Get the season schedule
        try:
            schedule = ff1.get_event_schedule(year)
            if max_races:
                schedule = schedule.head(max_races)

            all_qualifying_data = []

            for idx, event in schedule.iterrows():
                race_name = event['EventName']
                country = event['Country']
                circuit = event['Location']

                print(f"Processing: {race_name}")

                try:
                    # Load qualifying session
                    session = ff1.get_session(year, event['RoundNumber'], 'Q')
                    session.load()

                    # Get qualifying results
                    results = session.results

                    if not results.empty:
                        for _, driver_result in results.iterrows():
                            qualifying_record = {
                                'year': year,
                                'round': event['RoundNumber'],
                                'race_name': race_name,
                                'country': country,
                                'circuit_name': circuit,
                                'date': event['EventDate'].strftime('%Y-%m-%d'),
                                'session_date': session.date.strftime('%Y-%m-%d') if session.date else None,

                                # Driver info
                                'driver_number': driver_result['DriverNumber'],
                                'driver_code': driver_result['Abbreviation'],
                                'driver_name': driver_result['FullName'],
                                'team_name': driver_result['TeamName'],
                                'team_color': driver_result['TeamColor'],

                                # Qualifying performance
                                'position': driver_result['Position'],
                                'q1_time': self._convert_timedelta_to_seconds(driver_result['Q1']),
                                'q2_time': self._convert_timedelta_to_seconds(driver_result['Q2']),
                                'q3_time': self._convert_timedelta_to_seconds(driver_result['Q3']),
                                'best_time': self._convert_timedelta_to_seconds(driver_result['Time']),

                                # Additional data
                                'grid_position': driver_result.get('GridPosition', driver_result['Position']),
                                'pole_position': 1 if driver_result['Position'] == 1 else 0,
                            }
                            all_qualifying_data.append(qualifying_record)

                    # Small delay to be respectful
                    import time
                    time.sleep(0.5)

                except Exception as e:
                    print(f" Error loading {race_name}: {str(e)}")
                    continue

            df = pd.DataFrame(all_qualifying_data)
            print(f"Collected {len(df)} qualifying results from {year}")
            return df

        except Exception as e:
            print(f"Error loading {year} season: {str(e)}")
            return pd.DataFrame()

    def get_practice_session_data(self, year, round_number, session_type='FP3'):
        """
        Get practice session data (useful for predicting qualifying performance)

        Args:
            session_type: 'FP1', 'FP2', 'FP3', 'Sprint' etc.
        """
        try:
            session = ff1.get_session(year, round_number, session_type)
            session.load()

            results = session.results
            lap_times = session.laps

            practice_data = []

            for _, driver_result in results.iterrows():
                driver_number = driver_result['DriverNumber']
                driver_laps = lap_times[lap_times['DriverNumber'] == driver_number]

                if not driver_laps.empty:
                    # Calculate practice performance metrics
                    fastest_lap = driver_laps['LapTime'].min()
                    total_laps = len(driver_laps)

                    practice_record = {
                        'year': year,
                        'round': round_number,
                        'session_type': session_type,
                        'driver_number': driver_number,
                        'driver_code': driver_result['Abbreviation'],
                        'driver_name': driver_result['FullName'],
                        'team_name': driver_result['TeamName'],
                        'position': driver_result['Position'],
                        'fastest_lap_time': self._convert_timedelta_to_seconds(fastest_lap),
                        'total_laps': total_laps,
                        'best_time': self._convert_timedelta_to_seconds(driver_result['Time'])
                    }
                    practice_data.append(practice_record)

            return pd.DataFrame(practice_data)

        except Exception as e:
            print(f"Error loading {session_type} data: {str(e)}")
            return pd.DataFrame()

    def get_current_season_data(self, year=2025):
        """
        Get comprehensive current season data including qualifying and race results
        """
        print(f"Loading current {year} season data...")

        # Get qualifying data
        qualifying_df = self.get_qualifying_data_season(year)

        # Get race results for championship standings
        race_results_df = self.get_race_results_season(year)

        # Calculate real championship standings from race results
        if not race_results_df.empty:
            standings = self.calculate_championship_standings(race_results_df, year)
            return qualifying_df, race_results_df, standings
        else:
            print("No race results available - cannot calculate championship standings")
            return qualifying_df, pd.DataFrame(), pd.DataFrame()

    def get_race_results_season(self, year):
        """
        Get race results for championship standings calculation
        """
        print(f"Loading {year} race results for championship standings...")

        try:
            schedule = ff1.get_event_schedule(year)
            all_race_data = []

            for idx, event in schedule.iterrows():
                race_name = event['EventName']

                try:
                    # Load race session
                    session = ff1.get_session(year, event['RoundNumber'], 'R')  # 'R' for Race
                    session.load()

                    # Get race results
                    results = session.results

                    if not results.empty:
                        for _, driver_result in results.iterrows():
                            race_record = {
                                'year': year,
                                'round': event['RoundNumber'],
                                'race_name': race_name,
                                'driver_name': driver_result['FullName'],
                                'team_name': driver_result['TeamName'],
                                'position': driver_result['Position'],
                                'points': driver_result.get('Points', 0),  # FastF1 includes points
                                'grid_position': driver_result.get('GridPosition', 0),
                                'status': driver_result.get('Status', 'Finished')
                            }
                            all_race_data.append(race_record)

                    import time
                    time.sleep(0.5)  # Be respectful to the API

                except Exception as e:
                    print(f" Error loading {race_name} race: {str(e)}")
                    continue

            return pd.DataFrame(all_race_data)

        except Exception as e:
            print(f"Error loading {year} race results: {str(e)}")
            return pd.DataFrame()

    def calculate_championship_standings(self, race_results_df, year):
        """
        Calculate REAL championship standings using standard F1 points system

        Standard F1 Points (2010-present):
        1st: 25, 2nd: 18, 3rd: 15, 4th: 12, 5th: 10,
        6th: 8, 7th: 6, 8th: 4, 9th: 2, 10th: 1
        """

        # Standard F1 points system (unchanged since 2010)
        F1_POINTS_SYSTEM = {
            1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
            6: 8, 7: 6, 8: 4, 9: 2, 10: 1
        }

        if race_results_df.empty:
            print("No race results available for championship calculation")
            return pd.DataFrame()

        standings_data = []

        for driver in race_results_df['driver_name'].unique():
            driver_races = race_results_df[race_results_df['driver_name'] == driver]

            # Calculate total championship points
            total_points = 0
            wins = 0
            podiums = 0

            for _, race in driver_races.iterrows():
                position = race['position']

                # Use FastF1 points if available, otherwise calculate from position
                if pd.notna(race.get('points')) and race['points'] > 0:
                    total_points += race['points']
                elif position in F1_POINTS_SYSTEM:
                    total_points += F1_POINTS_SYSTEM[position]

                # Count wins and podiums
                if position == 1:
                    wins += 1
                if position <= 3:
                    podiums += 1

            standings_data.append({
                'driver_name': driver,
                'team_name': driver_races['team_name'].iloc[0] if len(driver_races) > 0 else 'Unknown',
                'points': total_points,
                'wins': wins,
                'podiums': podiums,
                'races_entered': len(driver_races)
            })

        # Create standings DataFrame
        standings_df = pd.DataFrame(standings_data)
        standings_df = standings_df.sort_values(['points', 'wins'], ascending=False)
        standings_df['championship_position'] = range(1, len(standings_df) + 1)

        print(f"Calculated {year} championship standings using standard F1 points")
        return standings_df

    def create_belgian_gp_features(self, historical_data):
        """
        Create features specifically for Belgian GP prediction
        Using historical Spa data
        """
        print(" Creating Belgian GP specific features...")

        # Filter for Spa historical data
        spa_data = historical_data[
            historical_data['circuit_name'].str.contains('Spa', case=False, na=False) |
            historical_data['race_name'].str.contains('Belgian', case=False, na=False)
            ].copy()

        if spa_data.empty:
            print("No historical Spa data found")
            return pd.DataFrame()

        # Calculate Spa-specific performance metrics
        spa_features = []

        for driver in spa_data['driver_name'].unique():
            driver_spa_data = spa_data[spa_data['driver_name'] == driver]

            if len(driver_spa_data) > 0:
                avg_position = driver_spa_data['position'].mean()
                best_position = driver_spa_data['position'].min()
                poles_at_spa = len(driver_spa_data[driver_spa_data['position'] == 1])

                # Recent form (if multiple years)
                if len(driver_spa_data) > 1:
                    recent_performance = driver_spa_data.tail(2)['position'].mean()
                else:
                    recent_performance = avg_position

                spa_features.append({
                    'driver_name': driver,
                    'spa_avg_qualifying_position': avg_position,
                    'spa_best_qualifying_position': best_position,
                    'spa_poles': poles_at_spa,
                    'spa_recent_form': recent_performance,
                    'spa_experience': len(driver_spa_data)
                })

        spa_features_df = pd.DataFrame(spa_features)
        print(f"Created Spa features for {len(spa_features_df)} drivers")
        return spa_features_df

    def engineer_ml_features(self, qualifying_df):
        """
        Engineer machine learning features from FastF1 data
        """
        print("Engineering ML features...")

        df = qualifying_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['driver_name', 'date'])

        # Recent form score (last 5 races)
        df['recent_form_score'] = df.groupby('driver_name')['position'].transform(
            lambda x: 1 - (x.rolling(window=5, min_periods=1).mean() - 1) / 19)

        # Team performance
        df['team_form'] = df.groupby(['team_name', 'year'])['position'].transform(
            lambda x: 1 - (x.rolling(window=3, min_periods=1).mean() - 1) / 19)

        # Qualifying improvement trend
        df['position_trend'] = df.groupby('driver_name')['position'].transform(
            lambda x: x.diff().rolling(window=3, min_periods=1).mean())

        # Circuit type classification
        # Spa-Francorchamps is a POWER circuit because:
        # Kemmel Straight - 1.1km of flat-out speed
        # Long straights where engine power dominates
        # Low downforce setup needed for top speed
        power_circuits = ['Spa-Francorchamps', 'Monza', 'Silverstone', 'Baku']
        downforce_circuits = ['Monaco', 'Hungary', 'Singapore']

        df['circuit_type'] = 'balanced'
        df.loc[df['circuit_name'].str.contains('|'.join(power_circuits), case=False, na=False), 'circuit_type'] = 'power'
        df.loc[df['circuit_name'].str.contains('|'.join(downforce_circuits), case=False,na=False), 'circuit_type'] = 'downforce'

        # Power circuit performance
        # since lower is better
        # TIMES (lower is better, like lap times)
        # score = (max_time - time) / (max_time - min_time)
        # 19 because - denominator = (worst_possible - best_possible). F1 qualifying: 20 - 1 = 19
        df['power_circuit_performance'] = df[df['circuit_type'] == 'power'].groupby('driver_name')[
            'position'].transform(
            lambda x: 1 - (x.mean() - 1) / 19 if len(x) > 0 else 0.5
        ).fillna(0.5)

        print(f" Engineered features for {len(df)} records")
        return df

    def _convert_timedelta_to_seconds(self, time_delta):
        """Convert pandas Timedelta to seconds (float)"""
        if pd.isna(time_delta):
            return None
        try:
            return time_delta.total_seconds()
        except:
            return None

    def load_multi_season_data(self, years=None):
        """
        Load multiple seasons of data for comprehensive training
        """
        if years is None:
            years = [2021, 2022, 2023, 2024, 2025]  # Recent years with current regulations

        all_data = []

        print(f"Loading {len(years)} seasons of data...")

        for year in years:
            season_data = self.get_qualifying_data_season(year, max_races=5 if year == 2025 else None)
            if not season_data.empty:
                all_data.append(season_data)

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            engineered_df = self.engineer_ml_features(combined_df)

            print(f"Combined dataset: {len(engineered_df)} records")
            print(f" Date range: {engineered_df['date'].min()} to {engineered_df['date'].max()}")
            print(f" Unique drivers: {engineered_df['driver_name'].nunique()}")
            print(f" Unique circuits: {engineered_df['circuit_name'].nunique()}")

            return engineered_df

        return pd.DataFrame()


def collect_fastf1_data_for_prediction():
    """
    Main function to collect F1 data using FastF1 for Belgian GP prediction
    """
    print("F1 DATA COLLECTION WITH FASTF1")
    print("=" * 50)
    print("Installing FastF1 if needed...")

    try:
        import fastf1
        print("FastF1 already installed")
    except ImportError:
        print("Installing FastF1...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'fastf1'])
        import fastf1 as ff1

    # Initialize collector
    collector = F1FastF1DataCollector()

    # Load recent seasons data
    historical_data = collector.load_multi_season_data([2022, 2023, 2024, 2025])

    if historical_data.empty:
        print("No data loaded. Check your internet connection and try again.")
        return None, None, None

    # Get current 2025 season data (qualifying + race results + standings)
    current_qualifying, race_results, standings = collector.get_current_season_data(2025)

    # Create Belgian GP specific features
    spa_features = collector.create_belgian_gp_features(historical_data)

    # Save data for future use
    historical_data.to_csv('f1_fastf1_historical_data.csv', index=False)
    if not race_results.empty:
        race_results.to_csv('f1_2025_race_results.csv', index=False)
    if not standings.empty:
        standings.to_csv('f1_2025_standings_fastf1.csv', index=False)
    if not spa_features.empty:
        spa_features.to_csv('f1_spa_features.csv', index=False)

    print(f"\n Data saved:")
    print(f"   Historical data: f1_fastf1_historical_data.csv")
    print(f"   2025 race results: f1_2025_race_results.csv")
    print(f"   2025 standings: f1_2025_standings_fastf1.csv")
    print(f"   Spa features: f1_spa_features.csv")

    # Show current championship standings
    if not standings.empty:
        print(f"\n CURRENT 2025 CHAMPIONSHIP STANDINGS:")
        print(standings[['championship_position', 'driver_name', 'team_name', 'points', 'wins']].head(10))

    # Show recent qualifying performances
    if not current_qualifying.empty:
        print(f"\n RECENT 2025 QUALIFYING PERFORMANCES:")
        recent_races = current_qualifying.groupby('race_name').apply(lambda x: x.loc[x['position'] == 1]).tail(3)
        for _, race in recent_races.iterrows():
            print(f"   {race['race_name']}: Pole = {race['driver_name']} ({race['team_name']})")

    return historical_data, spa_features, standings


# Example usage
if __name__ == "__main__":
    print(" RECOMMENDED: Using FastF1 for F1 Data Collection")
    print(" This is the modern, official way to get F1 timing data")
    print("\n Starting data collection...")

    try:
        historical_data, spa_features, standings = collect_fastf1_data_for_prediction()

        if historical_data is not None:
            print("\n BELGIAN GP 2025 INSIGHTS:")

            # Show current championship leaders
            if not standings.empty:
                print("\n🏆 Championship leaders (standard F1 points):")
                top_5 = standings.head(5)
                for _, driver in top_5.iterrows():
                    print(f"   {driver['championship_position']}. {driver['driver_name']} ({driver['team_name']}) - "
                          f"{driver['points']} pts, {driver['wins']} wins")

            # Show drivers with best Spa history
            if not spa_features.empty:
                print("\n🏆 Best historical Spa performers:")
                top_spa = spa_features.nsmallest(5, 'spa_avg_qualifying_position')
                for _, driver in top_spa.iterrows():
                    print(f"   {driver['driver_name']}: Avg pos {driver['spa_avg_qualifying_position']:.1f}, "
                          f"{driver['spa_poles']} poles")

            # Show recent form leaders
            if not historical_data.empty:
                recent_form = historical_data[historical_data['year'] == 2025].groupby('driver_name').agg({
                    'recent_form_score': 'last',
                    'position': 'mean'
                }).sort_values('recent_form_score', ascending=False).head(5)

                print(f"\n📈 Best 2025 qualifying form:")
                for driver, stats in recent_form.iterrows():
                    print(f"   {driver}: Form score {stats['recent_form_score']:.3f}, "
                          f"Avg pos {stats['position']:.1f}")

            print(f"\n Ready to train prediction model with real F1 data!")
            print(f" Championship standings calculated using standard F1 points system:")
            print(f"   1st: 25 pts, 2nd: 18 pts, 3rd: 15 pts, etc.")

        else:
            print(" Data collection failed. Check installation and internet connection.")

    except Exception as e:
        print(f"  Error: {str(e)}")
        print(" Make sure you have internet connection and run: pip install fastf1")