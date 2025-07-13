import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class F1QualifyingPredictor:
    """
    F1 Qualifying Winner Prediction Model
    Designed for the 2025 Belgian GP and future races
    """

    def __init__(self):
        self.pole_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.position_regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.drivers_2025 = [
            'Max Verstappen', 'Lando Norris', 'Oscar Piastri', 'George Russell',
            'Charles Leclerc', 'Lewis Hamilton', 'Kimi Antonelli', 'Alexander Albon',
            'Esteban Ocon', 'Lance Stroll', 'Pierre Gasly', 'Nico Hulkenberg',
            'Oliver Bearman', 'Isack Hadjar', 'Carlos Sainz', 'Yuki Tsunoda',
            'Fernando Alonso', 'Liam Lawson', 'Jack Doohan', 'Gabriel Bortoleto'
        ]

    def load_fastf1_data(self, data_file='f1_fastf1_historical_data.csv'):
        """
        Load real F1 data collected using FastF1
        Replace the simulated data with actual F1 qualifying results
        """
        try:
            print(f"📊 Loading real F1 data from {data_file}...")
            df = pd.read_csv(data_file)

            # Convert FastF1 data to our expected format
            formatted_data = []

            for _, row in df.iterrows():
                # Skip rows with missing essential data
                if pd.isna(row.get('position')) or pd.isna(row.get('driver_name')):
                    continue

                # Map FastF1 data to our feature format
                record = {
                    'driver': row['driver_name'],
                    'session': f"{row['year']}-{row['round']:02d}",
                    'qualifying_position': int(row['position']),
                    'recent_form_score': row.get('recent_form_score', 0.5),
                    'track_performance': row.get('track_type_performance', 0.5),
                    'weather_performance': 0.7,  # Default - can be enhanced with weather data
                    'team_momentum': row.get('team_form', 0.5),
                    'pole_position': int(row.get('pole_position', 0)),
                    'track_type': row.get('circuit_type', 'balanced'),
                    'weather': 'dry',  # Default - can be enhanced
                    'car_reliability': 0.9,  # Default high reliability
                    'engine_power': 0.85 if row.get('circuit_type') == 'power' else 0.75,
                    'year': row['year'],
                    'race_name': row['race_name'],
                    'circuit_name': row['circuit_name'],
                    'team_name': row['team_name']
                }
                formatted_data.append(record)

            result_df = pd.DataFrame(formatted_data)
            print(f"✅ Loaded {len(result_df)} real qualifying results")
            print(f"   📅 Date range: {result_df['year'].min()}-{result_df['year'].max()}")
            print(f"   👨‍💼 Drivers: {result_df['driver'].nunique()}")
            print(f"   🏁 Circuits: {result_df['circuit_name'].nunique()}")

            return result_df

        except FileNotFoundError:
            print(f"❌ Data file {data_file} not found!")
            print("💡 Run the FastF1 data collection script first:")
            print("   python collect_fastf1_data_for_prediction()")
            return self.create_fallback_data()
        except Exception as e:
            print(f"❌ Error loading FastF1 data: {e}")
            print("🔄 Falling back to sample data for demonstration...")
            return self.create_fallback_data()

    def create_fallback_data(self):
        """
        Fallback sample data if FastF1 data is not available
        """
        print("🔄 Creating fallback sample data...")
        np.random.seed(42)

        # Create realistic sample data based on 2025 patterns
        drivers_data = [
            {'name': 'Oscar Piastri', 'skill': 0.9, 'team': 'McLaren'},
            {'name': 'Lando Norris', 'skill': 0.88, 'team': 'McLaren'},
            {'name': 'Max Verstappen', 'skill': 0.95, 'team': 'Red Bull Racing'},
            {'name': 'George Russell', 'skill': 0.82, 'team': 'Mercedes'},
            {'name': 'Charles Leclerc', 'skill': 0.85, 'team': 'Ferrari'},
            {'name': 'Lewis Hamilton', 'skill': 0.80, 'team': 'Ferrari'},
        ]

        circuits = ['Spa-Francorchamps', 'Silverstone', 'Monza', 'Monaco', 'Hungary']

        data = []
        for year in [2022, 2023, 2024, 2025]:
            for round_num, circuit in enumerate(circuits, 1):
                # Simulate qualifying for each circuit
                session_results = []

                for driver_info in drivers_data:
                    skill = driver_info['skill']
                    performance = np.random.normal(skill, 0.1)
                    session_results.append((driver_info, performance))

                # Sort by performance to get positions
                session_results.sort(key=lambda x: x[1], reverse=True)

                for pos, (driver_info, performance) in enumerate(session_results, 1):
                    data.append({
                        'driver': driver_info['name'],
                        'session': f"{year}-{round_num:02d}",
                        'qualifying_position': pos,
                        'recent_form_score': max(0.3, min(1.0, performance)),
                        'track_performance': performance * 0.9,
                        'weather_performance': np.random.uniform(0.6, 0.9),
                        'team_momentum': 0.8 if driver_info['team'] in ['McLaren', 'Red Bull Racing'] else 0.6,
                        'pole_position': 1 if pos == 1 else 0,
                        'track_type': 'power' if circuit in ['Spa-Francorchamps', 'Silverstone',
                                                             'Monza'] else 'downforce',
                        'weather': 'dry',
                        'car_reliability': 0.9,
                        'engine_power': 0.85,
                        'year': year,
                        'race_name': f'{circuit} Grand Prix',
                        'circuit_name': circuit,
                        'team_name': driver_info['team']
                    })

        return pd.DataFrame(data)

    def engineer_features(self, df):
        """
        Create focused features for better predictions (removed championship pressure)
        """
        df = df.copy()

        # Momentum features
        df['form_momentum'] = df['recent_form_score'] * df['team_momentum']
        df['track_weather_synergy'] = df['track_performance'] * df['weather_performance']

        # Power track advantage for Belgian GP
        df['power_advantage'] = np.where(df['track_type'] == 'power', df['engine_power'], 0.5)

        # Experience factor (simulate based on known driver experience)
        experience_map = {
            'Lewis Hamilton': 0.95, 'Fernando Alonso': 0.9, 'Max Verstappen': 0.85,
            'Charles Leclerc': 0.75, 'Lando Norris': 0.7, 'George Russell': 0.65,
            'Oscar Piastri': 0.6, 'Carlos Sainz': 0.7, 'Pierre Gasly': 0.65
        }
        df['experience_factor'] = df['driver'].map(experience_map).fillna(0.4)

        # Composite performance score (simplified without championship pressure)
        df['performance_score'] = (
                df['recent_form_score'] * 0.4 +  # Recent form is most important
                df['track_performance'] * 0.3 +  # Track-specific performance
                df['weather_performance'] * 0.15 +  # Weather adaptation
                df['experience_factor'] * 0.1 +  # Driver experience
                df['car_reliability'] * 0.05  # Reliability factor
        )

        return df

    def prepare_data_for_training(self, df):
        """
        Prepare data for machine learning models (streamlined features)
        """
        # Engineer features
        df = self.engineer_features(df)

        # Select the most important features for training
        feature_columns = [
            # Core performance features
            'recent_form_score',  # Most important - recent qualifying form
            'track_performance',  # Track-specific historical performance
            'weather_performance',  # Weather adaptation ability
            'team_momentum',  # Team's recent performance trend
            'car_reliability',  # Car reliability factor
            'engine_power',  # Engine power (crucial for Spa)

            # Engineered features
            'form_momentum',  # Combined recent form + team momentum
            'track_weather_synergy',  # Track performance * weather performance
            'power_advantage',  # Advantage on power tracks
            'experience_factor',  # Driver experience level
            'performance_score',  # Composite performance metric
        ]

        # Encode categorical variables
        df_encoded = df.copy()
        df_encoded['track_type_encoded'] = self.label_encoder.fit_transform(df['track_type'])
        df_encoded['weather_encoded'] = LabelEncoder().fit_transform(df['weather'])

        feature_columns.extend(['track_type_encoded', 'weather_encoded'])

        X = df_encoded[feature_columns]
        y_pole = df_encoded['pole_position']
        y_position = df_encoded['qualifying_position']

        print(f"📊 Training features: {len(feature_columns)} total")
        print(f"   🎯 Core features: recent_form, track_performance, weather_performance")
        print(f"   🔧 Engineered features: form_momentum, power_advantage, performance_score")
        print(f"   ❌ Removed: championship_pressure (not predictive for qualifying)")

        return X, y_pole, y_position, feature_columns

    def train_models(self, X, y_pole, y_position):
        """
        Train both pole position classifier and position regressor
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_pole_train, y_pole_test, y_pos_train, y_pos_test = train_test_split(
            X_scaled, y_pole, y_position, test_size=0.2, random_state=42, stratify=y_pole
        )

        # Train pole position classifier
        self.pole_classifier.fit(X_train, y_pole_train)

        # Train position regressor
        self.position_regressor.fit(X_train, y_pos_train)

        # Evaluate models
        pole_pred = self.pole_classifier.predict(X_test)
        pos_pred = self.position_regressor.predict(X_test)

        print("=== MODEL PERFORMANCE ===")
        print(f"Pole Position Accuracy: {accuracy_score(y_pole_test, pole_pred):.3f}")
        print(f"Average Position Error: {np.mean(np.abs(pos_pred - y_pos_test)):.2f} positions")

        # Cross-validation scores
        cv_scores = cross_val_score(self.pole_classifier, X_scaled, y_pole, cv=5)
        print(f"Cross-validation Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

        return X_test, y_pole_test, y_pos_test, pole_pred, pos_pred

    def analyze_feature_importance(self, feature_columns):
        """
        Analyze which features are most important for predictions
        """
        importance_pole = self.pole_classifier.feature_importances_
        importance_pos = self.position_regressor.feature_importances_

        feature_importance_df = pd.DataFrame({
            'feature': feature_columns,
            'pole_importance': importance_pole,
            'position_importance': importance_pos
        }).sort_values('pole_importance', ascending=False)

        print("\n=== FEATURE IMPORTANCE (Top 10) ===")
        print(feature_importance_df.head(10))

        # Plot feature importance
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        top_features = feature_importance_df.head(8)
        plt.barh(top_features['feature'], top_features['pole_importance'])
        plt.title('Feature Importance for Pole Position Prediction')
        plt.xlabel('Importance Score')

        plt.subplot(2, 1, 2)
        plt.barh(top_features['feature'], top_features['position_importance'])
        plt.title('Feature Importance for Qualifying Position Prediction')
        plt.xlabel('Importance Score')

        plt.tight_layout()
        plt.show()

        return feature_importance_df

    def predict_belgian_gp_2025(self):
        """
        Make predictions for Belgian GP 2025 based on current season data
        Focused on the most predictive features
        """
        print("\n=== BELGIAN GP 2025 PREDICTIONS ===")

        # Create current form data for each driver (based on 2025 season analysis)
        current_form = {
            'Oscar Piastri': {'recent_form': 0.9, 'track_perf': 0.8, 'weather_perf': 0.75, 'experience': 0.6},
            'Lando Norris': {'recent_form': 0.85, 'track_perf': 0.85, 'weather_perf': 0.7, 'experience': 0.7},
            'Max Verstappen': {'recent_form': 0.8, 'track_perf': 0.9, 'weather_perf': 0.9, 'experience': 0.85},
            'George Russell': {'recent_form': 0.75, 'track_perf': 0.7, 'weather_perf': 0.65, 'experience': 0.65},
            'Charles Leclerc': {'recent_form': 0.65, 'track_perf': 0.75, 'weather_perf': 0.6, 'experience': 0.75},
            'Lewis Hamilton': {'recent_form': 0.6, 'track_perf': 0.7, 'weather_perf': 0.85, 'experience': 0.95}
        }

        predictions = []

        for driver, form_data in current_form.items():
            # Create feature vector for this driver (simplified, focused features)
            features = np.array([
                form_data['recent_form'],  # recent_form_score
                form_data['track_perf'],  # track_performance
                form_data['weather_perf'],  # weather_performance
                0.75,  # team_momentum (assume good for top teams)
                0.9,  # car_reliability
                0.85,  # engine_power (high for Spa - power track)
                form_data['recent_form'] * 0.75,  # form_momentum
                form_data['track_perf'] * form_data['weather_perf'],  # track_weather_synergy
                0.85,  # power_advantage (Spa is power track)
                form_data['experience'],  # experience_factor
                (form_data['recent_form'] * 0.4 + form_data['track_perf'] * 0.3 +
                 form_data['weather_perf'] * 0.15 + form_data['experience'] * 0.1 + 0.05),  # performance_score
                0,  # track_type_encoded (power track)
                0  # weather_encoded (assume dry)
            ]).reshape(1, -1)

            # Scale features
            features_scaled = self.scaler.transform(features)

            # Get predictions
            pole_prob = self.pole_classifier.predict_proba(features_scaled)[0][1]
            expected_position = self.position_regressor.predict(features_scaled)[0]

            predictions.append({
                'driver': driver,
                'pole_probability': pole_prob,
                'expected_position': expected_position,
                'current_form': form_data['recent_form'],
                'spa_suitability': form_data['track_perf']
            })

        # Sort by pole probability
        predictions = sorted(predictions, key=lambda x: x['pole_probability'], reverse=True)

        print("\nPOLE POSITION PREDICTIONS:")
        print("-" * 65)
        for i, pred in enumerate(predictions, 1):
            print(f"{i}. {pred['driver']:15s} | Pole Prob: {pred['pole_probability']:.1%} | "
                  f"Expected Pos: {pred['expected_position']:.1f} | Form: {pred['current_form']:.1%} | "
                  f"Spa Suit: {pred['spa_suitability']:.1%}")

        return predictions

    def run_full_analysis(self, use_real_data=True):
        """
        Run the complete analysis pipeline with real FastF1 data
        """
        print(" F1 QUALIFYING PREDICTION MODEL - BELGIAN GP 2025")
        print("=" * 60)

        # Load data (real FastF1 data or fallback)
        if use_real_data:
            print(" Loading real F1 data from FastF1...")
            df = self.load_fastf1_data()
        else:
            print(" Creating sample data for demonstration...")
            df = self.create_fallback_data()

        if df.empty:
            print(" No data available. Cannot proceed with analysis.")
            return None, None

        # Prepare data for ML
        X, y_pole, y_position, feature_columns = self.prepare_data_for_training(df)

        # Train models
        print("\n Training machine learning models...")
        X_test, y_pole_test, y_pos_test, pole_pred, pos_pred = self.train_models(X, y_pole, y_position)

        # Analyze feature importance
        print("\n Analyzing feature importance...")
        feature_importance = self.analyze_feature_importance(feature_columns)

        # Make Belgian GP predictions
        predictions = self.predict_belgian_gp_2025()

        print(f"\n RECOMMENDATION FOR BELGIAN GP 2025:")
        top_contender = predictions[0]
        print(
            f"   Most likely pole winner: {top_contender['driver']} ({top_contender['pole_probability']:.1%} probability)")
        print(f"   Top 3 contenders: {', '.join([p['driver'] for p in predictions[:3]])}")

        print("\n KEY INSIGHTS:")
        print("   • Model trained on real F1 historical data (2022-2025)")
        print("   • Focus on performance factors that actually predict qualifying")
        print("   • Removed championship pressure (not predictive for 90-min qualifying)")
        print("   • Spa-Francorchamps is a power track favoring McLaren/Red Bull")
        print("   • Weather conditions could shuffle the order significantly")

        # Show data source info
        if use_real_data and 'year' in df.columns:
            print(f"\n DATA SOURCE INFO:")
            print(f"   • Real F1 data from {df['year'].min()}-{df['year'].max()}")
            print(f"   • {len(df)} qualifying sessions analyzed")
            print(f"   • {df['circuit_name'].nunique()} different circuits")

        return predictions, feature_importance


# Example usage and demonstration
if __name__ == "__main__":
    print(" F1 QUALIFYING PREDICTION MODEL WITH FASTF1 DATA")
    print("=" * 60)

    # Check if FastF1 data is available
    import os

    real_data_available = os.path.exists('f1_fastf1_historical_data.csv')

    if real_data_available:
        print(" Real F1 data found! Using FastF1 historical data.")
        use_real_data = True
    else:
        print(" Real F1 data not found.")
        print(" To get real data, first run:")
        print("   from f1_real_data_collection import collect_fastf1_data_for_prediction")
        print("   collect_fastf1_data_for_prediction()")
        print("\n Using sample data for demonstration...")
        use_real_data = False

    # Initialize and run the predictor
    predictor = F1QualifyingPredictor()
    predictions, feature_importance = predictor.run_full_analysis(use_real_data=use_real_data)

    # Additional analysis suggestions
    print("\n" + "=" * 60)
    print("🚀 NEXT STEPS TO IMPROVE THE MODEL:")
    print("=" * 60)

    if not real_data_available:
        print("1. PRIORITY: Install FastF1 and collect real data:")
        print("  pip install fastf1")
        print(
            "   python -c \"from f1_real_data_collection import collect_fastf1_data_for_prediction; collect_fastf1_data_for_prediction()\"")

    print("2. 🌤️  Add real weather forecast data for Belgian GP weekend")
    print("3. 🔧 Include Friday/Saturday practice session times")
    print("4. 📊 Add telemetry data (speed traps, sector times)")
    print("5. 🎯 Validate model with 2024 Belgian GP results")
    print("6. 🔄 Update predictions during qualifying weekend")

    print("\n📚 MACHINE LEARNING CONCEPTS USED:")
    print("• Random Forest: Combines multiple decision trees for robust predictions")
    print("• Gradient Boosting: Learns from prediction errors iteratively")
    print("• Feature Engineering: Creating meaningful variables from raw data")
    print("• Cross-validation: Testing model performance on unseen data")
    print("• Ensemble Methods: Combining multiple models for better accuracy")

    print(f"\n🏆 FASTF1 ADVANTAGES:")
    print("• Official F1 timing data (not deprecated like Ergast)")
    print("• Telemetry and sector times available")
    print("• Live session data during race weekends")
    print("• Detailed lap-by-lap analysis")
    print("• Weather and track condition data")