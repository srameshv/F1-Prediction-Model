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

    def create_sample_data(self):
        """
        Create sample training data based on 2025 season patterns
        In practice, you'd load this from actual F1 databases
        """
        np.random.seed(42)

        # Simulate 200 qualifying sessions (2020-2025 data)
        n_sessions = 200
        data = []

        # Driver performance tiers based on 2025 championship
        tier_1 = ['Oscar Piastri', 'Lando Norris', 'Max Verstappen']  # Championship contenders
        tier_2 = ['George Russell', 'Charles Leclerc', 'Lewis Hamilton']  # Regular podium contenders
        tier_3 = ['Kimi Antonelli', 'Alexander Albon', 'Carlos Sainz']  # Midfield leaders

        for session in range(n_sessions):
            # Simulate each driver's performance for this session
            for i, driver in enumerate(self.drivers_2025):

                # Base performance based on tier
                if driver in tier_1:
                    base_performance = np.random.normal(0.8, 0.15)  # High performance
                    recent_form = np.random.normal(0.85, 0.1)
                elif driver in tier_2:
                    base_performance = np.random.normal(0.6, 0.2)  # Good performance
                    recent_form = np.random.normal(0.65, 0.15)
                else:
                    base_performance = np.random.normal(0.4, 0.25)  # Variable performance
                    recent_form = np.random.normal(0.45, 0.2)

                # Simulate features
                qualifying_position = max(1, min(20, int(np.random.exponential(5) + 1)))

                # Track-specific performance (some drivers better at power tracks)
                track_type = np.random.choice(['power', 'downforce', 'mixed'])
                if track_type == 'power' and driver in ['Max Verstappen', 'Lando Norris']:
                    track_performance = np.random.normal(0.9, 0.1)
                elif track_type == 'downforce' and driver in ['Charles Leclerc', 'Oscar Piastri']:
                    track_performance = np.random.normal(0.85, 0.1)
                else:
                    track_performance = np.random.normal(0.6, 0.2)

                # Weather impact
                weather = np.random.choice(['dry', 'wet', 'mixed'])
                if weather == 'wet' and driver in ['Max Verstappen', 'Lewis Hamilton']:
                    weather_performance = np.random.normal(0.9, 0.1)
                else:
                    weather_performance = np.random.normal(0.6, 0.2)

                data.append({
                    'driver': driver,
                    'session': session,
                    'qualifying_position': qualifying_position,
                    'recent_form_score': max(0, min(1, recent_form)),
                    'track_performance': max(0, min(1, track_performance)),
                    'weather_performance': max(0, min(1, weather_performance)),
                    'championship_position': min(20, np.random.poisson(8) + 1),
                    'team_momentum': np.random.normal(0.5, 0.2),
                    'pole_position': 1 if qualifying_position == 1 else 0,
                    'track_type': track_type,
                    'weather': weather,
                    'car_reliability': np.random.uniform(0.7, 1.0),
                    'engine_power': np.random.uniform(0.6, 1.0)
                })

        return pd.DataFrame(data)

    def engineer_features(self, df):
        """
        Create advanced features for better predictions
        """
        df = df.copy()

        # Momentum features
        df['form_momentum'] = df['recent_form_score'] * df['team_momentum']
        df['track_weather_synergy'] = df['track_performance'] * df['weather_performance']

        # Championship pressure (inverse of position - leaders have more pressure)
        df['championship_pressure'] = 1 / (df['championship_position'] + 1)

        # Power track advantage
        df['power_advantage'] = np.where(df['track_type'] == 'power', df['engine_power'], 0.5)

        # Experience factor (simulate based on known driver experience)
        experience_map = {
            'Lewis Hamilton': 0.95, 'Fernando Alonso': 0.9, 'Max Verstappen': 0.85,
            'Charles Leclerc': 0.75, 'Lando Norris': 0.7, 'George Russell': 0.65,
            'Oscar Piastri': 0.6, 'Carlos Sainz': 0.7, 'Pierre Gasly': 0.65
        }
        df['experience_factor'] = df['driver'].map(experience_map).fillna(0.4)

        # Composite performance score
        df['performance_score'] = (
                df['recent_form_score'] * 0.3 +
                df['track_performance'] * 0.25 +
                df['weather_performance'] * 0.2 +
                df['experience_factor'] * 0.15 +
                df['car_reliability'] * 0.1
        )

        return df

    def prepare_data_for_training(self, df):
        """
        Prepare data for machine learning models
        """
        # Engineer features
        df = self.engineer_features(df)

        # Select features for training
        feature_columns = [
            'recent_form_score', 'track_performance', 'weather_performance',
            'championship_position', 'team_momentum', 'car_reliability',
            'engine_power', 'form_momentum', 'track_weather_synergy',
            'championship_pressure', 'power_advantage', 'experience_factor',
            'performance_score'
        ]

        # Encode categorical variables
        df_encoded = df.copy()
        df_encoded['track_type_encoded'] = self.label_encoder.fit_transform(df['track_type'])
        df_encoded['weather_encoded'] = LabelEncoder().fit_transform(df['weather'])

        feature_columns.extend(['track_type_encoded', 'weather_encoded'])

        X = df_encoded[feature_columns]
        y_pole = df_encoded['pole_position']
        y_position = df_encoded['qualifying_position']

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
        """
        print("\n=== BELGIAN GP 2025 PREDICTIONS ===")

        # Create current form data for each driver (based on 2025 season analysis)
        current_form = {
            'Oscar Piastri': {'recent_form': 0.9, 'track_perf': 0.8, 'weather_perf': 0.75},
            'Lando Norris': {'recent_form': 0.85, 'track_perf': 0.85, 'weather_perf': 0.7},
            'Max Verstappen': {'recent_form': 0.8, 'track_perf': 0.9, 'weather_perf': 0.9},
            'George Russell': {'recent_form': 0.75, 'track_perf': 0.7, 'weather_perf': 0.65},
            'Charles Leclerc': {'recent_form': 0.65, 'track_perf': 0.75, 'weather_perf': 0.6},
            'Lewis Hamilton': {'recent_form': 0.6, 'track_perf': 0.7, 'weather_perf': 0.85}
        }

        predictions = []

        for driver, form_data in current_form.items():
            # Create feature vector for this driver
            features = np.array([
                form_data['recent_form'],  # recent_form_score
                form_data['track_perf'],  # track_performance
                form_data['weather_perf'],  # weather_performance
                list(current_form.keys()).index(driver) + 1,  # championship_position (approximate)
                0.75,  # team_momentum (assume good for top teams)
                0.9,  # car_reliability
                0.85,  # engine_power (high for Spa)
                form_data['recent_form'] * 0.75,  # form_momentum
                form_data['track_perf'] * form_data['weather_perf'],  # track_weather_synergy
                1 / (list(current_form.keys()).index(driver) + 2),  # championship_pressure
                0.85,  # power_advantage (Spa is power track)
                0.8,  # experience_factor
                (form_data['recent_form'] + form_data['track_perf']) / 2,  # performance_score
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
                'current_form': form_data['recent_form']
            })

        # Sort by pole probability
        predictions = sorted(predictions, key=lambda x: x['pole_probability'], reverse=True)

        print("\nPOLE POSITION PREDICTIONS:")
        print("-" * 50)
        for i, pred in enumerate(predictions, 1):
            print(f"{i}. {pred['driver']:15s} | Pole Prob: {pred['pole_probability']:.1%} | "
                  f"Expected Pos: {pred['expected_position']:.1f} | Form: {pred['current_form']:.1%}")

        return predictions

    def run_full_analysis(self):
        """
        Run the complete analysis pipeline
        """
        print("🏎️  F1 QUALIFYING PREDICTION MODEL - BELGIAN GP 2025")
        print("=" * 60)

        # Create and prepare data
        print("📊 Creating training data...")
        df = self.create_sample_data()
        X, y_pole, y_position, feature_columns = self.prepare_data_for_training(df)

        # Train models
        print("\n🤖 Training machine learning models...")
        X_test, y_pole_test, y_pos_test, pole_pred, pos_pred = self.train_models(X, y_pole, y_position)

        # Analyze feature importance
        print("\n📈 Analyzing feature importance...")
        feature_importance = self.analyze_feature_importance(feature_columns)

        # Make Belgian GP predictions
        predictions = self.predict_belgian_gp_2025()

        print(f"\n🎯 RECOMMENDATION FOR BELGIAN GP 2025:")
        top_contender = predictions[0]
        print(
            f"   Most likely pole winner: {top_contender['driver']} ({top_contender['pole_probability']:.1%} probability)")
        print(f"   Top 3 contenders: {', '.join([p['driver'] for p in predictions[:3]])}")

        print("\n💡 KEY INSIGHTS:")
        print("   • McLaren drivers (Piastri/Norris) have strong current form")
        print("   • Verstappen remains dangerous at Spa historically")
        print("   • Weather conditions could shuffle the order significantly")
        print("   • Track characteristics favor cars with good power unit performance")

        return predictions, feature_importance


# Example usage and demonstration
if __name__ == "__main__":
    # Initialize and run the predictor
    predictor = F1QualifyingPredictor()
    predictions, feature_importance = predictor.run_full_analysis()

    # Additional analysis suggestions
    print("\n" + "=" * 60)
    print("🚀 NEXT STEPS TO IMPROVE THE MODEL:")
    print("=" * 60)
    print("1. 📥 Collect real historical data from Ergast API or F1 websites")
    print("2. 🌤️  Add real weather forecast data for race weekend")
    print("3. 🔧 Include practice session times for real-time updates")
    print("4. 📊 Add more sophisticated feature engineering")
    print("5. 🎯 Validate model with 2024 season data")
    print("6. 🔄 Update predictions during qualifying weekend")

    print("\n📚 MACHINE LEARNING CONCEPTS USED:")
    print("• Random Forest: Combines multiple decision trees for robust predictions")
    print("• Gradient Boosting: Learns from prediction errors iteratively")
    print("• Feature Engineering: Creating meaningful variables from raw data")
    print("• Cross-validation: Testing model performance on unseen data")
    print("• Ensemble Methods: Combining multiple models for better accuracy")