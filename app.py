import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

class HostelAI:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_columns = None
        self.label_encoders = {}
        self.scaler = None
        self.best_params = {}
        
    def load_and_prepare_data(self, filepath="Lira_University_Hostel_Dataset.xlsx"):
        """Load and prepare data with advanced preprocessing"""
        # Load data
        df = pd.read_excel(filepath)
        
        # Clean budget
        df["Budget (UGX/sem)"] = (
            df["Budget (UGX/sem)"]
            .astype(str)
            .str.replace(",", "")
            .str.replace("UGX", "")
            .str.strip()
        )
        df["Budget (UGX/sem)"] = pd.to_numeric(df["Budget (UGX/sem)"], errors='coerce')
        
        # Fill missing values
        df["Kitchen"] = df["Kitchen"].fillna(df["Kitchen"].mode()[0])
        df["Water"] = df["Water"].fillna("Always Available")
        df["Security"] = df["Security"].fillna("Basic")
        
        return df
    
    def engineer_features(self, df):
        """Create advanced features for better model performance"""
        df = df.copy()
        
        # 1. Budget efficiency (budget per room type)
        room_budget_map = {
            'Single': df[df['Room Type'] == 'Single']['Budget (UGX/sem)'].mean(),
            'Double': df[df['Room Type'] == 'Double']['Budget (UGX/sem)'].mean(),
            'Triple': df[df['Room Type'] == 'Triple']['Budget (UGX/sem)'].mean(),
            'Quad': df[df['Room Type'] == 'Quad']['Budget (UGX/sem)'].mean()
        }
        df['Budget_Efficiency'] = df['Budget (UGX/sem)'] / df['Room Type'].map(room_budget_map)
        
        # 2. Comfort score (combination of amenities)
        comfort_factors = {
            'WiFi': {'Yes': 1, 'No': 0},
            'Bathroom': {'Private': 1, 'Shared': 0},
            'Kitchen': {'Private': 1, 'Shared': 0}
        }
        
        comfort_score = 0
        for factor, mapping in comfort_factors.items():
            if factor in df.columns:
                comfort_score += df[factor].map(mapping)
        df['Comfort_Score'] = comfort_score / len(comfort_factors)
        
        # 3. Security index
        security_map = {
            '24/7 Guard + CCTV': 4,
            'Security Guard': 3,
            'Gated Only': 2,
            'Basic': 1
        }
        df['Security_Index'] = df['Security'].map(security_map)
        
        # 4. Water reliability
        water_map = {
            'Always Available': 3,
            'Sometimes Interrupted': 2,
            'Irregular': 1
        }
        df['Water_Reliability'] = df['Water'].map(water_map)
        
        # 5. Distance score (inverse of distance)
        df['Distance_Score'] = 1 / (df['Distance (km)'] + 0.1)
        
        # 6. Value for money (budget / combined features)
        df['Value_Score'] = (
            df['Comfort_Score'] + 
            (df['Security_Index'] / 4) + 
            (df['Water_Reliability'] / 3)
        ) / (df['Budget (UGX/sem)'] / 100000)
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        # Select features
        feature_cols = [
            'Budget (UGX/sem)',
            'Distance (km)',
            'Gender',
            'WiFi',
            'Water',
            'Security',
            'Room Type',
            'Bathroom',
            'Kitchen',
            'Budget_Efficiency',
            'Comfort_Score',
            'Security_Index',
            'Water_Reliability',
            'Distance_Score',
            'Value_Score'
        ]
        
        # Use only available columns
        self.feature_columns = [col for col in feature_cols if col in df.columns]
        
        # Separate features and target (using target as recommendation score)
        # Generate recommendation score based on features
        X = df[self.feature_columns].copy()
        
        # Create target variable (synthetic recommendation score)
        # Higher score = better hostel
        y = (
            (X['Comfort_Score'] if 'Comfort_Score' in X else 0) * 3 +
            (X['Security_Index'] if 'Security_Index' in X else 0) * 2 +
            (X['Water_Reliability'] if 'Water_Reliability' in X else 0) * 1.5 +
            (X['Distance_Score'] if 'Distance_Score' in X else 0) * 2 +
            (X['Value_Score'] if 'Value_Score' in X else 0) * 2
        )
        
        # Normalize to 1-5 scale
        y = (y - y.min()) / (y.max() - y.min()) * 4 + 1
        
        return X, y
    
    def create_preprocessor(self):
        """Create preprocessing pipeline"""
        # Define columns by type
        numerical_cols = [
            'Budget (UGX/sem)', 'Distance (km)',
            'Budget_Efficiency', 'Comfort_Score', 
            'Security_Index', 'Water_Reliability', 
            'Distance_Score', 'Value_Score'
        ]
        categorical_cols = [
            'Gender', 'WiFi', 'Water', 'Security', 
            'Room Type', 'Bathroom', 'Kitchen'
        ]
        
        # Filter to available columns
        numerical_cols = [col for col in numerical_cols if col in self.feature_columns]
        categorical_cols = [col for col in categorical_cols if col in self.feature_columns]
        
        # Preprocessors
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', LabelEncoder())
        ])
        
        # Combined preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        return preprocessor
    
    def train_model(self, filepath="Lira_University_Hostel_Dataset.xlsx"):
        """Train enhanced AI model"""
        # Load and prepare data
        df = self.load_and_prepare_data(filepath)
        df = self.engineer_features(df)
        
        # Prepare features and target
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create preprocessor
        self.preprocessor = self.create_preprocessor()
        
        # Try multiple models
        models = {
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        best_model = None
        best_score = -float('inf')
        best_model_name = ''
        
        for name, model in models.items():
            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', model)
            ])
            
            # Cross validation
            scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
            mean_score = scores.mean()
            
            if mean_score > best_score:
                best_score = mean_score
                best_model = pipeline
                best_model_name = name
        
        # Hyperparameter tuning for best model
        if best_model_name in ['Random Forest', 'Gradient Boosting']:
            self.best_params = self.hyperparameter_tuning(
                X_train, y_train, best_model_name
            )
            
            # Retrain with best parameters
            if best_model_name == 'Random Forest':
                model = RandomForestRegressor(**self.best_params, random_state=42)
            else:
                model = GradientBoostingRegressor(**self.best_params, random_state=42)
            
            self.model = Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', model)
            ])
            self.model.fit(X_train, y_train)
        else:
            self.model = best_model
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'R² Score': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'Best Model': best_model_name,
            'Best CV Score': best_score,
            'Best Params': self.best_params
        }
        
        print("Model Training Complete!")
        print("=" * 50)
        for key, value in metrics.items():
            print(f"{key}: {value}")
        print("=" * 50)
        
        return metrics
    
    def hyperparameter_tuning(self, X_train, y_train, model_type):
        """Perform hyperparameter tuning"""
        param_grid = {}
        
        if model_type == 'Random Forest':
            param_grid = {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__max_depth': [None, 10, 20, 30],
                'regressor__min_samples_split': [2, 5, 10],
                'regressor__min_samples_leaf': [1, 2, 4]
            }
        elif model_type == 'Gradient Boosting':
            param_grid = {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__learning_rate': [0.01, 0.05, 0.1],
                'regressor__max_depth': [3, 4, 5],
                'regressor__min_samples_split': [2, 5]
            }
        
        # Create pipeline with preprocessor
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))
        ])
        
        # Grid search
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=5, 
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_params_
    
    def predict(self, user_input):
        """Predict recommendation score for user"""
        if self.model is None:
            raise ValueError("Model not trained! Call train_model() first.")
        
        # Ensure input has all required features
        input_df = pd.DataFrame([user_input])
        
        # Add engineered features
        input_df = self.engineer_features(input_df)
        
        # Select only features used in training
        X_input = input_df[self.feature_columns]
        
        # Predict
        prediction = self.model.predict(X_input)[0]
        
        # Clip to 1-5 range
        return max(1, min(5, prediction))
    
    def save_model(self, filepath="hostel_ai_model.pkl"):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save! Train first.")
        
        # Package model with metadata
        model_package = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_columns': self.feature_columns,
            'best_params': self.best_params
        }
        
        joblib.dump(model_package, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath="hostel_ai_model.pkl"):
        """Load a trained model"""
        model_package = joblib.load(filepath)
        self.model = model_package['model']
        self.preprocessor = model_package['preprocessor']
        self.feature_columns = model_package['feature_columns']
        self.best_params = model_package.get('best_params', {})
        print(f"Model loaded from {filepath}")

# ---------------------------------------
# ENHANCED TRAINING SCRIPT
# ---------------------------------------
def train_enhanced_model():
    """Train and save the enhanced AI model"""
    print("Starting Enhanced AI Model Training...")
    print("=" * 60)
    
    # Initialize model
    hostel_ai = HostelAI()
    
    # Train with data
    metrics = hostel_ai.train_model("Lira_University_Hostel_Dataset.xlsx")
    
    # Save model
    hostel_ai.save_model("hostel_ai_model_enhanced.pkl")
    
    # Test prediction
    test_input = {
        'Budget (UGX/sem)': 350000,
        'Gender': 'Mixed',
        'Distance (km)': 0.5,
        'WiFi': 'Yes',
        'Water': 'Always Available',
        'Security': '24/7 Guard + CCTV',
        'Room Type': 'Single',
        'Bathroom': 'Private',
        'Kitchen': 'Private'
    }
    
    score = hostel_ai.predict(test_input)
    print(f"\nTest Prediction Score: {score:.2f}/5")
    
    return hostel_ai, metrics

# ---------------------------------------
# UPDATED STREAMLIT APP (WITHOUT EMOJIS)
# ---------------------------------------
def create_streamlit_app():
    """Updated Streamlit app that uses the enhanced model"""
    import streamlit as st
    
    st.set_page_config(page_title="Hostel AI", layout="wide")
    
    # Custom CSS
    st.markdown("""
    <style>
        .main { background: #f5f7fa; }
        .block-container { padding-top: 2rem; }
        h1 { color: #003366; }
        .card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.title("Lira University Hostel AI")
    st.write("AI-powered hostel recommendation system")
    
    # Load enhanced model
    try:
        model = HostelAI()
        model.load_model("hostel_ai_model_enhanced.pkl")
        
        # Sidebar inputs
        st.sidebar.header("Preferences")
        
        budget = st.sidebar.number_input(
            "Budget (UGX/semester)",
            min_value=150000,
            max_value=1000000,
            value=300000,
            step=10000
        )
        
        gender = st.sidebar.selectbox(
            "Gender",
            ["Mixed", "Female Only", "Male Only"]
        )
        
        distance = st.sidebar.slider(
            "Distance (km)",
            0.1, 5.0, 1.0
        )
        
        wifi = st.sidebar.selectbox("WiFi", ["Yes", "No"])
        water = st.sidebar.selectbox("Water", ["Always Available", "Sometimes Interrupted", "Irregular"])
        security = st.sidebar.selectbox("Security", ["24/7 Guard + CCTV", "Security Guard", "Gated Only", "Basic"])
        room_type = st.sidebar.selectbox("Room Type", ["Single", "Double", "Triple", "Quad"])
        bathroom = st.sidebar.selectbox("Bathroom", ["Private", "Shared"])
        kitchen = st.sidebar.selectbox("Kitchen", ["Private", "Shared"])
        
        # Predict button
        if st.sidebar.button("Get Recommendation"):
            # Prepare input
            user_input = {
                'Budget (UGX/sem)': budget,
                'Gender': gender,
                'Distance (km)': distance,
                'WiFi': wifi,
                'Water': water,
                'Security': security,
                'Room Type': room_type,
                'Bathroom': bathroom,
                'Kitchen': kitchen
            }
            
            # Get prediction
            score = model.predict(user_input)
            
            # Display results
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Recommendation Score")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"<h1 style='text-align: center;'>{score:.2f}</h1>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;'>out of 5.0</p>", unsafe_allow_html=True)
                
                # Progress bar
                st.progress(score / 5)
                
                # Rating
                if score >= 4.0:
                    st.success("Excellent match for your preferences")
                elif score >= 3.0:
                    st.info("Good match for your preferences")
                else:
                    st.warning("Consider adjusting your preferences")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show feature importance
            with st.expander("Recommendation Breakdown"):
                st.write("""
                **Score Components:**
                - Budget compatibility
                - Distance from campus
                - Security level
                - Water reliability
                - Comfort amenities
                - Overall value
                """)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please train the model first by running train_enhanced_model()")

# ---------------------------------------
# MAIN EXECUTION
# ---------------------------------------
if __name__ == "__main__":
    # Train enhanced model
    train_enhanced_model()
    
    # Uncomment to run Streamlit app
    # create_streamlit_app()
