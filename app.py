import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------
# PAGE CONFIG
# ---------------------------------------
st.set_page_config(
    page_title="Lira University Hostel AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------
# CUSTOM CSS
# ---------------------------------------
st.markdown("""
<style>
.main {
    background: #f5f7fa;
}
.block-container {
    padding-top: 2rem;
}
h1 {
    color: #003366;
}
.card {
    background: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.metric-card {
    background: white;
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    text-align: center;
    margin: 10px 0;
}
.footer {
    text-align: center;
    color: #666;
    padding: 30px;
    font-size: 14px;
    border-top: 1px solid #e0e0e0;
    margin-top: 30px;
}
.badge {
    background: #003366;
    color: white;
    padding: 3px 10px;
    border-radius: 15px;
    font-size: 12px;
    display: inline-block;
    margin: 2px;
}
.score-bar {
    height: 6px;
    background: #e0e0e0;
    border-radius: 3px;
    margin: 8px 0;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #28a745, #003366);
    border-radius: 3px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------
# CACHE DATA LOADING
# ---------------------------------------
@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    try:
        df = pd.read_excel("Lira_University_Hostel_Dataset.xlsx")
        
        # Clean budget column
        df["Budget (UGX/sem)"] = (
            df["Budget (UGX/sem)"]
            .astype(str)
            .str.replace(",", "")
            .str.replace("UGX", "")
            .str.strip()
        )
        df["Budget (UGX/sem)"] = pd.to_numeric(df["Budget (UGX/sem)"], errors='coerce')
        
        # Fill missing values
        if "Kitchen" in df.columns:
            df["Kitchen"] = df["Kitchen"].fillna(df["Kitchen"].mode()[0])
        if "Water" in df.columns:
            df["Water"] = df["Water"].fillna("Always Available")
        if "Security" in df.columns:
            df["Security"] = df["Security"].fillna("Basic")
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        return joblib.load("hostel_ai_model.pkl")
    except:
        return None

# ---------------------------------------
# ENHANCED RECOMMENDATION ENGINE
# ---------------------------------------
class EnhancedRecommender:
    def __init__(self, df):
        self.df = df
        self.categorical_columns = ['Gender', 'WiFi', 'Water', 'Security', 'Room Type', 'Bathroom', 'Kitchen']
        self.numerical_columns = ['Budget (UGX/sem)', 'Distance (km)']
        self.label_encoders = {}
        self._prepare_features()
    
    def _prepare_features(self):
        """Prepare features for similarity calculations"""
        # Filter existing columns
        existing_cat = [col for col in self.categorical_columns if col in self.df.columns]
        self.categorical_columns = existing_cat
        
        # Encode categorical features
        for col in existing_cat:
            le = LabelEncoder()
            self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
        
        # Normalize numerical features
        self.numeric_stats = {}
        for col in self.numerical_columns:
            if col in self.df.columns:
                mean = self.df[col].mean()
                std = self.df[col].std()
                self.numeric_stats[col] = {'mean': mean, 'std': std}
                
                if std > 0:
                    self.df[f'{col}_normalized'] = (self.df[col] - mean) / std
                else:
                    self.df[f'{col}_normalized'] = 0
        
        # Create feature matrix
        feature_cols = []
        for col in self.numerical_columns:
            if f'{col}_normalized' in self.df.columns:
                feature_cols.append(f'{col}_normalized')
        
        for col in existing_cat:
            if f'{col}_encoded' in self.df.columns:
                feature_cols.append(f'{col}_encoded')
        
        self.feature_matrix = self.df[feature_cols].values
        self.feature_cols = feature_cols
    
    def get_recommendations(self, preferences, n=5):
        """Get top N recommendations"""
        # Create preference vector
        preference_vector = []
        
        # Numerical features
        for col in self.numerical_columns:
            if col in preferences and col in self.numeric_stats:
                stats = self.numeric_stats[col]
                if stats['std'] > 0:
                    norm_val = (preferences[col] - stats['mean']) / stats['std']
                else:
                    norm_val = 0
                preference_vector.append(norm_val)
        
        # Categorical features
        for col in self.categorical_columns:
            if col in preferences and col in self.label_encoders:
                try:
                    encoded = self.label_encoders[col].transform([preferences[col]])[0]
                    preference_vector.append(encoded)
                except:
                    preference_vector.append(0)
            else:
                preference_vector.append(0)
        
        preference_vector = np.array(preference_vector).reshape(1, -1)
        
        # Ensure dimensions match
        min_dim = min(preference_vector.shape[1], self.feature_matrix.shape[1])
        preference_vector = preference_vector[:, :min_dim]
        feature_matrix = self.feature_matrix[:, :min_dim]
        
        # Calculate similarities
        similarities = cosine_similarity(preference_vector, feature_matrix)[0]
        
        # Get top N
        top_indices = np.argsort(similarities)[::-1][:n]
        
        recommendations = self.df.iloc[top_indices].copy()
        recommendations['Match Score'] = similarities[top_indices] * 100
        
        # Calculate budget compatibility
        if 'Budget (UGX/sem)' in preferences and 'Budget (UGX/sem)' in recommendations.columns:
            budget_diff = abs(recommendations['Budget (UGX/sem)'] - preferences['budget'])
            max_diff = budget_diff.max() if budget_diff.max() > 0 else 1
            recommendations['Budget Match'] = (1 - budget_diff / max_diff) * 100
        
        # Calculate overall score
        recommendations['Overall Score'] = (
            recommendations['Match Score'] * 0.6 + 
            recommendations.get('Budget Match', recommendations['Match Score']) * 0.4
        )
        
        return recommendations.sort_values('Overall Score', ascending=False)

# ---------------------------------------
# MAIN APP
# ---------------------------------------
def main():
    # Load data and model
    df = load_data()
    if df is None:
        st.error("Failed to load data. Please check the dataset file.")
        return
    
    model = load_model()
    recommender = EnhancedRecommender(df)
    
    # Title
    st.title("Lira University Hostel Recommendation System")
    st.write("AI-powered recommendation system to find your ideal hostel")
    
    # Sidebar
    with st.sidebar:
        st.header("Student Preferences")
        st.markdown("---")
        
        budget = st.number_input(
            "Budget (UGX/semester)",
            min_value=150000,
            max_value=1000000,
            value=300000,
            step=10000
        )
        
        # Quick budget options
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Low", use_container_width=True):
                st.session_state.budget = 200000
        with col2:
            if st.button("Medium", use_container_width=True):
                st.session_state.budget = 350000
        with col3:
            if st.button("High", use_container_width=True):
                st.session_state.budget = 500000
        
        if 'budget' in st.session_state:
            budget = st.session_state.budget
        
        st.markdown("---")
        
        gender = st.selectbox(
            "Gender",
            ["Mixed", "Female Only", "Male Only"]
        )
        
        distance = st.slider(
            "Maximum Distance (km)",
            0.1, 5.0, 1.0, 0.1
        )
        
        wifi = st.selectbox("WiFi", ["Yes", "No"])
        water = st.selectbox("Water Availability", ["Always Available", "Sometimes Interrupted", "Irregular"])
        security = st.selectbox("Security", ["24/7 Guard + CCTV", "Security Guard", "Gated Only", "Basic"])
        room = st.selectbox("Room Type", ["Single", "Double", "Triple", "Quad"])
        bathroom = st.selectbox("Bathroom", ["Private", "Shared"])
        kitchen = st.selectbox("Kitchen", ["Private", "Shared"])
        
        st.markdown("---")
        num_recommendations = st.slider("Number of recommendations", 1, 10, 5)
        
        if st.button("Find Best Hostel", use_container_width=True):
            st.session_state.search = True
    
    # Main content
    if 'search' in st.session_state and st.session_state.search:
        # Prepare preferences
        preferences = {
            'budget': budget,
            'gender': gender,
            'distance': distance,
            'wifi': wifi,
            'water': water,
            'security': security,
            'room_type': room,
            'bathroom': bathroom,
            'kitchen': kitchen
        }
        
        # Get recommendations
        recommendations = recommender.get_recommendations(preferences, num_recommendations)
        
        # Display results
        display_recommendations(recommendations, preferences, model)
        
        st.session_state.search = False
    else:
        display_overview(df)

def display_recommendations(recommendations, preferences, model):
    """Display recommendations"""
    if len(recommendations) == 0:
        st.warning("No recommendations found. Please adjust your preferences.")
        return
    
    top_hostel = recommendations.iloc[0]
    
    # Get AI score from model if available
    if model is not None:
        try:
            input_df = pd.DataFrame([{
                "Budget (UGX/sem)": [preferences['budget']],
                "Gender": [preferences['gender']],
                "Distance (km)": [preferences['distance']],
                "WiFi": [preferences['wifi']],
                "Water": [preferences['water']],
                "Security": [preferences['security']],
                "Room Type": [preferences['room_type']],
                "Bathroom": [preferences['bathroom']],
                "Kitchen": [preferences['kitchen']]
            }])
            ai_score = model.predict(input_df)[0]
        except:
            ai_score = top_hostel['Overall Score'] / 20
    else:
        ai_score = top_hostel['Overall Score'] / 20
    
    st.success("Recommendation Generated Successfully")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #003366; margin: 0;">{top_hostel['Hostel']}</h3>
            <p style="color: #666; margin: 5px 0;">Top Pick</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #28a745; margin: 0;">{ai_score:.1f}</h2>
            <p style="color: #666; margin: 5px 0;">AI Score / 5</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #003366; margin: 0;">UGX {int(top_hostel['Budget (UGX/sem)']):,}</h3>
            <p style="color: #666; margin: 5px 0;">Budget</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #003366; margin: 0;">{top_hostel['Distance (km)']} km</h3>
            <p style="color: #666; margin: 5px 0;">Distance</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detailed recommendation
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class="card">
            <h3>{top_hostel['Hostel']}</h3>
            <p><strong>Match Score:</strong> {top_hostel['Match Score']:.1f}%</p>
            <div class="score-bar">
                <div class="score-bar-fill" style="width: {top_hostel['Match Score']:.1f}%;"></div>
            </div>
            <p><strong>Budget:</strong> UGX {int(top_hostel['Budget (UGX/sem)']):,} | <strong>Distance:</strong> {top_hostel['Distance (km)']} km</p>
            <div style="margin: 10px 0;">
                <span class="badge">WiFi: {top_hostel['WiFi']}</span>
                <span class="badge">Water: {top_hostel['Water']}</span>
                <span class="badge">Security: {top_hostel['Security']}</span>
                <span class="badge">Room: {top_hostel['Room Type']}</span>
                <span class="badge">Bathroom: {top_hostel['Bathroom']}</span>
                <span class="badge">Kitchen: {top_hostel['Kitchen']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # AI recommendation message
        if ai_score >= 4.5:
            st.success("Excellent match for your preferences. Highly recommended.")
        elif ai_score >= 3.5:
            st.info("Good match for your preferences. Recommended.")
        elif ai_score >= 2.5:
            st.warning("Moderate match. Consider adjusting your preferences.")
        else:
            st.error("Low match. Please adjust your preferences for better results.")
    
    with col2:
        # Match breakdown
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Match Breakdown")
        
        match_data = {
            'Overall Match': top_hostel['Overall Score'],
            'Similarity': top_hostel['Match Score'],
            'Budget': top_hostel.get('Budget Match', top_hostel['Match Score'])
        }
        
        for key, value in match_data.items():
            st.markdown(f"""
            <div>
                <div style="display: flex; justify-content: space-between;">
                    <span>{key}</span>
                    <span>{value:.1f}%</span>
                </div>
                <div class="score-bar">
                    <div class="score-bar-fill" style="width: {value:.1f}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Alternative recommendations
    if len(recommendations) > 1:
        st.markdown("---")
        st.subheader("Alternative Recommendations")
        
        cols = st.columns(min(3, len(recommendations) - 1))
        for idx, (_, hostel) in enumerate(recommendations.iloc[1:].iterrows()):
            if idx < 3:
                with cols[idx]:
                    st.markdown(f"""
                    <div class="card" style="padding: 15px;">
                        <h4 style="color: #003366; margin: 0;">{hostel['Hostel']}</h4>
                        <p style="margin: 5px 0;">
                            <strong>Match:</strong> {hostel['Overall Score']:.1f}%
                        </p>
                        <div class="score-bar">
                            <div class="score-bar-fill" style="width: {hostel['Overall Score']:.1f}%;"></div>
                        </div>
                        <p style="font-size: 14px; margin: 5px 0;">
                            UGX {int(hostel['Budget (UGX/sem)']):,} | {hostel['Distance (km)']} km
                        </p>
                        <span class="badge">{hostel['Room Type']}</span>
                        <span class="badge">{hostel['Bathroom']}</span>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Data preview
    with st.expander("View All Hostels"):
        st.dataframe(
            recommendations,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Budget (UGX/sem)": st.column_config.NumberColumn("Budget (UGX)", format="UGX %d"),
                "Match Score": st.column_config.NumberColumn("Match %", format="%.1f%%"),
                "Overall Score": st.column_config.NumberColumn("Overall %", format="%.1f%%")
            }
        )

def display_overview(df):
    """Display overview when no search is performed"""
    st.markdown("---")
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="margin: 0;">{len(df)}</h2>
            <p style="color: #666;">Available Hostels</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_budget = df['Budget (UGX/sem)'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="margin: 0;">UGX {int(avg_budget):,}</h2>
            <p style="color: #666;">Average Budget</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if 'Room Type' in df.columns:
            top_room = df['Room Type'].value_counts().index[0]
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="margin: 0;">{top_room}</h2>
                <p style="color: #666;">Most Common Room Type</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Browse hostels
    with st.expander("Browse Available Hostels"):
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            gender_filter = st.selectbox("Gender", ["All"] + list(df['Gender'].unique()))
        with col2:
            if 'Room Type' in df.columns:
                room_filter = st.selectbox("Room Type", ["All"] + list(df['Room Type'].unique()))
        with col3:
            if 'WiFi' in df.columns:
                wifi_filter = st.selectbox("WiFi", ["All"] + list(df['WiFi'].unique()))
        
        # Apply filters
        filtered = df.copy()
        if gender_filter != "All":
            filtered = filtered[filtered['Gender'] == gender_filter]
        if room_filter != "All":
            filtered = filtered[filtered['Room Type'] == room_filter]
        if wifi_filter != "All":
            filtered = filtered[filtered['WiFi'] == wifi_filter]
        
        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Budget (UGX/sem)": st.column_config.NumberColumn("Budget (UGX)", format="UGX %d")
            }
        )
    
    # Usage instructions
    st.info("Use the sidebar to set your preferences and click 'Find Best Hostel' to get AI-powered recommendations.")

# ---------------------------------------
# FOOTER
# ---------------------------------------
st.markdown("""
<div class="footer">
Lira University Hostel Recommendation System<br>
Powered by AI | Machine Learning | Smart Matching
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
