import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------
# PAGE CONFIG
# ---------------------------------------
st.set_page_config(
    page_title="Lira University Hostel AI - Smart Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------
# CUSTOM CSS
# ---------------------------------------
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8edf5 100%);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #003366;
        font-weight: 700;
    }
    .card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        transition: transform 0.3s ease;
        border-left: 5px solid #003366;
        margin-bottom: 20px;
    }
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0,0,0,0.12);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        text-align: center;
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
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        display: inline-block;
    }
    .stButton > button {
        background: #003366;
        color: white;
        border: none;
        padding: 10px 30px;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background: #004d99;
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(0,51,102,0.3);
    }
    .stSidebar .sidebar-content {
        background: #f8f9fc;
    }
    .match-high {
        color: #28a745;
        font-weight: bold;
    }
    .match-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .match-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------
# CACHE DATA LOADING
# ---------------------------------------
@st.cache_data
def load_data():
    """Load and preprocess the dataset with caching"""
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
        df["Kitchen"] = df["Kitchen"].fillna(df["Kitchen"].mode()[0])
        df["Water"] = df["Water"].fillna("Always Available")
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
# AI ENHANCEMENTS - Advanced Recommendation Engine
# ---------------------------------------
class AdvancedHostelRecommender:
    def __init__(self, df):
        self.df = df
        self.feature_columns = ['Budget (UGX/sem)', 'Distance (km)']
        self.categorical_columns = ['Gender', 'WiFi', 'Water', 'Security', 'Room Type', 'Bathroom', 'Kitchen']
        self.label_encoders = {}
        self.vectorizer = None
        self._prepare_features()
    
    def _prepare_features(self):
        """Prepare feature matrix for similarity calculations"""
        # Encode categorical features
        for col in self.categorical_columns:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
        
        # Create TF-IDF features for text matching
        text_features = self.df[self.categorical_columns].astype(str).agg(' '.join, axis=1)
        self.vectorizer = TfidfVectorizer(max_features=50)
        self.text_features = self.vectorizer.fit_transform(text_features)
        
        # Normalize numerical features
        self.numeric_features = self.df[self.feature_columns].copy()
        self.numeric_features['Budget (UGX/sem)'] = (
            self.numeric_features['Budget (UGX/sem)'] - self.numeric_features['Budget (UGX/sem)'].mean()
        ) / self.numeric_features['Budget (UGX/sem)'].std()
        self.numeric_features['Distance (km)'] = (
            self.numeric_features['Distance (km)'] - self.numeric_features['Distance (km)'].mean()
        ) / self.numeric_features['Distance (km)'].std()
        
        # Combine all features
        self.feature_matrix = np.hstack([
            self.numeric_features.values,
            self.df[[f'{col}_encoded' for col in self.categorical_columns if f'{col}_encoded' in self.df.columns]].values,
            self.text_features.toarray()
        ])
    
    def get_recommendations(self, preferences, n_recommendations=5):
        """Get top N hostel recommendations based on preferences"""
        # Encode user preferences
        preference_vector = []
        
        # Numerical features
        budget_norm = (preferences['budget'] - self.df['Budget (UGX/sem)'].mean()) / self.df['Budget (UGX/sem)'].std()
        distance_norm = (preferences['distance'] - self.df['Distance (km)'].mean()) / self.df['Distance (km)'].std()
        preference_vector.extend([budget_norm, distance_norm])
        
        # Categorical features
        for col in self.categorical_columns:
            if col in preferences:
                try:
                    encoded = self.label_encoders[col].transform([preferences[col]])[0]
                    preference_vector.append(encoded)
                except:
                    preference_vector.append(0)
        
        # Text features
        user_text = ' '.join([str(preferences.get(col, '')) for col in self.categorical_columns if col in preferences])
        user_text_features = self.vectorizer.transform([user_text]).toarray()
        preference_vector.extend(user_text_features[0])
        
        preference_vector = np.array(preference_vector).reshape(1, -1)
        
        # Calculate similarities
        similarities = cosine_similarity(preference_vector, self.feature_matrix)[0]
        
        # Get top N recommendations
        top_indices = np.argsort(similarities)[::-1][:n_recommendations]
        
        recommendations = self.df.iloc[top_indices].copy()
        recommendations['Similarity Score'] = similarities[top_indices]
        
        # Calculate additional metrics
        budget_diff = abs(recommendations['Budget (UGX/sem)'] - preferences['budget'])
        recommendations['Budget Compatibility'] = 1 - (budget_diff / budget_diff.max()) if budget_diff.max() > 0 else 1
        recommendations['Overall Score'] = (
            recommendations['Similarity Score'] * 0.6 + 
            recommendations['Budget Compatibility'] * 0.4
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
    recommender = AdvancedHostelRecommender(df)
    
    # Title and description
    st.title("🏠 Lira University Hostel AI Recommendation System")
    st.markdown("### Smart Accommodation Matching Powered by Advanced AI")
    
    st.markdown("""
    Discover the perfect hostel that matches your unique preferences using our 
    **AI-powered recommendation engine**. We analyze multiple factors including budget, 
    facilities, and location to find your ideal accommodation.
    """)
    
    # Sidebar for user input
    with st.sidebar:
        st.header("🎯 Your Preferences")
        st.markdown("---")
        
        # Budget input with preset options
        budget = st.number_input(
            "💰 Budget (UGX/semester)",
            min_value=150000,
            max_value=1000000,
            value=300000,
            step=10000,
            help="Your maximum budget per semester"
        )
        
        # Quick budget presets
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Low", key="budget_low"):
                st.session_state.budget = 200000
        with col2:
            if st.button("Mid", key="budget_mid"):
                st.session_state.budget = 350000
        with col3:
            if st.button("High", key="budget_high"):
                st.session_state.budget = 500000
        
        if 'budget' in st.session_state:
            budget = st.session_state.budget
        
        gender = st.selectbox(
            "👤 Gender Preference",
            ["Mixed", "Female Only", "Male Only"]
        )
        
        distance = st.slider(
            "📏 Maximum Distance (km)",
            0.1, 5.0, 1.0, 0.1,
            help="Maximum distance from campus"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            wifi = st.selectbox("📶 WiFi", ["Yes", "No"])
            security = st.selectbox(
                "🔒 Security",
                ["24/7 Guard + CCTV", "Security Guard", "Gated Only", "Basic"]
            )
            bathroom = st.selectbox("🚿 Bathroom", ["Private", "Shared"])
        
        with col2:
            water = st.selectbox(
                "💧 Water Availability",
                ["Always Available", "Sometimes Interrupted", "Irregular"]
            )
            room = st.selectbox(
                "🛏️ Room Type",
                ["Single", "Double", "Triple", "Quad"]
            )
            kitchen = st.selectbox("🍳 Kitchen", ["Private", "Shared"])
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            n_recommendations = st.slider(
                "Number of recommendations",
                1, 10, 5
            )
            show_similar_hostels = st.checkbox("Show similar hostels", True)
        
        # Recommendation button
        st.markdown("---")
        recommend_button = st.button(
            "🔍 Find Best Hostel",
            use_container_width=True
        )
        
        if st.button("🔄 Reset Preferences", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    
    # Main content area
    if recommend_button or 'recommendations' in st.session_state:
        # Gather preferences
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
        recommendations = recommender.get_recommendations(preferences, n_recommendations)
        st.session_state.recommendations = recommendations
        
        # Display recommendations
        display_recommendations(recommendations, preferences, model, df)
    
    else:
        # Display overview when no recommendations yet
        display_overview(df)
    
    # Footer
    st.markdown("""
    <div class='footer'>
        <strong>Lira University Hostel Recommendation System</strong><br>
        Powered by Advanced AI | Machine Learning | Smart Matching Algorithms<br>
        <span style='font-size:12px;'>Last Updated: {}</span>
    </div>
    """.format(datetime.now().strftime("%B %d, %Y")), unsafe_allow_html=True)

def display_recommendations(recommendations, preferences, model, df):
    """Display the AI recommendations with enhanced UI"""
    
    # Top recommendation
    top_hostel = recommendations.iloc[0]
    
    # AI-generated score
    if model is not None:
        try:
            input_df = pd.DataFrame([{
                "Hostel": ["Unknown"],
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
            ai_score = top_hostel['Overall Score'] * 5
    else:
        ai_score = top_hostel['Overall Score'] * 5
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h3 style='color:#003366; margin:0;'>{top_hostel['Hostel']}</h3>
            <p style='color:#666; margin:5px 0;'>🏆 Top Recommendation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h2 style='color:#28a745; margin:0;'>{ai_score:.1f}</h2>
            <p style='color:#666; margin:5px 0;'>⭐ AI Score / 5</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <h2 style='color:#003366; margin:0;'>UGX {int(top_hostel['Budget (UGX/sem)']):,}</h2>
            <p style='color:#666; margin:5px 0;'>💰 Budget</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <h2 style='color:#003366; margin:0;'>{top_hostel['Distance (km)']} km</h2>
            <p style='color:#666; margin:5px 0;'>📏 Distance</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed top recommendation
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class='card'>
            <h3>🏠 {top_hostel['Hostel']}</h3>
            <p style='color:#666; margin:10px 0;'><strong>Similarity Match:</strong> 
                <span class='match-high'>{top_hostel['Similarity Score']*100:.1f}%</span>
            </p>
            <p style='color:#666; margin:10px 0;'>
                <strong>Budget:</strong> UGX {int(top_hostel['Budget (UGX/sem)']):,} | 
                <strong>Distance:</strong> {top_hostel['Distance (km)']} km
            </p>
            <div style='display:flex; gap:10px; flex-wrap:wrap; margin:15px 0;'>
                <span class='badge'>📶 {top_hostel['WiFi']}</span>
                <span class='badge'>💧 {top_hostel['Water']}</span>
                <span class='badge'>🔒 {top_hostel['Security']}</span>
                <span class='badge'>🛏️ {top_hostel['Room Type']}</span>
                <span class='badge'>🚿 {top_hostel['Bathroom']}</span>
                <span class='badge'>🍳 {top_hostel['Kitchen']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # AI explanation
        if ai_score >= 4.5:
            st.success("🌟 **Excellent Match!** This hostel perfectly aligns with your preferences.")
        elif ai_score >= 3.5:
            st.info("👍 **Good Match!** This hostel meets most of your criteria.")
        else:
            st.warning("⚠️ **Partial Match.** Consider adjusting your preferences for better options.")
    
    with col2:
        # Similarity radar chart
        categories = ['Budget', 'Facilities', 'Location', 'Security', 'Comfort']
        values = [
            top_hostel['Budget Compatibility'] * 100,
            top_hostel['Similarity Score'] * 100,
            (1 - top_hostel['Distance (km)'] / 5) * 100,
            min(100, top_hostel['Similarity Score'] * 120),
            min(100, top_hostel['Similarity Score'] * 110)
        ]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            line_color='#003366',
            fillcolor='rgba(0,51,102,0.2)'
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=False,
            height=250,
            margin=dict(l=40, r=40, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Alternative recommendations
    if len(recommendations) > 1:
        st.markdown("---")
        st.subheader("🔄 Alternative Recommendations")
        
        # Show top alternatives in a grid
        cols = st.columns(min(3, len(recommendations) - 1))
        for idx, (_, hostel) in enumerate(recommendations.iloc[1:].iterrows()):
            if idx < 3:
                with cols[idx]:
                    st.markdown(f"""
                    <div class='card' style='padding:15px;'>
                        <h4 style='color:#003366; margin:0;'>{hostel['Hostel']}</h4>
                        <p style='margin:5px 0;'>
                            <strong>Score:</strong> {hostel['Overall Score']*100:.1f}%
                        </p>
                        <p style='margin:5px 0; font-size:14px;'>
                            UGX {int(hostel['Budget (UGX/sem)']):,} • {hostel['Distance (km)']} km
                        </p>
                        <div style='margin-top:10px;'>
                            <span class='badge'>{hostel['Room Type']}</span>
                            <span class='badge'>{hostel['Bathroom']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Data exploration
    with st.expander("📊 Explore All Hostels"):
        display_data_exploration(df)

def display_overview(df):
    """Display overview statistics and visualizations"""
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h2 style='margin:0;'>{len(df)}</h2>
            <p style='color:#666;'>🏠 Available Hostels</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_budget = df['Budget (UGX/sem)'].mean()
        st.markdown(f"""
        <div class='metric-card'>
            <h2 style='margin:0;'>UGX {int(avg_budget):,}</h2>
            <p style='color:#666;'>💰 Average Budget</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        room_types = df['Room Type'].value_counts().index[0]
        st.markdown(f"""
        <div class='metric-card'>
            <h2 style='margin:0;'>{room_types}</h2>
            <p style='color:#666;'>🛏️ Most Common Room Type</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick filters and data preview
    with st.expander("🔍 Browse Hostels"):
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_gender = st.selectbox("Filter by Gender", ["All"] + list(df['Gender'].unique()))
        with col2:
            filter_room = st.selectbox("Filter by Room Type", ["All"] + list(df['Room Type'].unique()))
        with col3:
            filter_wifi = st.selectbox("Filter by WiFi", ["All"] + list(df['WiFi'].unique()))
        
        # Apply filters
        filtered_df = df.copy()
        if filter_gender != "All":
            filtered_df = filtered_df[filtered_df['Gender'] == filter_gender]
        if filter_room != "All":
            filtered_df = filtered_df[filtered_df['Room Type'] == filter_room]
        if filter_wifi != "All":
            filtered_df = filtered_df[filtered_df['WiFi'] == filter_wifi]
        
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Budget (UGX/sem)": st.column_config.NumberColumn("Budget (UGX)", format="UGX %d")
            }
        )

def display_data_exploration(df):
    """Display interactive data exploration visualizations"""
    # Budget distribution
    fig1 = px.histogram(
        df,
        x='Budget (UGX/sem)',
        nbins=20,
        title='Budget Distribution',
        color_discrete_sequence=['#003366']
    )
    fig1.update_layout(height=300)
    
    # Room type distribution
    room_counts = df['Room Type'].value_counts()
    fig2 = px.pie(
        values=room_counts.values,
        names=room_counts.index,
        title='Room Type Distribution',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig2.update_layout(height=300)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()
