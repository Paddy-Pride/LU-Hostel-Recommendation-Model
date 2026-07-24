import streamlit as st
import pandas as pd
import numpy as np
import joblib
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
        model = joblib.load("hostel_ai_model.pkl")
        return model
    except Exception as e:
        st.warning(f"Model loading warning: {str(e)}")
        return None

# ---------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------
def get_match_scores(hostel, preferences):
    """Calculate match scores for a hostel"""
    scores = {}
    
    # 1. Budget match
    if 'Budget (UGX/sem)' in hostel and 'budget' in preferences:
        budget_diff = abs(hostel['Budget (UGX/sem)'] - preferences['budget'])
        max_budget = 1000000
        scores['Budget'] = max(0, 100 - (budget_diff / max_budget * 100))
    else:
        scores['Budget'] = 50
    
    # 2. Distance match
    if 'Distance (km)' in hostel and 'distance' in preferences:
        if hostel['Distance (km)'] <= preferences['distance']:
            scores['Distance'] = 100
        else:
            scores['Distance'] = max(0, 100 - ((hostel['Distance (km)'] - preferences['distance']) / 5 * 100))
    else:
        scores['Distance'] = 50
    
    # 3. Facility matches
    facility_cols = ['Gender', 'WiFi', 'Water', 'Security', 'Room Type', 'Bathroom', 'Kitchen']
    matches = 0
    total = 0
    
    for col in facility_cols:
        if col in hostel and col.lower() in preferences:
            total += 1
            hostel_val = str(hostel[col]).lower()
            pref_val = str(preferences[col.lower()]).lower()
            if hostel_val == pref_val:
                matches += 1
    
    scores['Facilities'] = (matches / total * 100) if total > 0 else 50
    
    # 4. Overall match
    scores['Overall'] = (
        scores['Budget'] * 0.30 +
        scores['Distance'] * 0.25 +
        scores['Facilities'] * 0.45
    )
    
    return scores

def get_recommendations(df, model, preferences, n=5):
    """Get recommendations using the trained model"""
    # Start with all data
    filtered = df.copy()
    
    # Apply basic filters
    if 'gender' in preferences:
        filtered = filtered[filtered['Gender'] == preferences['gender']]
    
    if 'wifi' in preferences:
        filtered = filtered[filtered['WiFi'] == preferences['wifi']]
    
    if 'room_type' in preferences:
        filtered = filtered[filtered['Room Type'] == preferences['room_type']]
    
    # If no results, use all data
    if len(filtered) == 0:
        filtered = df.copy()
    
    # Get AI scores from model
    if model is not None:
        try:
            ai_scores = []
            for _, row in filtered.iterrows():
                # Prepare input for model - match original model format
                input_data = {
                    "Hostel": [row['Hostel']],
                    "Budget (UGX/sem)": [row['Budget (UGX/sem)']],
                    "Gender": [row['Gender']],
                    "Distance (km)": [row['Distance (km)']],
                    "WiFi": [row['WiFi']],
                    "Water": [row['Water']],
                    "Security": [row['Security']],
                    "Room Type": [row['Room Type']],
                    "Bathroom": [row['Bathroom']],
                    "Kitchen": [row['Kitchen']]
                }
                input_df = pd.DataFrame(input_data)
                
                # Predict
                score = model.predict(input_df)[0]
                ai_scores.append(score)
            
            filtered['AI_Score'] = ai_scores
            
        except Exception as e:
            st.warning(f"Model prediction issue: {str(e)}")
            # Fallback: use budget similarity
            filtered['AI_Score'] = 1 - abs(filtered['Budget (UGX/sem)'] - preferences['budget']) / 1000000
    
    # Calculate match scores
    match_scores = []
    for _, row in filtered.iterrows():
        scores = get_match_scores(row, preferences)
        match_scores.append(scores)
    
    match_df = pd.DataFrame(match_scores)
    filtered = pd.concat([filtered.reset_index(drop=True), match_df], axis=1)
    
    # Calculate final score (combine AI and match scores)
    if 'AI_Score' in filtered.columns:
        # Normalize AI score to percentage
        ai_max = filtered['AI_Score'].max()
        ai_min = filtered['AI_Score'].min()
        if ai_max > ai_min:
            filtered['AI_Percentage'] = ((filtered['AI_Score'] - ai_min) / (ai_max - ai_min)) * 100
        else:
            filtered['AI_Percentage'] = 50
        
        # Final score: 60% AI, 40% match
        filtered['Final_Score'] = (
            filtered['AI_Percentage'] * 0.6 + 
            filtered['Overall'] * 0.4
        )
    else:
        filtered['Final_Score'] = filtered['Overall']
    
    # Sort by final score
    filtered = filtered.sort_values('Final_Score', ascending=False)
    
    return filtered.head(n)

# ---------------------------------------
# DISPLAY FUNCTIONS
# ---------------------------------------
def display_recommendations(recommendations, preferences, model):
    """Display recommendations with match scores"""
    if len(recommendations) == 0:
        st.warning("No recommendations found. Please adjust your preferences.")
        return
    
    top_hostel = recommendations.iloc[0]
    
    # Get AI score
    if model is not None and 'AI_Score' in top_hostel:
        ai_score = top_hostel['AI_Score']
    else:
        ai_score = top_hostel['Final_Score'] / 20
    
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
            <p><strong>Overall Match:</strong> {top_hostel['Overall']:.1f}%</p>
            <div class="score-bar">
                <div class="score-bar-fill" style="width: {top_hostel['Overall']:.1f}%;"></div>
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
            'Overall Match': top_hostel['Overall'],
            'Budget': top_hostel['Budget'],
            'Facilities': top_hostel['Facilities'],
            'Distance': top_hostel['Distance']
        }
        
        for key, value in match_data.items():
            color = '#28a745' if value >= 70 else '#ffc107' if value >= 50 else '#dc3545'
            st.markdown(f"""
            <div>
                <div style="display: flex; justify-content: space-between;">
                    <span>{key}</span>
                    <span style="color: {color}; font-weight: bold;">{value:.1f}%</span>
                </div>
                <div class="score-bar">
                    <div class="score-bar-fill" style="width: {value:.1f}%; background: {color};"></div>
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
                            <strong>Match:</strong> {hostel['Overall']:.1f}%
                        </p>
                        <div class="score-bar">
                            <div class="score-bar-fill" style="width: {hostel['Overall']:.1f}%;"></div>
                        </div>
                        <p style="font-size: 14px; margin: 5px 0;">
                            UGX {int(hostel['Budget (UGX/sem)']):,} | {hostel['Distance (km)']} km
                        </p>
                        <div>
                            <span class="badge">{hostel['Room Type']}</span>
                            <span class="badge">{hostel['Bathroom']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Show all recommendations in table
    with st.expander("View All Recommendations"):
        display_cols = ['Hostel', 'Budget (UGX/sem)', 'Distance (km)', 
                       'Budget', 'Facilities', 'Distance', 'Overall']
        display_cols = [col for col in display_cols if col in recommendations.columns]
        
        display_df = recommendations[display_cols].copy()
        display_df['Budget (UGX/sem)'] = display_df['Budget (UGX/sem)'].apply(lambda x: f"UGX {int(x):,}")
        
        display_df.columns = ['Hostel', 'Budget', 'Distance', 
                             'Budget %', 'Facilities %', 'Distance %', 'Overall %']
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
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
        col1, col2, col3 = st.columns(3)
        with col1:
            gender_filter = st.selectbox("Gender", ["All"] + list(df['Gender'].unique()))
        with col2:
            if 'Room Type' in df.columns:
                room_filter = st.selectbox("Room Type", ["All"] + list(df['Room Type'].unique()))
        with col3:
            if 'WiFi' in df.columns:
                wifi_filter = st.selectbox("WiFi", ["All"] + list(df['WiFi'].unique()))
        
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
    
    st.info("Use the sidebar to set your preferences and click 'Find Best Hostel' to get AI-powered recommendations.")

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
        recommendations = get_recommendations(df, model, preferences, num_recommendations)
        
        # Display results
        display_recommendations(recommendations, preferences, model)
        
        st.session_state.search = False
    else:
        display_overview(df)

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
