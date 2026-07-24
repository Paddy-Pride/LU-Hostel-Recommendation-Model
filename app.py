import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import warnings
warnings.filterwarnings('ignore')

# Try to import NLTK, but provide fallback
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    # Download NLTK data with error handling
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        except:
            pass
    
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False

# ---------------------------------------
# PAGE CONFIG
# ---------------------------------------
st.set_page_config(
    page_title="Lira University Hostel AI - NLP Enhanced",
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
.nlp-highlight {
    background: #e3f2fd;
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #003366;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------
# SIMPLE NLP PROCESSING (No NLTK dependency)
# ---------------------------------------
class SimpleTextProcessor:
    """Simple text processing without NLTK"""
    def __init__(self):
        self.stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers',
            'it', 'its', 'they', 'them', 'their', 'theirs', 'themselves',
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at',
            'to', 'by', 'with', 'without', 'about', 'against', 'between',
            'through', 'during', 'within', 'upon', 'towards', 'among'
        }
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = text.split()
        cleaned_tokens = [
            token for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        return ' '.join(cleaned_tokens)

# Define HostelNLP class
class HostelNLP(SimpleTextProcessor):
    def __init__(self):
        super().__init__()
        self.vectorizer = None
        self.hostel_descriptions = None
        self.tfidf_matrix = None
        
        self.keyword_map = {
            'budget': ['budget', 'cost', 'price', 'affordable', 'cheap', 'expensive', 'money'],
            'distance': ['distance', 'near', 'close', 'far', 'walking', 'commute'],
            'security': ['security', 'safe', 'guard', 'cctv', 'protected'],
            'wifi': ['wifi', 'internet', 'network', 'wireless', 'connectivity'],
            'water': ['water', 'tap', 'supply', 'running water'],
            'room': ['room', 'single', 'double', 'triple', 'quad', 'spacious'],
            'bathroom': ['bathroom', 'toilet', 'washroom', 'private', 'shared'],
            'kitchen': ['kitchen', 'cooking', 'stove', 'fridge'],
            'gender': ['gender', 'mixed', 'female', 'male', 'ladies', 'gentlemen']
        }
    
    def build_corpus(self, df):
        """Build corpus from hostel features"""
        descriptions = []
        
        for _, row in df.iterrows():
            desc = []
            
            # Create natural language description
            desc.append(f"hostel {row.get('Hostel', '')}")
            desc.append(f"budget {row.get('Budget (UGX/sem)', '')}")
            desc.append(f"distance {row.get('Distance (km)', '')} km")
            
            # Add categorical features
            for col in ['Gender', 'WiFi', 'Water', 'Security', 'Room Type', 'Bathroom', 'Kitchen']:
                if col in row:
                    desc.append(str(row[col]).lower())
            
            descriptions.append(' '.join(desc))
        
        self.hostel_descriptions = descriptions
        return descriptions
    
    def fit_vectorizer(self, descriptions):
        """Fit TF-IDF vectorizer on hostel descriptions"""
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(descriptions)
        return self.tfidf_matrix
    
    def extract_preferences_from_text(self, query):
        """Extract preferences from natural language query"""
        preferences = {
            'budget': 300000,
            'distance': 1.0,
            'gender': 'Mixed',
            'wifi': 'Yes',
            'water': 'Always Available',
            'security': '24/7 Guard + CCTV',
            'room_type': 'Single',
            'bathroom': 'Private',
            'kitchen': 'Private'
        }
        
        # Extract budget
        budget_patterns = [
            r'(\d+)\s*(?:thousand|k)',
            r'(?:ugx|shs)\s*(\d+)',
            r'budget\s*(\d+)'
        ]
        for pattern in budget_patterns:
            match = re.search(pattern, query.lower())
            if match:
                try:
                    budget = int(match.group(1))
                    if budget < 1000:
                        budget = budget * 1000
                    preferences['budget'] = budget
                except:
                    pass
        
        # Extract distance
        distance_patterns = [
            r'(\d+\.?\d*)\s*km',
            r'(\d+\.?\d*)\s*kilometer',
            r'within\s*(\d+\.?\d*)',
            r'near\s*(\d+\.?\d*)'
        ]
        for pattern in distance_patterns:
            match = re.search(pattern, query.lower())
            if match:
                try:
                    preferences['distance'] = float(match.group(1))
                except:
                    pass
        
        # Extract gender preference
        if any(word in query.lower() for word in ['female', 'ladies', 'girls']):
            preferences['gender'] = 'Female Only'
        elif any(word in query.lower() for word in ['male', 'gentlemen', 'boys']):
            preferences['gender'] = 'Male Only'
        elif any(word in query.lower() for word in ['mixed', 'both']):
            preferences['gender'] = 'Mixed'
        
        # Extract room type
        room_words = {
            'single': 'Single',
            'double': 'Double',
            'triple': 'Triple',
            'quad': 'Quad',
            'shared': 'Double'
        }
        for word, room_type in room_words.items():
            if word in query.lower():
                preferences['room_type'] = room_type
                break
        
        # Extract bathroom preference
        if 'private' in query.lower() and 'bathroom' in query.lower():
            preferences['bathroom'] = 'Private'
        elif 'shared' in query.lower() and 'bathroom' in query.lower():
            preferences['bathroom'] = 'Shared'
        
        # Extract kitchen preference
        if 'private' in query.lower() and 'kitchen' in query.lower():
            preferences['kitchen'] = 'Private'
        elif 'shared' in query.lower() and 'kitchen' in query.lower():
            preferences['kitchen'] = 'Shared'
        
        # Extract WiFi preference
        if 'wifi' in query.lower() or 'internet' in query.lower():
            if 'no' in query.lower():
                preferences['wifi'] = 'No'
            else:
                preferences['wifi'] = 'Yes'
        
        # Extract security preference
        if '24/7' in query.lower() or 'cctv' in query.lower():
            preferences['security'] = '24/7 Guard + CCTV'
        elif 'guard' in query.lower():
            preferences['security'] = 'Security Guard'
        elif 'gated' in query.lower():
            preferences['security'] = 'Gated Only'
        
        # Extract water preference
        if 'always' in query.lower() and 'water' in query.lower():
            preferences['water'] = 'Always Available'
        elif 'irregular' in query.lower() and 'water' in query.lower():
            preferences['water'] = 'Irregular'
        
        return preferences
    
    def get_semantic_similarity(self, query, top_n=5):
        """Get semantic similarity between query and hostels"""
        if self.vectorizer is None or self.tfidf_matrix is None:
            return None, None
        
        cleaned_query = self.preprocess_text(query)
        query_vector = self.vectorizer.transform([cleaned_query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        top_indices = np.argsort(similarities)[::-1][:top_n]
        return top_indices, similarities[top_indices]

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

@st.cache_resource
def initialize_nlp():
    """Initialize NLP processor"""
    return HostelNLP()

@st.cache_data
def build_nlp_corpus(df):
    """Build NLP corpus from data"""
    nlp = initialize_nlp()
    descriptions = nlp.build_corpus(df)
    nlp.fit_vectorizer(descriptions)
    return nlp

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
    filtered = df.copy()
    
    # Apply filters
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
                input_df = pd.DataFrame([{
                    "Budget (UGX/sem)": [row['Budget (UGX/sem)']],
                    "Gender": [row['Gender']],
                    "Distance (km)": [row['Distance (km)']],
                    "WiFi": [row['WiFi']],
                    "Water": [row['Water']],
                    "Security": [row['Security']],
                    "Room Type": [row['Room Type']],
                    "Bathroom": [row['Bathroom']],
                    "Kitchen": [row['Kitchen']]
                }])
                score = model.predict(input_df)[0]
                ai_scores.append(score)
            
            filtered['AI_Score'] = ai_scores
            
        except Exception as e:
            st.warning(f"Model prediction warning: {str(e)}")
            filtered['AI_Score'] = 1 - abs(filtered['Budget (UGX/sem)'] - preferences['budget']) / 1000000
    
    # Calculate match scores
    match_scores = []
    for _, row in filtered.iterrows():
        scores = get_match_scores(row, preferences)
        match_scores.append(scores)
    
    match_df = pd.DataFrame(match_scores)
    filtered = pd.concat([filtered.reset_index(drop=True), match_df], axis=1)
    
    # Calculate final score
    if 'AI_Score' in filtered.columns:
        filtered['Final_Score'] = (
            filtered['AI_Score'] * 0.5 + 
            filtered['Overall'] / 20 * 0.5
        )
    else:
        filtered['Final_Score'] = filtered['Overall'] / 20
    
    filtered = filtered.sort_values('Final_Score', ascending=False)
    
    return filtered.head(n)

# ---------------------------------------
# DISPLAY FUNCTIONS
# ---------------------------------------
def display_recommendations(recommendations, preferences, model, nlp_query=None):
    """Display recommendations with match scores"""
    if len(recommendations) == 0:
        st.warning("No recommendations found. Please adjust your preferences.")
        return
    
    top_hostel = recommendations.iloc[0]
    
    # Get AI score
    if model is not None and 'AI_Score' in top_hostel:
        ai_score = top_hostel['AI_Score']
    else:
        ai_score = top_hostel['Final_Score'] * 5
    
    # Show NLP query if provided
    if nlp_query:
        st.markdown(f"""
        <div class="nlp-highlight">
            <strong>NLP Query:</strong> {nlp_query}
        </div>
        """, unsafe_allow_html=True)
    
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
    
    # Show all recommendations
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
    
    # NLP examples
    with st.expander("Try NLP Search Examples"):
        st.markdown("""
        Try typing these examples in the NLP search box:
        - "I need a single room with wifi and private bathroom near campus"
        - "Looking for a hostel with 24/7 security and reliable water supply"
        - "Budget 250k for a double room with shared kitchen"
        - "Female only hostel within 1km with good security"
        - "Mixed hostel with wifi and water always available"
        """)

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
    
    # Initialize NLP
    try:
        nlp = initialize_nlp()
        nlp = build_nlp_corpus(df)
        nlp_available = True
    except Exception as e:
        st.warning(f"NLP initialization warning: {str(e)}")
        nlp_available = False
    
    # Title
    st.title("Lira University Hostel Recommendation System")
    st.write("AI-powered recommendation with Natural Language Processing")
    
    # NLP Search Box
    st.markdown("---")
    st.subheader("Natural Language Search")
    st.markdown("Describe what you're looking for in natural language")
    
    nlp_query = st.text_input("Type your hostel requirements:", placeholder="e.g., I want a single room with wifi and private bathroom near campus")
    
    if nlp_query and nlp_available:
        with st.spinner("Analyzing your request with NLP..."):
            try:
                # Extract preferences from NLP
                nlp_preferences = nlp.extract_preferences_from_text(nlp_query)
                
                # Get semantic similarity matches
                semantic_indices, semantic_scores = nlp.get_semantic_similarity(nlp_query, top_n=5)
                
                # Display extracted preferences
                st.markdown('<div class="nlp-highlight">', unsafe_allow_html=True)
                st.markdown("**NLP Extracted Preferences:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"Budget: UGX {nlp_preferences['budget']:,}")
                    st.write(f"Distance: {nlp_preferences['distance']} km")
                    st.write(f"Gender: {nlp_preferences['gender']}")
                with col2:
                    st.write(f"Room Type: {nlp_preferences['room_type']}")
                    st.write(f"WiFi: {nlp_preferences['wifi']}")
                    st.write(f"Water: {nlp_preferences['water']}")
                with col3:
                    st.write(f"Security: {nlp_preferences['security']}")
                    st.write(f"Bathroom: {nlp_preferences['bathroom']}")
                    st.write(f"Kitchen: {nlp_preferences['kitchen']}")
                
                # Show semantic matches
                if semantic_indices is not None:
                    st.markdown("**Semantic Matches (NLP Understanding):**")
                    for idx, score in zip(semantic_indices, semantic_scores):
                        hostel = df.iloc[idx]
                        st.write(f"- {hostel['Hostel']} (Relevance: {score*100:.1f}%)")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Get recommendations based on NLP preferences
                recommendations = get_recommendations(df, model, nlp_preferences, 5)
                
                # Display recommendations
                st.markdown("---")
                st.subheader("Recommended Hostels")
                display_recommendations(recommendations, nlp_preferences, model, nlp_query)
            
            except Exception as e:
                st.error(f"NLP processing error: {str(e)}")
                st.info("Please try using the structured preferences in the sidebar instead.")
    
    st.markdown("---")
    
    # Traditional sidebar for structured input
    with st.sidebar:
        st.header("Structured Preferences")
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
    
    # Traditional search results
    if 'search' in st.session_state and st.session_state.search:
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
        
        recommendations = get_recommendations(df, model, preferences, num_recommendations)
        display_recommendations(recommendations, preferences, model)
        
        st.session_state.search = False
    elif not nlp_query:
        display_overview(df)

# ---------------------------------------
# FOOTER
# ---------------------------------------
st.markdown("""
<div class="footer">
Lira University Hostel Recommendation System<br>
Powered by AI | NLP | Machine Learning | Smart Matching
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
