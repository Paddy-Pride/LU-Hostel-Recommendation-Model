import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------
# PAGE CONFIG
# ---------------------------------------
st.set_page_config(
    page_title="Hostel AI Chatbot",
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
.chat-container {
    max-height: 500px;
    overflow-y: auto;
    padding: 10px;
    background: white;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.message {
    padding: 12px 16px;
    border-radius: 12px;
    margin: 8px 0;
    max-width: 80%;
}
.user-message {
    background: #003366;
    color: white;
    margin-left: auto;
    text-align: right;
}
.bot-message {
    background: #e8edf5;
    color: #003366;
    margin-right: auto;
}
.timestamp {
    font-size: 10px;
    color: #999;
    margin-top: 4px;
}
.input-area {
    display: flex;
    gap: 10px;
    padding: 10px 0;
}
.card {
    background: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    margin-bottom: 20px;
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
.suggestion-chip {
    background: #e3f2fd;
    padding: 8px 16px;
    border-radius: 20px;
    cursor: pointer;
    display: inline-block;
    margin: 4px;
    font-size: 14px;
    transition: all 0.3s ease;
}
.suggestion-chip:hover {
    background: #003366;
    color: white;
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
# CHATBOT CLASS
# ---------------------------------------
class HostelChatbot:
    def __init__(self, df, model):
        self.df = df
        self.model = model
        self.context = {}
        self.preferences = {}
        self.conversation_history = []
        self.required_preferences = ['budget', 'gender', 'distance', 'wifi', 'water', 'security', 'room_type']
        
    def reset(self):
        """Reset chatbot state"""
        self.context = {}
        self.preferences = {}
        self.conversation_history = []
    
    def extract_preferences_from_text(self, text):
        """Extract preferences from user text"""
        text_lower = text.lower()
        extracted = {}
        
        # Extract budget
        budget_patterns = [
            r'(\d+)\s*(?:thousand|k)',
            r'(?:ugx|shs)\s*(\d+)',
            r'budget\s*(\d+)',
            r'(\d+)\s*(?:million|m)'
        ]
        for pattern in budget_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    budget = int(match.group(1))
                    if 'million' in text_lower or 'm' in text_lower:
                        budget = budget * 1000000
                    elif budget < 1000:
                        budget = budget * 1000
                    extracted['budget'] = budget
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
            match = re.search(pattern, text_lower)
            if match:
                try:
                    extracted['distance'] = float(match.group(1))
                except:
                    pass
        
        # Extract gender
        if any(word in text_lower for word in ['female', 'ladies', 'girls', 'women']):
            extracted['gender'] = 'Female Only'
        elif any(word in text_lower for word in ['male', 'gentlemen', 'boys', 'men']):
            extracted['gender'] = 'Male Only'
        elif any(word in text_lower for word in ['mixed', 'both', 'co-ed']):
            extracted['gender'] = 'Mixed'
        
        # Extract room type
        room_words = {
            'single': 'Single',
            'double': 'Double',
            'triple': 'Triple',
            'quad': 'Quad'
        }
        for word, room_type in room_words.items():
            if word in text_lower:
                extracted['room_type'] = room_type
                break
        
        # Extract bathroom
        if 'private' in text_lower and ('bathroom' in text_lower or 'toilet' in text_lower):
            extracted['bathroom'] = 'Private'
        elif 'shared' in text_lower and ('bathroom' in text_lower or 'toilet' in text_lower):
            extracted['bathroom'] = 'Shared'
        
        # Extract kitchen
        if 'private' in text_lower and 'kitchen' in text_lower:
            extracted['kitchen'] = 'Private'
        elif 'shared' in text_lower and 'kitchen' in text_lower:
            extracted['kitchen'] = 'Shared'
        
        # Extract WiFi
        if 'wifi' in text_lower or 'internet' in text_lower:
            if 'no' in text_lower and ('wifi' in text_lower or 'internet' in text_lower):
                extracted['wifi'] = 'No'
            else:
                extracted['wifi'] = 'Yes'
        
        # Extract security
        if '24/7' in text_lower or 'cctv' in text_lower:
            extracted['security'] = '24/7 Guard + CCTV'
        elif 'guard' in text_lower:
            extracted['security'] = 'Security Guard'
        elif 'gated' in text_lower:
            extracted['security'] = 'Gated Only'
        
        # Extract water
        if 'always' in text_lower and 'water' in text_lower:
            extracted['water'] = 'Always Available'
        elif 'irregular' in text_lower and 'water' in text_lower:
            extracted['water'] = 'Irregular'
        
        return extracted
    
    def get_missing_preferences(self):
        """Get list of missing preferences"""
        missing = []
        for pref in self.required_preferences:
            if pref not in self.preferences:
                missing.append(pref)
        return missing
    
    def get_next_question(self):
        """Get next question to ask user"""
        missing = self.get_missing_preferences()
        
        if not missing:
            return None
        
        question_map = {
            'budget': "What is your budget per semester in UGX? (e.g., 300000)",
            'gender': "Which gender preference do you have? (Mixed, Female Only, Male Only)",
            'distance': "What is the maximum distance from campus in km? (e.g., 1.0)",
            'wifi': "Do you need WiFi? (Yes/No)",
            'water': "What water availability do you prefer? (Always Available, Sometimes Interrupted, Irregular)",
            'security': "What security level do you prefer? (24/7 Guard + CCTV, Security Guard, Gated Only, Basic)",
            'room_type': "What room type do you prefer? (Single, Double, Triple, Quad)",
            'bathroom': "Do you prefer private or shared bathroom?",
            'kitchen': "Do you prefer private or shared kitchen?"
        }
        
        next_pref = missing[0]
        return question_map.get(next_pref, f"Please tell me your {next_pref} preference.")
    
    def update_preferences(self, user_input):
        """Update preferences from user input"""
        extracted = self.extract_preferences_from_text(user_input)
        
        # Update preferences with extracted values
        for key, value in extracted.items():
            if key in self.required_preferences:
                self.preferences[key] = value
        
        # Try to parse budget if not extracted
        if 'budget' not in self.preferences:
            try:
                numbers = re.findall(r'\d+', user_input)
                if numbers:
                    budget = int(numbers[0])
                    if budget < 1000:
                        budget = budget * 1000
                    self.preferences['budget'] = budget
            except:
                pass
        
        # Try to parse distance if not extracted
        if 'distance' not in self.preferences:
            try:
                numbers = re.findall(r'\d+\.?\d*', user_input)
                if numbers:
                    dist = float(numbers[0])
                    if dist > 0 and dist <= 10:
                        self.preferences['distance'] = dist
            except:
                pass
        
        return extracted
    
    def get_recommendations(self):
        """Get recommendations based on current preferences"""
        if len(self.preferences) < len(self.required_preferences):
            return None
        
        # Use the get_recommendations function
        return get_recommendations(self.df, self.model, self.preferences)
    
    def format_recommendation_message(self, recommendations):
        """Format recommendations for display"""
        if recommendations is None or len(recommendations) == 0:
            return "I couldn't find any hostels matching your preferences. Could you adjust your requirements?"
        
        top_hostel = recommendations.iloc[0]
        
        # Get AI score
        if self.model is not None and 'AI_Score' in top_hostel:
            ai_score = top_hostel['AI_Score']
        else:
            ai_score = top_hostel.get('Final_Score', 3) * 5 / 20
        
        message = f"""
        Based on your preferences, I recommend:

        🏠 **{top_hostel['Hostel']}**
        - AI Score: {ai_score:.1f}/5
        - Budget: UGX {int(top_hostel['Budget (UGX/sem)']):,}
        - Distance: {top_hostel['Distance (km)']} km from campus
        - WiFi: {top_hostel['WiFi']}
        - Water: {top_hostel['Water']}
        - Security: {top_hostel['Security']}
        - Room Type: {top_hostel['Room Type']}
        - Bathroom: {top_hostel['Bathroom']}
        - Kitchen: {top_hostel['Kitchen']}
        """
        
        if len(recommendations) > 1:
            message += "\n\n**Alternatives:**\n"
            for i in range(1, min(3, len(recommendations))):
                hostel = recommendations.iloc[i]
                message += f"- {hostel['Hostel']} (UGX {int(hostel['Budget (UGX/sem)']):,}, {hostel['Distance (km)']} km)\n"
        
        message += "\n\nWould you like to adjust any preferences or get more details?"
        
        return message

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
                score = model.predict(input_df)[0]
                ai_scores.append(score)
            
            filtered['AI_Score'] = ai_scores
            
        except Exception as e:
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
        ai_max = filtered['AI_Score'].max()
        ai_min = filtered['AI_Score'].min()
        if ai_max > ai_min:
            filtered['AI_Percentage'] = ((filtered['AI_Score'] - ai_min) / (ai_max - ai_min)) * 100
        else:
            filtered['AI_Percentage'] = 50
        
        filtered['Final_Score'] = (
            filtered['AI_Percentage'] * 0.6 + 
            filtered['Overall'] * 0.4
        )
    else:
        filtered['Final_Score'] = filtered['Overall']
    
    filtered = filtered.sort_values('Final_Score', ascending=False)
    
    return filtered.head(n)

def display_hostel_card(hostel, ai_score=None):
    """Display a hostel as a card"""
    if ai_score is None and 'AI_Score' in hostel:
        ai_score = hostel['AI_Score']
    elif ai_score is None:
        ai_score = hostel.get('Final_Score', 3) * 5 / 20
    
    st.markdown(f"""
    <div class="card">
        <h3 style="color: #003366; margin: 0;">{hostel['Hostel']}</h3>
        <p><strong>AI Score:</strong> {ai_score:.1f}/5</p>
        <div class="score-bar">
            <div class="score-bar-fill" style="width: {ai_score/5*100:.1f}%;"></div>
        </div>
        <p><strong>Budget:</strong> UGX {int(hostel['Budget (UGX/sem)']):,} | <strong>Distance:</strong> {hostel['Distance (km)']} km</p>
        <div style="margin: 10px 0;">
            <span class="badge">WiFi: {hostel['WiFi']}</span>
            <span class="badge">Water: {hostel['Water']}</span>
            <span class="badge">Security: {hostel['Security']}</span>
            <span class="badge">Room: {hostel['Room Type']}</span>
            <span class="badge">Bathroom: {hostel['Bathroom']}</span>
            <span class="badge">Kitchen: {hostel['Kitchen']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = HostelChatbot(df, model)
        st.session_state.messages = []
        st.session_state.show_recommendations = False
        st.session_state.recommendations = None
        st.session_state.chat_complete = False
        
        # Add welcome message
        welcome_msg = "Hello! I'm your hostel assistant. I'll help you find the perfect hostel based on your preferences.\n\nWhat type of hostel are you looking for? (e.g., 'Single room with wifi, budget 300k near campus')"
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
    
    # Title
    st.title("Hostel AI Assistant")
    st.write("Chat with me to find your ideal hostel at Lira University")
    
    # Chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        chat_html = '<div class="chat-container">'
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_html += f"""
                <div class="message user-message">
                    {msg["content"]}
                    <div class="timestamp">{datetime.now().strftime("%I:%M %p")}</div>
                </div>
                """
            else:
                chat_html += f"""
                <div class="message bot-message">
                    {msg["content"]}
                    <div class="timestamp">{datetime.now().strftime("%I:%M %p")}</div>
                </div>
                """
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)
        
        # Show recommendations if available
        if st.session_state.show_recommendations and st.session_state.recommendations is not None:
            st.markdown("### Top Recommendations")
            recommendations = st.session_state.recommendations
            
            # Display top 3 recommendations
            for i in range(min(3, len(recommendations))):
                hostel = recommendations.iloc[i]
                if i == 0:
                    st.success("🏆 Best Match")
                display_hostel_card(hostel)
            
            # Show all recommendations in expander
            with st.expander("View all recommendations"):
                for i, (_, hostel) in enumerate(recommendations.iterrows()):
                    st.write(f"**{i+1}. {hostel['Hostel']}**")
                    st.write(f"   Budget: UGX {int(hostel['Budget (UGX/sem)']):,} | Distance: {hostel['Distance (km)']} km | AI Score: {hostel.get('AI_Score', 3):.1f}/5")
                    if i < len(recommendations) - 1:
                        st.divider()
            
            # Buttons for next actions
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Refine Search", use_container_width=True):
                    st.session_state.chatbot.reset()
                    st.session_state.messages = []
                    st.session_state.show_recommendations = False
                    st.session_state.recommendations = None
                    st.session_state.chat_complete = False
                    welcome_msg = "Let's refine your search. What would you like to change?"
                    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
                    st.rerun()
            
            with col2:
                if st.button("Show All Hostels", use_container_width=True):
                    st.session_state.show_all = True
            
            with col3:
                if st.button("New Search", use_container_width=True):
                    st.session_state.chatbot.reset()
                    st.session_state.messages = []
                    st.session_state.show_recommendations = False
                    st.session_state.recommendations = None
                    st.session_state.chat_complete = False
                    welcome_msg = "Starting a new search! Tell me what you're looking for."
                    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
                    st.rerun()
        
        # Show suggestion chips
        if not st.session_state.show_recommendations:
            col1, col2, col3, col4 = st.columns(4)
            suggestions = [
                ("Single room with wifi", "I need a single room with wifi"),
                ("Budget 300k near campus", "Looking for hostel within 300k near campus"),
                ("Female only hostel", "I want a female only hostel"),
                ("Private bathroom", "I need a private bathroom")
            ]
            
            for col, (label, text) in zip([col1, col2, col3, col4], suggestions):
                with col:
                    if st.button(label, use_container_width=True):
                        # Add user message
                        st.session_state.messages.append({"role": "user", "content": text})
                        
                        # Process with chatbot
                        chatbot = st.session_state.chatbot
                        
                        # Extract preferences
                        extracted = chatbot.update_preferences(text)
                        
                        # Check if we have all preferences
                        missing = chatbot.get_missing_preferences()
                        
                        if not missing:
                            # Get recommendations
                            recommendations = chatbot.get_recommendations()
                            if recommendations is not None and len(recommendations) > 0:
                                st.session_state.recommendations = recommendations
                                st.session_state.show_recommendations = True
                                st.session_state.chat_complete = True
                                
                                # Add bot response
                                response = chatbot.format_recommendation_message(recommendations)
                                st.session_state.messages.append({"role": "assistant", "content": response})
                            else:
                                st.session_state.messages.append({"role": "assistant", "content": "I couldn't find matching hostels. Could you adjust your preferences?"})
                        else:
                            # Ask next question
                            question = chatbot.get_next_question()
                            st.session_state.messages.append({"role": "assistant", "content": question})
                        
                        st.rerun()
    
    # Chat input
    if not st.session_state.show_recommendations:
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Process with chatbot
            chatbot = st.session_state.chatbot
            
            # Extract preferences
            extracted = chatbot.update_preferences(user_input)
            
            # Check if we have all preferences
            missing = chatbot.get_missing_preferences()
            
            if not missing:
                # Get recommendations
                recommendations = chatbot.get_recommendations()
                if recommendations is not None and len(recommendations) > 0:
                    st.session_state.recommendations = recommendations
                    st.session_state.show_recommendations = True
                    st.session_state.chat_complete = True
                    
                    # Add bot response
                    response = chatbot.format_recommendation_message(recommendations)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": "I couldn't find matching hostels. Could you adjust your preferences?"})
            else:
                # Ask next question
                question = chatbot.get_next_question()
                st.session_state.messages.append({"role": "assistant", "content": question})
            
            st.rerun()
    
    # Show all hostels if requested
    if 'show_all' in st.session_state and st.session_state.show_all:
        st.markdown("---")
        st.subheader("All Available Hostels")
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Budget (UGX/sem)": st.column_config.NumberColumn("Budget (UGX)", format="UGX %d")
            }
        )
        if st.button("Hide All Hostels"):
            st.session_state.show_all = False
            st.rerun()
    
    # Sidebar
    with st.sidebar:
        st.header("Current Preferences")
        
        if st.session_state.chatbot.preferences:
            prefs = st.session_state.chatbot.preferences
            st.write(f"**Budget:** UGX {prefs.get('budget', 'Not set'):,}" if 'budget' in prefs else "**Budget:** Not set")
            st.write(f"**Gender:** {prefs.get('gender', 'Not set')}")
            st.write(f"**Distance:** {prefs.get('distance', 'Not set')} km")
            st.write(f"**WiFi:** {prefs.get('wifi', 'Not set')}")
            st.write(f"**Water:** {prefs.get('water', 'Not set')}")
            st.write(f"**Security:** {prefs.get('security', 'Not set')}")
            st.write(f"**Room Type:** {prefs.get('room_type', 'Not set')}")
            st.write(f"**Bathroom:** {prefs.get('bathroom', 'Not set')}")
            st.write(f"**Kitchen:** {prefs.get('kitchen', 'Not set')}")
        else:
            st.write("No preferences set yet. Start chatting!")
        
        st.markdown("---")
        if st.button("Reset Chat", use_container_width=True):
            st.session_state.chatbot.reset()
            st.session_state.messages = []
            st.session_state.show_recommendations = False
            st.session_state.recommendations = None
            st.session_state.chat_complete = False
            welcome_msg = "Hello! I'm your hostel assistant. I'll help you find the perfect hostel based on your preferences.\n\nWhat type of hostel are you looking for?"
            st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
        **How to use:**
        1. Tell me what you're looking for
        2. I'll ask for missing details
        3. Get personalized recommendations
        4. Refine or start over anytime
        """)

# ---------------------------------------
# FOOTER
# ---------------------------------------
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px; font-size: 14px; border-top: 1px solid #e0e0e0; margin-top: 30px;">
Lira University Hostel AI Chatbot<br>
Powered by AI | Natural Language Processing | Smart Recommendations
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
