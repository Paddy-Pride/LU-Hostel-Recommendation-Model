import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
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
    text-align: center;
    width: 100%;
    border: none;
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
# HELPER FUNCTIONS - MATCHING ORIGINAL MODEL
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
    """Get recommendations using the trained model - SAME AS ORIGINAL APP"""
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
                # Prepare input exactly like original app
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
                
                # Predict using model
                score = model.predict(input_df)[0]
                ai_scores.append(score)
            
            filtered['AI_Score'] = ai_scores
            
        except Exception as e:
            st.warning(f"Model prediction issue: {str(e)}")
            # Fallback
            filtered['AI_Score'] = 3.0  # Default score
    
    # Calculate match scores
    match_scores = []
    for _, row in filtered.iterrows():
        scores = get_match_scores(row, preferences)
        match_scores.append(scores)
    
    match_df = pd.DataFrame(match_scores)
    filtered = pd.concat([filtered.reset_index(drop=True), match_df], axis=1)
    
    # Calculate final score - SAME AS ORIGINAL APP
    if 'AI_Score' in filtered.columns:
        # Normalize AI score
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
# CHATBOT CLASS
# ---------------------------------------
class HostelChatbot:
    def __init__(self, df, model):
        self.df = df
        self.model = model
        self.preferences = {}
        self.step = 0
        self.questions = [
            "What is your budget per semester in UGX? (e.g., 300000)",
            "Which gender preference do you have? (Mixed, Female Only, Male Only)",
            "What is the maximum distance from campus in km? (e.g., 1.0)",
            "Do you need WiFi? (Yes/No)",
            "What water availability do you prefer? (Always Available, Sometimes Interrupted, Irregular)",
            "What security level do you prefer? (24/7 Guard + CCTV, Security Guard, Gated Only, Basic)",
            "What room type do you prefer? (Single, Double, Triple, Quad)",
            "Do you prefer private or shared bathroom?",
            "Do you prefer private or shared kitchen?"
        ]
        self.preference_keys = [
            'budget', 'gender', 'distance', 'wifi', 
            'water', 'security', 'room_type', 'bathroom', 'kitchen'
        ]
        self.complete = False
        
    def reset(self):
        """Reset chatbot state"""
        self.preferences = {}
        self.step = 0
        self.complete = False
    
    def get_next_question(self):
        """Get the next question to ask"""
        if self.step < len(self.questions):
            return self.questions[self.step]
        return None
    
    def process_response(self, user_input):
        """Process user response and update preferences"""
        if self.step >= len(self.preference_keys):
            return None
        
        # Get current preference key
        key = self.preference_keys[self.step]
        
        # Extract value based on key type
        value = self._extract_value(key, user_input)
        
        if value is not None:
            self.preferences[key] = value
            self.step += 1
            
            # Check if complete
            if self.step >= len(self.preference_keys):
                self.complete = True
                return self.get_recommendations()
            
            return self.get_next_question()
        else:
            # Could not extract, ask again
            return f"I couldn't understand. Please tell me: {self.questions[self.step]}"
    
    def _extract_value(self, key, text):
        """Extract value for a specific preference key"""
        text_lower = text.lower()
        
        if key == 'budget':
            # Extract budget
            patterns = [
                r'(\d+)\s*(?:thousand|k)',
                r'(?:ugx|shs)\s*(\d+)',
                r'budget\s*(\d+)',
                r'(\d+)\s*(?:million|m)'
            ]
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    try:
                        budget = int(match.group(1))
                        if 'million' in text_lower or 'm' in text_lower:
                            budget = budget * 1000000
                        elif budget < 1000:
                            budget = budget * 1000
                        return budget
                    except:
                        pass
            
            # Try direct number
            numbers = re.findall(r'\d+', text)
            if numbers:
                try:
                    budget = int(numbers[0])
                    if budget < 1000:
                        budget = budget * 1000
                    return budget
                except:
                    pass
            return None
        
        elif key == 'gender':
            if 'female' in text_lower or 'ladies' in text_lower:
                return 'Female Only'
            elif 'male' in text_lower or 'gentlemen' in text_lower:
                return 'Male Only'
            elif 'mixed' in text_lower or 'both' in text_lower:
                return 'Mixed'
            return None
        
        elif key == 'distance':
            patterns = [
                r'(\d+\.?\d*)\s*km',
                r'(\d+\.?\d*)\s*kilometer',
                r'within\s*(\d+\.?\d*)',
                r'near\s*(\d+\.?\d*)'
            ]
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    try:
                        return float(match.group(1))
                    except:
                        pass
            
            # Try direct number
            numbers = re.findall(r'\d+\.?\d*', text)
            if numbers:
                try:
                    dist = float(numbers[0])
                    if 0 < dist <= 10:
                        return dist
                except:
                    pass
            return None
        
        elif key == 'wifi':
            if 'yes' in text_lower or 'wifi' in text_lower:
                return 'Yes'
            elif 'no' in text_lower:
                return 'No'
            return None
        
        elif key == 'water':
            if 'always' in text_lower:
                return 'Always Available'
            elif 'sometimes' in text_lower or 'interrupted' in text_lower:
                return 'Sometimes Interrupted'
            elif 'irregular' in text_lower:
                return 'Irregular'
            return None
        
        elif key == 'security':
            if '24/7' in text_lower or 'cctv' in text_lower:
                return '24/7 Guard + CCTV'
            elif 'guard' in text_lower:
                return 'Security Guard'
            elif 'gated' in text_lower:
                return 'Gated Only'
            elif 'basic' in text_lower:
                return 'Basic'
            return None
        
        elif key == 'room_type':
            if 'single' in text_lower:
                return 'Single'
            elif 'double' in text_lower:
                return 'Double'
            elif 'triple' in text_lower:
                return 'Triple'
            elif 'quad' in text_lower:
                return 'Quad'
            return None
        
        elif key == 'bathroom':
            if 'private' in text_lower:
                return 'Private'
            elif 'shared' in text_lower:
                return 'Shared'
            return None
        
        elif key == 'kitchen':
            if 'private' in text_lower:
                return 'Private'
            elif 'shared' in text_lower:
                return 'Shared'
            return None
        
        return None
    
    def get_recommendations(self):
        """Get recommendations based on current preferences"""
        if len(self.preferences) < len(self.preference_keys):
            return None
        
        return get_recommendations(self.df, self.model, self.preferences)
    
    def format_recommendation_message(self, recommendations):
        """Format recommendations for display"""
        if recommendations is None or len(recommendations) == 0:
            return "I couldn't find any hostels matching your preferences. Let's start over."
        
        # Get AI score for top hostel
        top_hostel = recommendations.iloc[0]
        
        if self.model is not None and 'AI_Score' in top_hostel:
            ai_score = top_hostel['AI_Score']
        else:
            ai_score = top_hostel.get('Final_Score', 3) * 5 / 20
        
        # Build message
        message = f"✅ **Great! Based on your preferences, here are my recommendations:**\n\n"
        message += f"🏆 **Top Pick: {top_hostel['Hostel']}**\n"
        message += f"   • AI Score: {ai_score:.1f}/5\n"
        message += f"   • Budget: UGX {int(top_hostel['Budget (UGX/sem)']):,}\n"
        message += f"   • Distance: {top_hostel['Distance (km)']} km from campus\n"
        message += f"   • WiFi: {top_hostel['WiFi']}\n"
        message += f"   • Water: {top_hostel['Water']}\n"
        message += f"   • Security: {top_hostel['Security']}\n"
        message += f"   • Room Type: {top_hostel['Room Type']}\n"
        message += f"   • Bathroom: {top_hostel['Bathroom']}\n"
        message += f"   • Kitchen: {top_hostel['Kitchen']}\n\n"
        
        if len(recommendations) > 1:
            message += "**Alternatives:**\n"
            for i in range(1, min(4, len(recommendations))):
                hostel = recommendations.iloc[i]
                alt_score = hostel.get('AI_Score', 3)
                message += f"• {hostel['Hostel']} (Score: {alt_score:.1f}/5, UGX {int(hostel['Budget (UGX/sem)']):,})\n"
        
        message += "\nYou can click 'Show Recommendations' below to see detailed cards or click 'Start Over' to try different preferences."
        
        return message

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
        st.session_state.recommendations = None
        st.session_state.show_recommendations = False
        
        # Add welcome message
        welcome = "👋 Hello! I'm your hostel assistant. I'll help you find the perfect hostel at Lira University.\n\nI'll ask you a few questions about your preferences and then recommend the best hostels for you.\n\nLet's start! What is your budget per semester in UGX? (e.g., 300000)"
        st.session_state.messages.append({"role": "assistant", "content": welcome})
    
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
            st.markdown("---")
            st.subheader("Top Recommendations")
            
            recommendations = st.session_state.recommendations
            
            # Display top 3 as cards
            for i in range(min(3, len(recommendations))):
                hostel = recommendations.iloc[i]
                if i == 0:
                    st.success("🏆 Best Match")
                
                ai_score = hostel.get('AI_Score', 3)
                
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
            
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Start Over", use_container_width=True):
                    st.session_state.chatbot.reset()
                    st.session_state.messages = []
                    st.session_state.recommendations = None
                    st.session_state.show_recommendations = False
                    welcome = "👋 Let's start over! What is your budget per semester in UGX? (e.g., 300000)"
                    st.session_state.messages.append({"role": "assistant", "content": welcome})
                    st.rerun()
            
            with col2:
                if st.button("📋 Show All Hostels", use_container_width=True):
                    st.session_state.show_all = True
            
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
        
        # Chat input - only show if not complete
        if not st.session_state.show_recommendations:
            user_input = st.chat_input("Type your response here...")
            
            if user_input:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Process with chatbot
                chatbot = st.session_state.chatbot
                
                if not chatbot.complete:
                    # Process response
                    response = chatbot.process_response(user_input)
                    
                    if response is not None:
                        # Check if response is a recommendation (dict)
                        if isinstance(response, pd.DataFrame):
                            st.session_state.recommendations = response
                            st.session_state.show_recommendations = True
                            
                            # Format recommendation message
                            bot_message = chatbot.format_recommendation_message(response)
                            st.session_state.messages.append({"role": "assistant", "content": bot_message})
                        else:
                            # It's a question
                            st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": "I'm sorry, I didn't understand. Can you please rephrase?"})
                    
                    st.rerun()
        
        # Show current preferences in sidebar
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
                st.write("No preferences set yet")
            
            st.markdown("---")
            st.markdown("""
            **How to use:**
            1. Answer each question
            2. I'll collect your preferences
            3. Get AI-powered recommendations
            4. View detailed hostel cards
            5. Start over anytime
            """)

# ---------------------------------------
# FOOTER
# ---------------------------------------
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px; font-size: 14px; border-top: 1px solid #e0e0e0; margin-top: 30px;">
Lira University Hostel AI Chatbot<br>
Powered by AI | Machine Learning | Smart Recommendations
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
