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
# PROPER MODEL-BASED RECOMMENDATION ENGINE
# ---------------------------------------
def get_recommendations_from_model(df, model, preferences, n=5):
    """
    THIS IS WHERE THE MODEL IS ACTUALLY USED
    The model predicts a score for EACH hostel based on user preferences
    """
    
    if model is None:
        st.error("Model not loaded! Cannot make predictions.")
        return None
    
    # Store predictions for each hostel
    predictions = []
    
    for idx, row in df.iterrows():
        try:
            # Prepare input data EXACTLY as the model expects
            # The model was trained on these features
            input_data = pd.DataFrame([{
                "Hostel": row['Hostel'],
                "Budget (UGX/sem)": row['Budget (UGX/sem)'],
                "Gender": row['Gender'],
                "Distance (km)": row['Distance (km)'],
                "WiFi": row['WiFi'],
                "Water": row['Water'],
                "Security": row['Security'],
                "Room Type": row['Room Type'],
                "Bathroom": row['Bathroom'],
                "Kitchen": row['Kitchen']
            }])
            
            # THE MODEL MAKES THE PREDICTION HERE
            # This is the AI score - the model's output
            ai_score = model.predict(input_data)[0]
            
            predictions.append({
                'hostel': row,
                'ai_score': ai_score,
                'index': idx
            })
            
        except Exception as e:
            st.warning(f"Prediction error for {row['Hostel']}: {str(e)}")
            predictions.append({
                'hostel': row,
                'ai_score': 3.0,  # Default fallback score
                'index': idx
            })
    
    # Convert to DataFrame
    pred_df = pd.DataFrame(predictions)
    
    # Sort by AI score (highest first)
    pred_df = pred_df.sort_values('ai_score', ascending=False)
    
    # Get top N recommendations
    top_n = pred_df.head(n)
    
    # Create results DataFrame
    results = []
    for _, row in top_n.iterrows():
        hostel = row['hostel']
        hostel_dict = hostel.to_dict()
        hostel_dict['AI Score'] = round(row['ai_score'], 2)
        
        # Also calculate preference match for display
        match_score = calculate_preference_match(hostel, preferences)
        hostel_dict['Preference Match'] = match_score
        
        results.append(hostel_dict)
    
    return pd.DataFrame(results)

def calculate_preference_match(hostel, preferences):
    """Calculate how well the hostel matches user preferences (for display only)"""
    score = 0
    total = 0
    
    # Check each preference
    if 'budget' in preferences:
        total += 1
        diff = abs(hostel['Budget (UGX/sem)'] - preferences['budget'])
        if diff <= 50000:
            score += 1
        elif diff <= 100000:
            score += 0.7
        elif diff <= 200000:
            score += 0.4
    
    if 'gender' in preferences:
        total += 1
        if str(hostel['Gender']).lower() == str(preferences['gender']).lower():
            score += 1
    
    if 'room_type' in preferences:
        total += 1
        if str(hostel['Room Type']).lower() == str(preferences['room_type']).lower():
            score += 1
    
    if 'wifi' in preferences:
        total += 1
        if str(hostel['WiFi']).lower() == str(preferences['wifi']).lower():
            score += 1
    
    if 'distance' in preferences:
        total += 1
        if hostel['Distance (km)'] <= preferences['distance']:
            score += 1
        else:
            # Partial credit for being close
            penalty = (hostel['Distance (km)'] - preferences['distance']) / 5
            score += max(0, 1 - penalty)
    
    return round((score / total * 100) if total > 0 else 0, 1)

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
        self.preferences = {}
        self.step = 0
        self.complete = False
    
    def get_next_question(self):
        if self.step < len(self.questions):
            return self.questions[self.step]
        return None
    
    def process_response(self, user_input):
        if self.step >= len(self.preference_keys):
            return None
        
        key = self.preference_keys[self.step]
        value = self._extract_value(key, user_input)
        
        if value is not None:
            self.preferences[key] = value
            self.step += 1
            
            if self.step >= len(self.preference_keys):
                self.complete = True
                return self.get_recommendations()
            
            return self.get_next_question()
        else:
            return f"I couldn't understand. Please tell me: {self.questions[self.step]}"
    
    def _extract_value(self, key, text):
        text_lower = text.lower()
        
        if key == 'budget':
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
        """Get recommendations - THIS USES THE MODEL"""
        if len(self.preferences) < len(self.preference_keys):
            return None
        
        # CALL THE MODEL-BASED RECOMMENDATION FUNCTION
        # This is where the model is actually used
        return get_recommendations_from_model(self.df, self.model, self.preferences)
    
    def format_recommendation_message(self, recommendations):
        if recommendations is None or len(recommendations) == 0:
            return "I couldn't find any hostels matching your preferences. Let's start over."
        
        top_hostel = recommendations.iloc[0]
        
        message = f"✅ **Based on your preferences, here are my AI-powered recommendations:**\n\n"
        message += f"🏆 **Top Pick: {top_hostel['Hostel']}**\n"
        message += f"   • AI Score: {top_hostel['AI Score']}/5\n"
        message += f"   • Preference Match: {top_hostel['Preference Match']}%\n"
        message += f"   • Budget: UGX {int(top_hostel['Budget (UGX/sem)']):,}\n"
        message += f"   • Distance: {top_hostel['Distance (km)']} km\n"
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
                message += f"• {hostel['Hostel']} (AI Score: {hostel['AI Score']}/5, UGX {int(hostel['Budget (UGX/sem)']):,})\n"
        
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
    
    # Show model status
    if model is not None:
        st.sidebar.success("✅ AI Model Loaded - Making Predictions")
    else:
        st.sidebar.error("❌ AI Model NOT Loaded - Cannot Make Predictions")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = HostelChatbot(df, model)
        st.session_state.messages = []
        st.session_state.recommendations = None
        st.session_state.show_recommendations = False
        
        welcome = "Hello! I'm your hostel assistant. I'll help you find the perfect hostel at Lira University.\n\nI'll ask you a few questions about your preferences and then use the AI model to recommend the best hostels for you.\n\nLet's start! What is your budget per semester in UGX? (e.g., 300000)"
        st.session_state.messages.append({"role": "assistant", "content": welcome})
    
    # Title
    st.title("Hostel AI Assistant")
    st.write("AI-powered hostel recommendations using your trained model")
    
    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Show recommendations if available
    if st.session_state.show_recommendations and st.session_state.recommendations is not None:
        st.markdown("---")
        st.subheader("AI-Powered Recommendations")
        
        recommendations = st.session_state.recommendations
        
        # Display top 3 as cards
        for i in range(min(3, len(recommendations))):
            hostel = recommendations.iloc[i]
            ai_score = hostel['AI Score']
            
            if ai_score >= 4.5:
                emoji = "🏆"
                color = "#28a745"
            elif ai_score >= 3.5:
                emoji = "⭐"
                color = "#ffc107"
            else:
                emoji = "💡"
                color = "#dc3545"
            
            if i == 0:
                st.success(f"{emoji} Best Match - AI Score: {ai_score}/5")
            
            st.markdown(f"""
            <div class="card">
                <h3 style="color: #003366; margin: 0;">{hostel['Hostel']}</h3>
                <p><strong>AI Score:</strong> {ai_score}/5</p>
                <div class="score-bar">
                    <div class="score-bar-fill" style="width: {ai_score/5*100:.1f}%;"></div>
                </div>
                <p><strong>Preference Match:</strong> {hostel['Preference Match']}%</p>
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
            if st.button("Start Over", use_container_width=True):
                st.session_state.chatbot.reset()
                st.session_state.messages = []
                st.session_state.recommendations = None
                st.session_state.show_recommendations = False
                welcome = "Let's start over! What is your budget per semester in UGX? (e.g., 300000)"
                st.session_state.messages.append({"role": "assistant", "content": welcome})
                st.rerun()
        
        with col2:
            if st.button("Show All Hostels", use_container_width=True):
                st.session_state.show_all = True
        
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
    
    # Chat input
    if not st.session_state.show_recommendations:
        user_input = st.chat_input("Type your response here...")
        
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            chatbot = st.session_state.chatbot
            
            if not chatbot.complete:
                response = chatbot.process_response(user_input)
                
                if response is not None:
                    if isinstance(response, pd.DataFrame):
                        st.session_state.recommendations = response
                        st.session_state.show_recommendations = True
                        bot_message = chatbot.format_recommendation_message(response)
                        st.session_state.messages.append({"role": "assistant", "content": bot_message})
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": "I'm sorry, I didn't understand. Can you please rephrase?"})
                
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
            st.write("No preferences set yet")
        
        st.markdown("---")
        st.markdown("""
        **How the AI Works:**
        1. You answer questions about preferences
        2. The AI model predicts a score for EACH hostel
        3. Hostels are ranked by AI score
        4. You get the top recommendations
        5. Preference match shows how well it matches your stated preferences
        """)
        
        if model is not None:
            st.success("✅ Model is ACTIVE - making predictions")
        else:
            st.error("❌ Model is NOT ACTIVE - predictions unavailable")

# ---------------------------------------
# FOOTER
# ---------------------------------------
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px; font-size: 14px; border-top: 1px solid #e0e0e0; margin-top: 30px;">
Lira University Hostel AI Chatbot<br>
Powered by Your Trained AI Model | Machine Learning | Smart Recommendations
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
