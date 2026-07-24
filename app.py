import streamlit as st
import pandas as pd
import joblib
from nlp_engine import extract_preferences

# ---------------------------------------
# PAGE CONFIG
# ---------------------------------------
st.set_page_config(
    page_title="Lira University Hostel AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------
# LOAD MODEL & DATA
# ---------------------------------------
model = joblib.load("hostel_ai_model.pkl")

df = pd.read_excel("Lira_University_Hostel_Dataset.xlsx")

# Clean budget
df["Budget (UGX/sem)"] = (
    df["Budget (UGX/sem)"]
    .astype(str)
    .str.replace(",", "")
    .str.replace("UGX", "")
    .str.strip()
)

df["Budget (UGX/sem)"] = pd.to_numeric(df["Budget (UGX/sem)"])

df["Kitchen"] = df["Kitchen"].fillna(df["Kitchen"].mode()[0])

# ---------------------------------------
# CUSTOM CSS
# ---------------------------------------
st.markdown("""
<style>

.main{
    background:#f5f7fa;
}

.block-container{
    padding-top:2rem;
}

h1{
    color:#003366;
}

.card{
    background:white;
    padding:20px;
    border-radius:12px;
    box-shadow:0px 2px 8px rgba(0,0,0,0.08);
}

.footer{
    text-align:center;
    color:gray;
    padding:30px;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------
# TITLE
# ---------------------------------------

st.title("Lira University Hostel Recommendation System")

st.write(
"""
Use Artificial Intelligence to discover the hostel that best matches
your accommodation preferences around Lira University.
"""
)

# ---------------------------------------
# SIDEBAR
# ---------------------------------------

st.sidebar.header("Student Preferences")

budget = st.sidebar.number_input(
    "Budget (UGX/semester)",
    min_value=150000,
    max_value=1000000,
    value=300000,
    step=10000
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Mixed","Female Only","Male Only"]
)

distance = st.sidebar.slider(
    "Maximum Distance (km)",
    0.1,
    5.0,
    1.0
)

wifi = st.sidebar.selectbox(
    "WiFi",
    ["Yes","No"]
)

water = st.sidebar.selectbox(
    "Water Availability",
    [
        "Always Available",
        "Sometimes Interrupted",
        "Irregular"
    ]
)

security = st.sidebar.selectbox(
    "Security",
    [
        "24/7 Guard + CCTV",
        "Security Guard",
        "Gated Only",
        "Basic"
    ]
)

room = st.sidebar.selectbox(
    "Room Type",
    [
        "Single",
        "Double",
        "Triple",
        "Quad"
    ]
)

bathroom = st.sidebar.selectbox(
    "Bathroom",
    [
        "Private",
        "Shared"
    ]
)

kitchen = st.sidebar.selectbox(
    "Kitchen",
    [
        "Private",
        "Shared"
    ]
)
# ---------------------------------------
# AI CHAT ASSISTANT
# ---------------------------------------

st.divider()

st.subheader("🤖 Chat With Hostel AI")

st.write(
    "Describe your ideal hostel and AI will understand your preferences."
)


user_message = st.text_area(
    "Example: I am a female student looking for a hostel near campus with WiFi, private bathroom and a budget of 350000"
)


if st.button("Find Hostel Using AI"):

    if user_message.strip() == "":

        st.warning(
            "Please describe the hostel you are looking for."
        )

    else:

        preferences = extract_preferences(
            user_message
        )


        st.success(
            "AI understood your requirements:"
        )


        st.json(
            preferences
        )


# ---------------------------------------
# PREDICT
# ---------------------------------------

if st.sidebar.button("Find Best Hostel"):

    input_df = pd.DataFrame({

        "Hostel":["Unknown"],

        "Budget (UGX/sem)":[budget],

        "Gender":[gender],

        "Distance (km)":[distance],

        "WiFi":[wifi],

        "Water":[water],

        "Security":[security],

        "Room Type":[room],

        "Bathroom":[bathroom],

        "Kitchen":[kitchen]

    })

    score = model.predict(input_df)[0]

    # Filter matching hostels

    filtered = df.copy()

    filtered = filtered[
        (filtered["Gender"]==gender) &
        (filtered["WiFi"]==wifi) &
        (filtered["Room Type"]==room)
    ]

    if len(filtered)==0:
        filtered = df

    filtered["Difference"] = abs(filtered["Budget (UGX/sem)"]-budget)

    filtered = filtered.sort_values(
        by=["Difference","Recommendation Score"],
        ascending=[True,False]
    )

    hostel = filtered.iloc[0]

    st.success("Recommendation Generated Successfully")

    col1,col2=st.columns([1,1])

    with col1:

        st.markdown("<div class='card'>",unsafe_allow_html=True)

        st.subheader("Recommended Hostel")

        st.write("###",hostel["Hostel"])

        st.write("**Recommendation Score:**",round(score,2),"/5")

        st.write("**Budget:** UGX {:,}".format(int(hostel["Budget (UGX/sem)"])))

        st.write("**Distance:**",hostel["Distance (km)"],"km")

        st.markdown("</div>",unsafe_allow_html=True)

    with col2:

        st.markdown("<div class='card'>",unsafe_allow_html=True)

        st.subheader("Facilities")

        st.write("WiFi:",hostel["WiFi"])

        st.write("Water:",hostel["Water"])

        st.write("Security:",hostel["Security"])

        st.write("Room Type:",hostel["Room Type"])

        st.write("Bathroom:",hostel["Bathroom"])

        st.write("Kitchen:",hostel["Kitchen"])

        st.markdown("</div>",unsafe_allow_html=True)

    st.subheader("AI Recommendation")

    if score>=4.5:

        st.info(
        """
        This hostel is an excellent match for your
        preferences and is highly recommended.
        """
        )

    elif score>=3.5:

        st.info(
        """
        This hostel matches most of your
        preferences and is recommended.
        """
        )

    else:

        st.warning(
        """
        The available hostel does not fully
        match your preferences.
        Consider adjusting your requirements.
        """
        )

# ---------------------------------------
# DATA PREVIEW
# ---------------------------------------

st.divider()

st.subheader("Available Hostels")

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True
)

# ---------------------------------------
# FOOTER
# ---------------------------------------

st.markdown("""
<div class='footer'>
Lira University Hostel Recommendation System<br>
Artificial Intelligence | Streamlit | Scikit-Learn
</div>
""",unsafe_allow_html=True)
