import streamlit as st
import pandas as pd
import joblib
from nlp_engine import extract_preferences


# ---------------------------------------
# PAGE CONFIG
# ---------------------------------------

st.set_page_config(
    page_title="Lira University Hostel AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------
# LOAD MODEL & DATA
# ---------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("hostel_ai_model.pkl")


@st.cache_data
def load_data():

    df = pd.read_excel(
        "Lira_University_Hostel_Dataset.xlsx"
    )

    df["Budget (UGX/sem)"] = (
        df["Budget (UGX/sem)"]
        .astype(str)
        .str.replace(",", "")
        .str.replace("UGX", "")
        .str.strip()
    )

    df["Budget (UGX/sem)"] = pd.to_numeric(
        df["Budget (UGX/sem)"],
        errors="coerce"
    )


    df["Kitchen"] = df["Kitchen"].fillna(
        df["Kitchen"].mode()[0]
    )

    return df



model = load_model()
df = load_data()



# ---------------------------------------
# CSS
# ---------------------------------------

st.markdown(
"""
<style>

.main{
background:#f5f7fa;
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
""",
unsafe_allow_html=True
)



# ---------------------------------------
# TITLE
# ---------------------------------------

st.title(
"Lira University Hostel Recommendation System"
)


st.write(
"""
Use Artificial Intelligence to discover the hostel
that best matches your accommodation preferences.
"""
)



# =====================================================
# NLP AI CHAT ASSISTANT
# =====================================================


st.divider()

st.subheader(
"🤖 Chat With Hostel AI"
)


user_message = st.text_area(
"Describe the hostel you want",
placeholder=
"I am a female student looking for a hostel near campus. My budget is 350000, I need WiFi and private bathroom."
)


if st.button("Find Hostel Using AI"):


    if user_message.strip()=="":

        st.warning(
            "Please describe your hostel requirements."
        )


    else:


        preferences = extract_preferences(
            user_message
        )


        st.success(
            "AI understood your requirements"
        )


        st.json(
            preferences
        )


        results = df.copy()


        results["Difference"] = abs(
            results["Budget (UGX/sem)"]
            -
            preferences["Budget (UGX/sem)"]
        )


        results["Match Score"] = 0



        for col in [
            "Gender",
            "WiFi",
            "Room Type",
            "Bathroom",
            "Kitchen"
        ]:


            results.loc[
                results[col]==preferences[col],
                "Match Score"
            ] += 1



        results = results.sort_values(
            by=[
                "Match Score",
                "Difference",
                "Recommendation Score"
            ],
            ascending=[
                False,
                True,
                False
            ]
        )



        recommended = results.iloc[0]



        st.divider()

        st.subheader(
            "🏠 AI Recommended Hostel"
        )


        col1,col2 = st.columns(2)



        with col1:

            st.markdown(
                "<div class='card'>",
                unsafe_allow_html=True
            )


            st.write(
                "##",
                recommended["Hostel"]
            )


            match = (
                recommended["Match Score"]/5
            )*100


            st.success(
                f"AI Match Score: {match:.0f}%"
            )


            st.write(
                "Budget:",
                f"UGX {int(recommended['Budget (UGX/sem)']):,}"
            )


            st.write(
                "Distance:",
                recommended["Distance (km)"],
                "km"
            )


            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )



        with col2:

            st.markdown(
                "<div class='card'>",
                unsafe_allow_html=True
            )


            st.subheader(
                "Facilities"
            )


            for item in [
                "WiFi",
                "Water",
                "Security",
                "Room Type",
                "Bathroom",
                "Kitchen"
            ]:

                st.write(
                    item+":",
                    recommended[item]
                )


            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )



        # Explanation

        st.divider()

        st.subheader(
            "🤖 Why AI Selected This Hostel"
        )


        reasons=[]


        if recommended["Difference"] < 50000:

            reasons.append(
                "✅ Matches your budget preference"
            )


        if recommended["WiFi"] == preferences["WiFi"]:

            reasons.append(
                "✅ Provides your preferred WiFi option"
            )


        if recommended["Bathroom"] == preferences["Bathroom"]:

            reasons.append(
                "✅ Matches bathroom preference"
            )


        if recommended["Room Type"] == preferences["Room Type"]:

            reasons.append(
                "✅ Matches room preference"
            )


        if recommended["Distance (km)"] <= preferences["Distance (km)"]:

            reasons.append(
                "✅ Within your preferred distance"
            )


        for r in reasons:

            st.write(r)



        if match >=80:

            st.success(
                "Recommendation Confidence: HIGH"
            )

        elif match >=50:

            st.info(
                "Recommendation Confidence: MEDIUM"
            )

        else:

            st.warning(
                "Recommendation Confidence: LOW"
            )



# =====================================================
# ORIGINAL SIDEBAR RECOMMENDER
# =====================================================


st.sidebar.header(
"Student Preferences"
)


budget = st.sidebar.number_input(
"Budget (UGX/semester)",
150000,
1000000,
300000,
10000
)


gender = st.sidebar.selectbox(
"Gender",
df["Gender"].unique()
)


distance = st.sidebar.slider(
"Maximum Distance (km)",
0.1,
5.0,
1.0
)


wifi = st.sidebar.selectbox(
"WiFi",
df["WiFi"].unique()
)


room = st.sidebar.selectbox(
"Room Type",
df["Room Type"].unique()
)


bathroom = st.sidebar.selectbox(
"Bathroom",
df["Bathroom"].unique()
)


kitchen = st.sidebar.selectbox(
"Kitchen",
df["Kitchen"].unique()
)



if st.sidebar.button(
"Find Best Hostel"
):


    filtered=df.copy()


    filtered["Difference"] = abs(
        filtered["Budget (UGX/sem)"]
        -
        budget
    )


    filtered=filtered.sort_values(
        by=[
            "Difference",
            "Recommendation Score"
        ],
        ascending=[
            True,
            False
        ]
    )


    hostel=filtered.iloc[0]


    st.success(
        "Recommendation Generated Successfully"
    )


    st.subheader(
        "🏠 Recommended Hostel"
    )


    st.write(
        hostel["Hostel"]
    )


    st.write(
        "Budget:",
        f"UGX {int(hostel['Budget (UGX/sem)']):,}"
    )


    st.write(
        "Distance:",
        hostel["Distance (km)"],
        "km"
    )



# ---------------------------------------
# DATA PREVIEW
# ---------------------------------------

st.divider()

st.subheader(
"Available Hostels"
)


st.dataframe(
    df,
    use_container_width=True,
    hide_index=True
)



# ---------------------------------------
# FOOTER
# ---------------------------------------

st.markdown(
"""
<div class='footer'>

Lira University Hostel Recommendation System<br>

Artificial Intelligence | NLP | Machine Learning | Streamlit

</div>
""",
unsafe_allow_html=True
)
