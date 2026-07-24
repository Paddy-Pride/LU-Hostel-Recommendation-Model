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
# LOAD DATA
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
# STYLE
# ---------------------------------------

st.markdown(
"""
<style>

.card{

background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 2px 8px rgba(0,0,0,0.1);

}

.footer{

text-align:center;
color:gray;
padding:30px;

}

</style>

""",
unsafe_allow_html=True
)



# ---------------------------------------
# HEADER
# ---------------------------------------

st.title(
"🏠 Lira University Hostel Recommendation AI"
)


st.write(
"""
An AI-powered accommodation assistant that recommends
hostels based on student preferences.
"""
)



# =====================================================
# NLP AI ASSISTANT
# =====================================================


st.divider()

st.subheader(
"🤖 Chat With Hostel AI"
)


message = st.text_area(
"Describe the hostel you need",
placeholder=
"I am a female student looking for a hostel near campus. My budget is 400000, I need WiFi, private bathroom and security."
)



if st.button(
"Find My Hostel"
):


    if message.strip()=="":


        st.warning(
            "Please describe your hostel requirements."
        )


    else:


        preferences = extract_preferences(
            message
        )


        st.success(
            "AI understood your requirements"
        )


        with st.expander(
            "View AI extracted preferences"
        ):

            st.json(
                preferences
            )



        results = df.copy()



        # -----------------------------
        # SMART SCORING SYSTEM
        # -----------------------------


        budget = preferences["Budget (UGX/sem)"]


        results["Budget Difference"] = abs(
            results["Budget (UGX/sem)"] - budget
        )


        # Budget score 40%

        results["Budget Score"] = (
            1 -
            (
                results["Budget Difference"] /
                budget
            )
        ).clip(0,1)



        # Preference score 40%

        results["Preference Score"] = 0



        for col in [

            "Gender",
            "WiFi",
            "Room Type",
            "Bathroom",
            "Kitchen",
            "Security"

        ]:


            results.loc[
                results[col] == preferences[col],
                "Preference Score"
            ] += 1



        results["Preference Score"] = (
            results["Preference Score"] / 6
        )



        # Distance score 20%

        results["Distance Score"] = (

            1 -
            (
                results["Distance (km)"] / 5
            )

        ).clip(0,1)



        # Final AI score

        results["AI Score"] = (

            results["Budget Score"] * 0.4

            +

            results["Preference Score"] * 0.4

            +

            results["Distance Score"] * 0.2

        )



        results = results.sort_values(
            "AI Score",
            ascending=False
        )



        top_hostels = results.head(5)



        st.divider()


        st.subheader(
            "🏆 Top AI Recommendations"
        )



        for index, hostel in top_hostels.iterrows():


            with st.container():


                st.markdown(
                    "<div class='card'>",
                    unsafe_allow_html=True
                )


                st.subheader(
                    hostel["Hostel"]
                )


                score = hostel["AI Score"] * 100


                st.success(
                    f"AI Match: {score:.0f}%"
                )


                col1,col2,col3 = st.columns(3)


                with col1:

                    st.write(
                        "💰 Budget"
                    )

                    st.write(
                        f"UGX {int(hostel['Budget (UGX/sem)']):,}"
                    )


                with col2:

                    st.write(
                        "📍 Distance"
                    )

                    st.write(
                        str(hostel["Distance (km)"])+" km"
                    )


                with col3:

                    st.write(
                        "📶 WiFi"
                    )

                    st.write(
                        hostel["WiFi"]
                    )



                st.write(
                    "Why recommended:"
                )


                if hostel["Budget Difference"] < 50000:

                    st.write(
                        "✅ Matches your budget"
                    )


                if hostel["WiFi"] == preferences["WiFi"]:

                    st.write(
                        "✅ Has your preferred WiFi"
                    )


                if hostel["Bathroom"] == preferences["Bathroom"]:

                    st.write(
                        "✅ Matches bathroom preference"
                    )


                if hostel["Room Type"] == preferences["Room Type"]:

                    st.write(
                        "✅ Matches room preference"
                    )



                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )


                st.write("")



# =====================================================
# ORIGINAL SIDEBAR SEARCH
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


    temp = df.copy()


    temp["Difference"] = abs(
        temp["Budget (UGX/sem)"] -
        budget
    )


    temp=temp.sort_values(
        [
            "Difference",
            "Recommendation Score"
        ],
        ascending=[
            True,
            False
        ]
    )


    hostel=temp.iloc[0]


    st.success(
        "Recommendation Generated"
    )


    st.subheader(
        "🏠 Recommended Hostel"
    )


    st.write(
        hostel["Hostel"]
    )


    st.write(
        f"Budget: UGX {int(hostel['Budget (UGX/sem)']):,}"
    )



# ---------------------------------------
# DATA
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
<div class="footer">

Lira University Hostel Recommendation AI<br>
Machine Learning + NLP + Streamlit

</div>
""",
unsafe_allow_html=True
)
