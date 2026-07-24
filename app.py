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
    layout="wide"
)


# ---------------------------------------
# LOAD MODEL & DATA
# ---------------------------------------

@st.cache_resource
def load_model():

    return joblib.load(
        "hostel_ai_model.pkl"
    )


@st.cache_data
def load_data():

    df = pd.read_excel(
        "Lira_University_Hostel_Dataset.xlsx"
    )


    df["Budget (UGX/sem)"] = (
        df["Budget (UGX/sem)"]
        .astype(str)
        .str.replace(",", "")
        .str.replace("UGX","")
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



try:

    model = load_model()

    df = load_data()


except Exception as e:

    st.error(
        "Application loading failed"
    )

    st.write(e)

    st.stop()



# ---------------------------------------
# CSS
# ---------------------------------------

st.markdown(
"""
<style>

.card{

background:white;

padding:20px;

border-radius:15px;

box-shadow:0px 3px 10px rgba(0,0,0,0.1);

}

</style>

""",
unsafe_allow_html=True
)



# ---------------------------------------
# TITLE
# ---------------------------------------

st.title(
" Lira University[LU] Hostel Recommendation AI"
)


st.write(
"""
An intelligent accommodation assistant that helps students
find hostels based on their personal preferences.
"""
)



# ---------------------------------------
# AI CHAT ASSISTANT
# ---------------------------------------

st.divider()


st.subheader(
"🤖 Chat With Hostel AI"
)


user_message = st.text_area(

"Describe your ideal hostel",

placeholder=
"I am a female student looking for a hostel near campus with WiFi, private bathroom and a budget of 350000"

)



if st.button(
"Find Hostel Using AI"
):


    if user_message.strip()=="":


        st.warning(
        "Please describe your hostel requirements."
        )


    else:


        # NLP extraction

        preferences = extract_preferences(
            user_message
        )


        st.success(
        "AI understood your requirements"
        )


        with st.expander(
        "View extracted preferences"
        ):

            st.json(
            preferences
            )



        # ---------------------------------------
        # HOSTEL MATCHING
        # ---------------------------------------


        results = df.copy()



        results["Difference"] = abs(

            results["Budget (UGX/sem)"]

            -

            preferences["Budget (UGX/sem)"]

        )



        results["Match Score"] = 0



        # Matching rules


        for column in [

            "Gender",

            "WiFi",

            "Room Type",

            "Bathroom",

            "Kitchen"

        ]:


            results.loc[

            results[column]==preferences[column],

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



        # ---------------------------------------
        # DISPLAY RESULT
        # ---------------------------------------


        st.divider()


        st.subheader(
        "🏆 Recommended Hostel"
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

            "💰 Budget:",

            f"UGX {int(recommended['Budget (UGX/sem)']):,}"

            )


            st.write(

            "📍 Distance:",

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


            st.write(
            "WiFi:",
            recommended["WiFi"]
            )


            st.write(
            "Water:",
            recommended["Water"]
            )


            st.write(
            "Security:",
            recommended["Security"]
            )


            st.write(
            "Room:",
            recommended["Room Type"]
            )


            st.write(
            "Bathroom:",
            recommended["Bathroom"]
            )


            st.write(
            "Kitchen:",
            recommended["Kitchen"]
            )


            st.markdown(

            "</div>",

            unsafe_allow_html=True

            )



        # ---------------------------------------
        # AI EXPLANATION
        # ---------------------------------------


        st.divider()


        st.subheader(
        "🤖 Why AI Selected This Hostel"
        )


        reasons=[]



        if recommended["Difference"] < 50000:

            reasons.append(
            "✅ The hostel price matches your budget."
            )


        if recommended["WiFi"]==preferences["WiFi"]:

            reasons.append(
            "✅ It has the WiFi option you requested."
            )


        if recommended["Bathroom"]==preferences["Bathroom"]:

            reasons.append(
            "✅ It matches your bathroom preference."
            )


        if recommended["Room Type"]==preferences["Room Type"]:

            reasons.append(
            "✅ It matches your room preference."
            )


        if recommended["Distance (km)"] <= preferences["Distance (km)"]:

            reasons.append(
            "✅ It is within your preferred distance."
            )



        for reason in reasons:

            st.write(reason)



        if match >= 80:

            st.success(
            "Recommendation confidence: HIGH"
            )

        elif match >=50:

            st.info(
            "Recommendation confidence: MEDIUM"
            )

        else:

            st.warning(
            "Recommendation confidence: LOW"
            )



# ---------------------------------------
# DATA VIEW
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

st.caption(
"Lira University Hostel AI | Machine Learning + NLP"
)
