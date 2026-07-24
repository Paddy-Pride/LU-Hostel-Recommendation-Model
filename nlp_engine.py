import re


def extract_preferences(text):

    text = text.lower()

    preferences = {

        "Budget (UGX/sem)": 300000,

        "Gender": "Mixed",

        "Distance (km)": 1.0,

        "WiFi": "No",

        "Water": "Always Available",

        "Security": "Security Guard",

        "Room Type": "Double",

        "Bathroom": "Shared",

        "Kitchen": "Shared"
    }


    # Budget
    numbers = re.findall(r'\d+', text)

    if numbers:

        amount = int(numbers[0])

        if amount < 10000:
            amount = amount * 1000

        preferences["Budget (UGX/sem)"] = amount



    # Gender

    if "female" in text or "girls" in text:

        preferences["Gender"] = "Female Only"


    elif "male" in text or "boys" in text:

        preferences["Gender"] = "Male Only"



    # WiFi

    if "wifi" in text or "internet" in text:

        preferences["WiFi"] = "Yes"



    # Bathroom

    if "private bathroom" in text:

        preferences["Bathroom"] = "Private"



    # Room

    if "single room" in text:

        preferences["Room Type"] = "Single"


    elif "double room" in text:

        preferences["Room Type"] = "Double"



    # Security

    if "cctv" in text:

        preferences["Security"] = "24/7 Guard + CCTV"


    return preferences
