import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* ==========================================
   PAGE BACKGROUND
   ========================================== */

.stApp {
    background: linear-gradient(
        135deg,
        #f8f9ff 0%,
        #eef3ff 50%,
        #f7f0ff 100%
    );
}


/* ==========================================
   MAIN CONTENT
   ========================================== */

.block-container {
    padding-top: 0.8rem !important;
    padding-bottom: 0.3rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}


/* ==========================================
   MAIN TITLE
   ========================================== */

h1 {
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    margin-top: 0 !important;
    margin-bottom: 0.2rem !important;
}


/* ==========================================
   SUBTITLE
   ========================================== */

p {
    font-size: 0.9rem !important;
    margin-bottom: 0.3rem !important;
}


/* ==========================================
   HEADINGS
   ========================================== */

h2, h3 {
    font-size: 1.2rem !important;
    margin-top: 0.2rem !important;
    margin-bottom: 0.2rem !important;
}


/* ==========================================
   FILE UPLOADER
   ========================================== */

.stFileUploader {
    margin-top: 0 !important;
    margin-bottom: 0.3rem !important;
}


/* ==========================================
   METRIC
   ========================================== */

div[data-testid="stMetric"] {
    padding: 0 !important;
    margin: 0 !important;
}

div[data-testid="stMetricValue"] {
    font-size: 1.4rem !important;
}

div[data-testid="stMetricLabel"] {
    font-size: 0.8rem !important;
}


/* ==========================================
   SUCCESS BOX
   ========================================== */

div[data-testid="stAlert"] {
    padding: 0.4rem !important;
    margin: 0.2rem 0 !important;
}


/* ==========================================
   PROGRESS BARS
   ========================================== */

div[data-testid="stProgress"] {
    margin-top: -5px !important;
    margin-bottom: -5px !important;
}


/* ==========================================
   COLUMNS
   ========================================== */

[data-testid="column"] {
    padding-left: 0.3rem !important;
    padding-right: 0.3rem !important;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL
# ==========================================

model = tf.keras.models.load_model(
    "hand_gesture_model.keras"
)


# ==========================================
# GESTURE ORDER
# ==========================================

gestures = [
    "fist",
    "palm",
    "peace",
    "thumbs_up"
]


# ==========================================
# TITLE
# ==========================================

st.title("✋ Hand Gesture Recognition")

st.write(
    "Upload a hand image and the CNN will recognize the gesture."
)


# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "📷 Upload Hand Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ==========================================
# PREDICTION
# ==========================================

if uploaded_file is not None:

    # ======================================
    # OPEN IMAGE
    # ======================================

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # ======================================
    # RESIZE IMAGE FOR MODEL
    # ======================================

    resized_image = image.resize(
        (128, 128)
    )


    # Convert image to NumPy
    img_array = np.array(
        resized_image
    )


    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    # ======================================
    # MODEL PREDICTION
    # ======================================

    prediction = model.predict(
        img_array,
        verbose=0
    )[0]


    # Find highest probability
    index = np.argmax(
        prediction
    )


    # Get gesture name
    gesture = gestures[index]


    # Calculate confidence
    confidence = (
        prediction[index] * 100
    )


    # ======================================
    # MAIN TWO COLUMNS
    # ======================================

    left_column, right_column = st.columns(
        [1, 1],
        gap="small"
    )


    # ======================================
    # LEFT COLUMN - IMAGE
    # ======================================

    with left_column:

        st.subheader(
            "🖼️ Uploaded Image"
        )

        st.image(
            image,
            width=180
        )


    # ======================================
    # RIGHT COLUMN - PREDICTION
    # ======================================

    with right_column:

        st.subheader(
            "🔮 Prediction"
        )


        # Main prediction
        st.success(
            f"Gesture: {gesture.upper()}"
        )


        # Confidence
        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


        # ==================================
        # PREDICTION DETAILS
        # ==================================

        st.subheader(
            "📊 Prediction Details"
        )


        # Four columns
        col1, col2, col3, col4 = st.columns(
            4
        )


        prediction_columns = [
            col1,
            col2,
            col3,
            col4
        ]


        # ==================================
        # DISPLAY ALL PREDICTIONS
        # ==================================

        for i in range(4):

            percentage = (
                prediction[i] * 100
            )


            with prediction_columns[i]:

                st.write(
                    f"**{gestures[i].upper()}**"
                )

                st.caption(
                    f"{percentage:.1f}%"
                )

                st.progress(
                    float(prediction[i])
                )


# ==========================================
# NO IMAGE MESSAGE
# ==========================================

else:

    st.info(
        "👆 Please upload a hand image to get a prediction."
    )