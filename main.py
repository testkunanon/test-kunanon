import streamlit as st
import numpy as np
from PIL import Image # For image processing
import tensorflow as tf # For loading and running Keras/TensorFlow model

# --- Custom CSS for enhanced styling ---
# This block injects custom CSS into the Streamlit app to override default styles
# and apply a more polished look.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="st-emotion"] {
        font-family: 'Inter', sans-serif;
        color: #2c3e50; /* Dark text for readability */
    }

    /* Main background and container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 900px; /* Limit width for better readability */
    }

    /* Header styling */
    h1 {
        color: #3498db; /* Primary blue for title */
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
        font-size: 2.5em;
    }

    h2 {
        color: #2c3e50;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    h3 {
        color: #34495e;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }

    /* Markdown text styling */
    .stMarkdown p {
        font-size: 1.1em;
        line-height: 1.6;
        color: #34495e;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    /* Custom file uploader button */
    .st-emotion-cache-1c7y2vl { /* Target Streamlit's file uploader container */
        display: none; /* Hide default uploader */
    }
    .custom-file-upload {
        display: inline-block;
        background-color: #2ecc71; /* Green button */
        color: white;
        padding: 12px 25px;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        font-size: 1.2em;
        font-weight: 600;
        transition: background-color 0.3s ease, transform 0.2s ease, box-shadow 0.3s ease;
        box-shadow: 0 5px 15px rgba(46, 204, 113, 0.4);
        margin-top: 1rem;
        margin-bottom: 2rem;
        text-align: center;
        width: fit-content;
        margin-left: auto;
        margin-right: auto;
    }
    .custom-file-upload:hover {
        background-color: #27ae60;
        transform: translateY(-2px);
        box-shadow: 0 7px 20px rgba(46, 204, 113, 0.6);
    }

    /* Image display styling */
    .stImage > img {
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
        border: 2px solid #ecf0f1;
    }

    /* Spinner styling */
    .stSpinner > div > div {
        color: #3498db;
    }
    .stSpinner > div > div > div {
        border-top-color: #3498db;
    }

    /* Results styling */
    .stAlert {
        border-radius: 10px;
        padding: 1rem 1.5rem;
        font-size: 1.1em;
        font-weight: 600;
    }
    .stAlert.st-emotion-cache-1f81n9p { /* Targeting success alert */
        background-color: #e6ffe6; /* Lighter green */
        color: #27ae60;
        border-left: 8px solid #2ecc71;
    }
    .stAlert.st-emotion-cache-1f81n9p + .stAlert.st-emotion-cache-1f81n9p { /* Targeting info alert after success */
        background-color: #e6f7ff; /* Lighter blue */
        color: #3498db;
        border-left: 8px solid #3498db;
    }
    
    /* Probability details styling */
    .st-emotion-cache-1r6dm1x { /* Target column container */
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        margin-bottom: 15px;
        transition: transform 0.2s ease;
    }
    .st-emotion-cache-1r6dm1x:hover {
        transform: translateY(-5px);
    }
    .st-emotion-cache-1r6dm1x h4 { /* Probability disease name */
        margin-bottom: 0.5rem;
        color: #2c3e50;
        font-weight: 600;
        font-size: 1.1em;
    }
    .st-emotion-cache-1r6dm1x .stProgress { /* Progress bar styling */
        margin-top: 0.5rem;
    }
    .st-emotion-cache-1r6dm1x .stProgress > div > div > div {
        background-color: #3498db !important; /* Blue progress bar */
    }
    .st-emotion-cache-1r6dm1x p { /* Probability percentage */
        text-align: right;
        font-size: 0.9em;
        color: #555;
        margin-top: 0.2rem;
    }

    /* Disclaimer styling */
    .stMarkdown > div > p > strong {
        color: #e74c3c; /* Red for warning text */
    }
    .stMarkdown > div > p {
        font-size: 0.9em;
        color: #7f8c8d;
        text-align: center;
    }
    .stMarkdown > div {
        border-top: 2px solid #ecf0f1;
        padding-top: 1rem;
        margin-top: 2rem;
    }

</style>
""", unsafe_allow_html=True)

# --- 1. Load Deep Learning Model (Cached) ---
@st.cache_resource
def load_my_dl_model():
    try:
        # Assuming the model file 'best_ham.keras' is in the 'model/' folder relative to this script.
        model = tf.keras.models.load_model("best_ham.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"ไม่สามารถโหลดโมเดลได้: {e}") # Cannot load model
        st.stop() # Stop the app if model loading fails

# Load the model when the app starts
model = load_my_dl_model()

# Define class names/diseases (IMPORTANT: The order must match your model's prediction order)
# Added emojis for visual appeal.
class_names_map = {
    'akiec': '☀️ Actinic Keratoses (AKIEC) ภาวะผิวหนังผิดปกติที่อาจเป็นมะเร็ง',
    'bcc': '🦠 Basal Cell Carcinoma (BCC) มะเร็งผิวหนังชนิดเบซัลเซลล์',
    'bkl': '🩹 Benign Keratosis-like Lesions (BKL) รอยโรคที่ไม่ใช่มะเร็ง เช่น กระแดด หรือ seborrheic keratoses',
    'df': '🔵 Dermatofibroma (DF) เนื้องอกใต้ผิวหนังที่ไม่เป็นอันตราย',
    'mel': '⚫ Melanoma (MEL) มะเร็งผิวหนังชนิดร้ายแรง',
    'nv': '🟠 Melanocytic Nevi (NV) ไฝหรือปาน',
    'vasc': '🩸 Vascular Lesions (VASC) ความผิดปกติของเส้นเลือด เช่น หลอดเลือดฝอยแตก'
}
# Create a list of class names in the correct order for prediction indexing
class_names_ordered = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# --- 2. User Interface (UI) with Streamlit ---
st.set_page_config(
    layout="centered", # Use "wide" for a wider layout
    page_title="AI วินิจฉัยโรคผิวหนัง", # Page title
    page_icon="🩺" # Page icon
)

st.title("🩺 AI วินิจฉัยโรคผิวหนังเบื้องต้น")
st.markdown("""
<div style="text-align: center; font-size: 1.1em; color: #34495e; margin-bottom: 2rem;">
    แอปพลิเคชันนี้ใช้โมเดล Deep Learning เพื่อวิเคราะห์รูปภาพผิวหนังของคุณ
    และให้การวินิจฉัยเบื้องต้นเกี่ยวกับสภาพผิวหนังที่เป็นไปได้
</div>
""", unsafe_allow_html=True)

# Custom file uploader button
# We hide the default st.file_uploader and use a styled label for the input
uploaded_file = st.file_uploader(
    " ", # Empty label to hide default text
    type=["jpg", "jpeg", "png"],
    key="file_uploader_hidden" # Unique key for the widget
)



# --- 3. Processing Logic when a file is uploaded ---
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='รูปภาพที่คุณอัปโหลด', use_column_width=True)
    st.write("---") # Horizontal line separator

    # Show processing message with a spinner
    with st.spinner("กำลังวิเคราะห์รูปภาพ... โปรดรอสักครู่"): # Analyzing image... Please wait
        try:
            # Read and preprocess the image for the model
            # IMPORTANT: Ensure image size and normalization match your model's expectations!
            # E.g., if your model expects 224x224 pixels and normalization by dividing by 255.0
            image = Image.open(uploaded_file).convert("RGB") # Ensure it's RGB
            image = image.resize((224, 224)) # Resize
            image_array = np.array(image) # Convert to NumPy array
            image_array = image_array / 255.0 # Normalize pixel values to 0-1
            # Add Batch Dimension (1, height, width, channels)
            image_array = np.expand_dims(image_array, axis=0)

            # Make prediction with the model
            predictions = model.predict(image_array)
            # The result will be an array of probabilities, e.g., [0.01, 0.05, 0.90, 0.02, 0.02]

            # Find the class with the highest probability
            predicted_class_index = np.argmax(predictions)
            # Get the short code and then the descriptive name
            predicted_class_short_code = class_names_ordered[predicted_class_index]
            predicted_disease_name = class_names_map.get(predicted_class_short_code, "ไม่ทราบโรค") # Unknown disease
            confidence_score = predictions[0][predicted_class_index] * 100 # Convert to percentage

            # --- 4. Display Analysis Results ---
            st.success(f"**ผลการวินิจฉัยเบื้องต้น:** `{predicted_disease_name}`") # Initial diagnosis
            st.info(f"**ความน่าจะเป็น:** `{confidence_score:.2f}%`") # Probability

            st.subheader("รายละเอียดความน่าจะเป็น:") # Probability Details
            
            # Use columns for better layout of probabilities
            # Create a list of columns to distribute items evenly
            cols_for_probs = st.columns(2) 
            
            for i, prob in enumerate(predictions[0]):
                disease_short_code = class_names_ordered[i]
                disease_full_name = class_names_map.get(disease_short_code, disease_short_code)
                
                # Display in columns, alternating between the two columns
                with cols_for_probs[i % 2]: 
                    st.markdown(f"**{disease_full_name}**") # Display disease name with emoji
                    st.progress(float(prob)) # Use st.progress for a visual bar (requires float)
                    st.write(f"({prob*100:.2f}%)") # Display percentage

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการประมวลผลรูปภาพ: {e}") # Error processing image
            st.write("โปรดตรวจสอบให้แน่ใจว่าเป็นรูปภาพที่ถูกต้องและลองใหม่อีกครั้ง") # Please ensure it's a valid image and try again

# --- 5. Important Disclaimer ---
st.markdown("""
---
**⚠️ คำเตือนสำคัญ:**

แอปพลิเคชันนี้สร้างขึ้นเพื่อวัตถุประสงค์ในการศึกษาและเป็นเครื่องมือช่วยวินิจฉัยเบื้องต้นเท่านั้น
ผลการวิเคราะห์ที่ได้เป็นเพียงการวินิจฉัยเบื้องต้นจากโมเดล AI และ **ไม่สามารถใช้ทดแทนคำแนะนำ
การวินิจฉัย หรือการรักษาจากแพทย์ผิวหนังผู้เชี่ยวชาญได้**

**โปรดปรึกษาแพทย์ผู้มีคุณสมบัติเหมาะสมเพื่อการวินิจฉัยและวางแผนการรักษาที่แม่นยำและถูกต้อง**
""")
