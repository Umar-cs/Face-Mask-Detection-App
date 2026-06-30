import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model = load_model('facemask_model.h5')

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(201, 250))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Rescale to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit app
st.title("Face Mask Detection App")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Make predictions if an image is uploaded
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    input_image = preprocess_image(uploaded_file)

    # Make predictions
    prediction = model.predict(input_image)

    # Display the prediction
    if prediction[0] > 0.5:
        st.success("Face Mask Detected!")
    else:
        st.error("No Face Mask Detected.")
