import streamlit as st
from pneunomia import PneumoniaModel
from PIL import Image

st.set_page_config(layout="centered")
st.title("Prediction Results")

images = st.session_state.get("images")

if not images:
    st.warning("No images found.")
    st.stop()

model = PneumoniaModel("grad_model.keras")

for i, image in enumerate(images):
    with st.container():
        file = Image.open(image)
        pred, output = model.pipeline_PIL(file)

        if output is None:
            st.warning("Grad-CAM not available for this image")
            st.image(file, caption="Original Image")
        else:
            if(pred > 0.5):
                st.text("Prediction: PNEUMONIA")
                st.text(f"Probability: {pred * 100:.2f}%")
            else:
                st.text("Prediction: NORMAL")
                st.text(f"Probability: {(1 - pred) * 100:.2f}%")

            cols = st.columns(2, gap="medium")
            with cols[0]:
                st.image(image, caption="Original X-ray")
            with cols[1]:
                st.image(output, caption="Highlighted")