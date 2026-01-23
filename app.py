import streamlit as st

st.set_page_config(
    page_title="Pneumonia Detection with Explainable DL",
    layout="wide"
)

st.title("Pneumonia Detection with Explainable Deep Learning")

uploaded_files = st.file_uploader(
    "Choose files to upload", type=["jpg", "png"], accept_multiple_files=True
)

if uploaded_files:
    if st.button("Predict", type="primary", width="stretch"):
        st.session_state["images"] = uploaded_files
        st.switch_page("pages/results.py")

    num_cols = 3

    for i in range(0, len(uploaded_files), num_cols):
        row_files = uploaded_files[i:i + num_cols]

        with st.container():
            cols = st.columns(num_cols, gap="medium")

            for col, file in zip(cols, row_files):
                with col:
                    st.image(
                        file,
                        use_container_width=True,
                        caption=file.name
                    )