import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import cv2, io
import easyocr

def sharpen_image(image_array, sharp_level):
    img = image_array

    mask = np.full((sharp_level,sharp_level),-1)
    mask[sharp_level//2][sharp_level//2] = sharp_level * sharp_level

    # mask = np.array([
    #     [-1,-1,-1],
    #     [-1,9,-1],
    #     [-1,-1,-1]
    # ])
    sharpened = cv2.filter2D(img, -1, mask)

    result = Image.fromarray(sharpened)

    # Save the result to a BytesIO object
    buf = io.BytesIO()
    result.save(buf, format='PNG')
    buf.seek(0)

    # Convert the BytesIO object to a PIL Image
    result = Image.open(buf)
    st.image(result, caption='Sharpened image.', use_column_width=True)

def smoothen_image(image_array, smooth_level):
    img = image_array
    
    smoothened = cv2.blur(img, (smooth_level,smooth_level))

    result = Image.fromarray(smoothened)

    # Save the result to a BytesIO object
    buf = io.BytesIO()
    result.save(buf, format='PNG')
    buf.seek(0)

    # Convert the BytesIO object to a PIL Image
    result = Image.open(buf)
    st.image(result, caption='Smoothened image.', use_column_width=True)

def denoise_image(image_array, denoise_level):

    img = cv2.cvtColor(np.array(image_array), cv2.COLOR_RGB2BGR)

    # Check if the image is not already in 8-bit, 3-channel format
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Apply Median filtering
    denoised = cv2.medianBlur(img, denoise_level)

    result = Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))

    # Save the result to a BytesIO object
    buf = io.BytesIO()
    result.save(buf, format='PNG')
    buf.seek(0)

    # Convert the BytesIO object to a PIL Image
    result = Image.open(buf)
    st.image(result, caption='Denoised image.', use_column_width=True)

def contrast_brightness(image_array, alpha, beta):
    img = image_array

    # Alpha = Contrast control
    # Beta = Brightness control

    # call convertScaleAbs function
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    result = Image.fromarray(adjusted)

    # Save the result to a BytesIO object
    buf = io.BytesIO()
    result.save(buf, format='PNG')
    buf.seek(0)

    # Convert the BytesIO object to a PIL Image
    result = Image.open(buf)
    st.image(result, caption='Contrast & brightness adjustment.', use_column_width=True)

def text_recognition(image):
    
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image)

    if len(result) > 0:
        table_data = []
        for i in range(len(result)):
            r = []
            for j in range(len(result[i])):
                if j>0:
                    r.append(result[i][j])
            table_data.append(r)

        df = pd.DataFrame(table_data, columns = ["Extracted Text","Confidence Level"])
        st.write(df)
    else:
        st.write("No text could be detected in the image.")

def histogram_equalization(image_array):
    img = image_array

    # convert from RGB color-space to YCrCb
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

    # convert back to RGB color-space from YCrCb
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

    result = Image.fromarray(equalized_img)

    # Save the result to a BytesIO object
    buf = io.BytesIO()
    result.save(buf, format='PNG')
    buf.seek(0)

    # Convert the BytesIO object to a PIL Image
    result = Image.open(buf)
    st.image(result, caption='Equalized image.', use_column_width=True)


st.sidebar.title("Image Processing Tool")
st.sidebar.write("")
st.sidebar.write("")

uploaded_file = st.sidebar.file_uploader("Upload an image to get started.", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    
    image_array = np.array(image)

    sharp_level = 1   
    with st.sidebar.expander("Sharpen image"):
        sharp_level = st.slider('Select the sharpness level.\n\nLevel 1 indicates the original image.', min_value=1, max_value=15, step=2, value=1, key="sharp")
    
    smooth_level = 1
    with st.sidebar.expander("Smoothen image"):
        smooth_level = st.slider('Select the smoothness level.\n\nLevel 1 indicates the original image.', min_value=1, max_value=15, step=2, value=1, key="smooth")

    denoise_level = 1
    with st.sidebar.expander("Denoise image"):
        denoise_level = st.slider('Select the denoise level.\n\nLevel 1 indicates the original image.', min_value=1, max_value=15, step=2, value=1, key="denoise")
    
    alpha = 1.0
    beta = 0.0
    with st.sidebar.expander("Adjust contrast & brightness"):
        alpha = st.slider('Select the contrast (alpha) level.\n\nLevel 1 indicates the original image.', min_value= 0.0, max_value= 10.0, step=0.1, value=1.0, key="contrast")

        beta = st.slider('Select the brightness (beta) level.\n\nLevel 0 indicates the original image.', min_value= -255, max_value= 255, step=5, value=0, key="brightness")

    if st.sidebar.button("Text Extraction"):
        text_recognition(image_array)

    if st.sidebar.button("Histogram equalization"):
        histogram_equalization(image_array)
    
    if sharp_level != 1:
        sharpen_image(image_array, sharp_level)
        
    if smooth_level != 1:
        smoothen_image(image_array, smooth_level)
    
    if denoise_level != 1:
        denoise_image(image_array, denoise_level)
    
    if alpha != 1.0  or beta != 0:
       contrast_brightness(image_array, alpha, beta)
    
else:
    st.write("An easier way to fine-tune your photos. Powered by Streamlit, OpenCV & EasyOCR.")
    st.image('home.jpg', use_column_width=True)