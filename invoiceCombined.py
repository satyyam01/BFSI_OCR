import streamlit as st
import pytesseract
from PIL import Image
import sqlite3
import pandas as pd
import cv2
import numpy as np
import re
import os
import plotly.express as px

# Step 1: Extract Text from Image with Preprocessing
def extract_text_from_image(image):
    try:
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

        # Apply Otsu's binarization
        _, binarized_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Convert the processed image back to PIL format for PyTesseract
        pil_image = Image.fromarray(binarized_image)

        # Extract text using PyTesseract
        text = pytesseract.image_to_string(pil_image)

        # Remove commas from the extracted text
        text = text.replace(",", "")
        return text
    except Exception as e:
        st.error(f"Error during text extraction: {e}")
        return ""

# Step 2: Save Extracted Text to Database
def save_text_to_db(text, db_path):
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('CREATE TABLE IF NOT EXISTS extracted_text (id INTEGER PRIMARY KEY, content TEXT)')
        c.execute('INSERT INTO extracted_text (content) VALUES (?)', (text,))
        conn.commit()
    finally:
        conn.close()

# Step 3: Query Text from Database
def query_text_from_db(db_path):
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('SELECT content FROM extracted_text ORDER BY id DESC LIMIT 1')
        result = c.fetchone()
        return result[0] if result else None
    finally:
        conn.close()

# Helper Function: Extract GST and Total Values
def extract_gst_and_total_values(text, gst_keywords_dict, total_keywords):
    gst_data = {"SGST": 0.0, "CGST": 0.0, "IGST": 0.0, "Total": 0.0}
    for gst_type, keywords in gst_keywords_dict.items():
        for keyword in keywords:
            pattern = rf"{keyword}\s*[:\-]?\s*₹?\s*([\d,]+\.\d+|\d+)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                gst_data[gst_type] = float(match.group(1).replace(",", ""))

    for keyword in total_keywords:
        pattern = rf"{keyword}\s*[:\-]?\s*₹?\s*([\d,]+\.\d+|\d+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            gst_data["Total"] = float(match.group(1).replace(",", ""))

    return gst_data

# Step 6: Streamlit App
def main():
    st.set_page_config(page_title="Invoice Analysis Dashboard", page_icon="📜", layout="wide")
    st.title("Invoice Analysis Dashboard")
    st.markdown("### Upload an invoice to extract and analyze GST data")

    gst_keywords_dict = {
        "SGST": ["sgst", "state gst", "sgst tax"],
        "CGST": ["cgst", "central gst", "cgst tax"],
        "IGST": ["igst", "integrated gst", "igst tax"]
    }

    total_keywords = ["total invoice value", "total amount", "grand total"]

    uploaded_image = st.file_uploader("Upload Invoice Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Invoice", use_column_width=True)

        # Extract text from image
        image = Image.open(uploaded_image)
        text = extract_text_from_image(image)

        if text:
            st.subheader("Extracted Text")
            st.text_area("Text from Invoice", text, height=200)

            # Save to database and query
            db_path = "gst_data.db"
            save_text_to_db(text, db_path)
            extracted_text = query_text_from_db(db_path)

            if extracted_text:
                gst_data = extract_gst_and_total_values(extracted_text, gst_keywords_dict, total_keywords)

                # Display GST summary
                st.subheader("GST Summary")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("SGST", f"₹{gst_data['SGST']:.2f}")
                col2.metric("CGST", f"₹{gst_data['CGST']:.2f}")
                col3.metric("IGST", f"₹{gst_data['IGST']:.2f}")
                col4.metric("Total Invoice Value", f"₹{gst_data['Total']:.2f}")

                # Save GST data to CSV
                csv_path = "gst_data.csv"
                pd.DataFrame([gst_data]).to_csv(csv_path, index=False)

                # Visualization
                st.subheader("Visualizations")
                st.markdown("#### GST Distribution")
                gst_df = pd.DataFrame(
                    {"GST Type": ["SGST", "CGST", "IGST"], "Amount": [gst_data["SGST"], gst_data["CGST"], gst_data["IGST"]]}
                )

                # Pie Chart
                pie_chart = px.pie(gst_df, names="GST Type", values="Amount", title="GST Distribution")
                st.plotly_chart(pie_chart, use_container_width=True)

                # Bar Chart
                st.markdown("#### GST Breakdown")
                bar_chart = px.bar(gst_df, x="GST Type", y="Amount", text="Amount", title="GST Breakdown", color="GST Type")
                st.plotly_chart(bar_chart, use_container_width=True)

                # Download CSV
                st.download_button(
                    label="Download GST Data as CSV",
                    data=open(csv_path, "rb").read(),
                    file_name="gst_data.csv",
                    mime="text/csv"
                )


if __name__ == "__main__":
    main()
