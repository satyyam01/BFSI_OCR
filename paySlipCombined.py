import pytesseract
from PIL import Image
import sqlite3
import pandas as pd
import cv2  # For image preprocessing
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px



# Step 1: Extract Text from Image with Preprocessing
def extract_text_from_image(uploaded_file):
    """
    Extract text from an image using PyTesseract with preprocessing.
    """
    try:
        # Load the image using OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
        print(f"Error during text extraction: {e}")
        return ""


# Step 2: Save Extracted Text to Database
def save_text_to_db(text, db_path):
    """
    Save extracted text to an SQLite database.
    """
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
    """
    Retrieve the most recent text entry from the database.
    """
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('SELECT content FROM extracted_text ORDER BY id DESC LIMIT 1')
        result = c.fetchone()
        return result[0] if result else None
    finally:
        conn.close()


# Helper Function: Extract Number from Line
def extract_number_from_line(line):
    """
    Extract the first numerical value from a line of text.
    """
    import re
    match = re.search(r'\d+(\.\d+)?', line)
    return float(match.group()) if match else 0.0


# Step 4: Extract Values Based on Keywords
def extract_values(text, keyword_dict):
    """
    Extract allowances and deductions based on keywords.
    """
    data = {key: 0 for key in keyword_dict["allowances"]}
    data.update({key: 0 for key in keyword_dict["deductions"]})

    lines = text.lower().split("\n")
    for line in lines:
        matched = set()
        matched = set()
        for key, keywords in keyword_dict["allowances"].items():
            if any(kw in line and kw not in matched for kw in keywords):
                data[key] += extract_number_from_line(line)
                matched.update(keywords)

        for key, keywords in keyword_dict["deductions"].items():
            if any(kw in line for kw in keywords):
                data[key] += extract_number_from_line(line)
    return data


# Step 5: Calculate Summary (Gross, Deductions, Net)
def calculate_summary(data, keyword_dict):
    """
    Calculate gross salary, total deductions, and net salary.
    """
    gross_salary = sum(data[key] for key in keyword_dict["allowances"])
    total_deductions = sum(data[key] for key in keyword_dict["deductions"])
    net_salary = gross_salary - total_deductions
    data.update({"gross_salary": gross_salary, "total_deductions": total_deductions, "net_salary": net_salary})
    return data


# Step 6: Save Data to CSV
def visualize_data(data):
    """
    Save the extracted and calculated data to a CSV file.
    """
    st.title("Salary Components Visualization")
    st.markdown("Visuals for salary components such as allowances, deductions, and overall summary metrics.")

    data = pd.DataFrame([data])

    # Define the keyword dictionary
    keyword_dict = {
        "allowances": {
            "basic_pay": ["basic pay", "basic salary", "pay"],
            "travel_allowance": ["travel allowance", "conveyance", "transportation"],
            "dearness_allowance": ["dearness allowance", "da"],
            "house_rent_allowance": ["house rent allowance", "hra"],
            "medical_allowance": ["medical allowance", "medical", "healthcare"],
            "child_care_allowance": ["child care allowance", "childcare", "cca"],
            "meal_allowance": ["meal allowance", "food"],
        },
        "deductions": {
            "income_tax": ["tax", "income tax"],
            "retirement_insurance": ["retirement insurance", "retire", "retirement"],
        },
    }

    # Extract allowances and deductions
    allowances = {col: data[col].iloc[0] for col in data.columns if col in keyword_dict["allowances"]}
    deductions = {col: data[col].iloc[0] for col in data.columns if col in keyword_dict["deductions"]}

    gross_salary = data["gross_salary"].iloc[0]
    net_salary = data["net_salary"].iloc[0]
    total_deductions = data["total_deductions"].iloc[0]

    # Summary Metrics
    st.subheader("Summary Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Gross Salary", f"₹{gross_salary:,.2f}")
    col2.metric("Net Salary", f"₹{net_salary:,.2f}")
    col3.metric("Total Deductions", f"₹{total_deductions:,.2f}")

    # Pie Chart for Allowances
    st.subheader("Gross Salary Breakdown (Allowances)")
    fig1, ax1 = plt.subplots(figsize=(6, 6))  # Adjust size for formality
    allowance_labels = list(allowances.keys())
    allowance_values = list(allowances.values())
    ax1.pie(
        allowance_values,
        labels=allowance_labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=plt.cm.Paired.colors,
    )
    ax1.set_title("Allowances Composition")
    st.pyplot(fig1)

    # Pie Chart for Deductions
    st.subheader("Deductions Breakdown")
    fig2, ax2 = plt.subplots(figsize=(6, 6))  # Adjust size for formality
    deduction_labels = list(deductions.keys())
    deduction_values = list(deductions.values())
    ax2.pie(
        deduction_values,
        labels=deduction_labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=plt.cm.Set3.colors,
    )
    ax2.set_title("Deductions Composition")
    st.pyplot(fig2)

    # Bar Chart for Allowances and Deductions
    st.subheader("Comparison: Allowances vs Deductions")
    combined_labels = allowance_labels + deduction_labels
    combined_values = allowance_values + deduction_values
    fig3, ax3 = plt.subplots(figsize=(8, 4))  # Wider bar chart
    ax3.bar(combined_labels, combined_values, color=plt.cm.tab20.colors)
    ax3.set_title("Allowances and Deductions Comparison")
    ax3.set_ylabel("Amount (₹)")
    ax3.set_xticklabels(combined_labels, rotation=45, ha="right")
    st.pyplot(fig3)

    # Detailed Data Table
    st.subheader("Detailed Salary Data")
    st.dataframe(data)


# Step 7: Main Flow
def main(db_path):
    """
    Main function to run the entire flow.
    """

    st.title("Salary Components Visualization Dashboard")
    st.markdown("Visuals for salary components such as allowances, deductions, and overall summary metrics.")

    # Define the keyword dictionary
    keyword_dict = {
        "allowances": {
            "basic_pay": ["basic pay", "basic salary", "pay"],
            "travel_allowance": ["travel allowance", "conveyance", "transportation"],
            "dearness_allowance": ["dearness allowance", "da"],
            "house_rent_allowance": ["house rent allowance", "hra"],
            "medical_allowance": ["medical allowance", "medical", "healthcare"],
            "child_care_allowance": ["child care allowance", "childcare", "cca"],
            "meal_allowance": ["meal allowance", "food"],

            # "provident fund": ["ppf", "public provident fund", "EPF", "provident fund", "pf"]

        },
        "deductions": {
            "income_tax": ["tax", "income tax"],
            "retirement_insurance": ["retirement insurance", "retire", "retirement"],
        },
    }

    # File Upload Feature
    uploaded_file = st.file_uploader("Upload your CSV file", type=["jpg"])

    if uploaded_file is not None:

        # Extract text from the image
        text = extract_text_from_image(uploaded_file)

        # Save extracted text to the database
        save_text_to_db(text, db_path)

        # Query text from the database
        extracted_text = query_text_from_db(db_path)
        if not extracted_text:
            print("No text found in the database. Exiting...")

        # Extract values from the text
        data = extract_values(extracted_text, keyword_dict)

        # Calculate salary summary
        data = calculate_summary(data, keyword_dict)

        data = pd.DataFrame([data])
        st.dataframe(data)

        # Extract allowances and deductions
        allowances = {col: data[col].iloc[0] for col in data.columns if col in keyword_dict["allowances"]}
        deductions = {col: data[col].iloc[0] for col in data.columns if col in keyword_dict["deductions"]}

        gross_salary = data["gross_salary"].iloc[0]
        net_salary = data["net_salary"].iloc[0]
        total_deductions = data["total_deductions"].iloc[0]

        # Summary Metrics
        st.subheader("Summary Metrics")
        gross_salary = data["gross_salary"].sum()
        net_salary = data["net_salary"].sum()
        total_deductions = data["total_deductions"].sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Gross Salary", f"₹{gross_salary:,.2f}")
        col2.metric("Net Salary", f"₹{net_salary:,.2f}")
        col3.metric("Total Deductions", f"₹{total_deductions:,.2f}")

        # Allowances and Deductions
        st.subheader("Breakdown of Allowances and Deductions")

        allowances = {col: data[col].sum() for col in data.columns if col in keyword_dict["allowances"]}
        deductions = {col: data[col].sum() for col in data.columns if col in keyword_dict["deductions"]}

        # Allowances Pie Chart
        allowance_labels = list(allowances.keys())
        allowance_values = list(allowances.values())
        allowance_pie = px.pie(
            values=allowance_values,
            names=allowance_labels,
            title="Allowances Breakdown",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(allowance_pie, use_container_width=True)

        # Deductions Pie Chart
        deduction_labels = list(deductions.keys())
        deduction_values = list(deductions.values())
        deduction_pie = px.pie(
            values=deduction_values,
            names=deduction_labels,
            title="Deductions Breakdown",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(deduction_pie, use_container_width=True)

        # Comparison Bar Chart
        st.subheader("Comparison: Allowances vs Deductions")
        combined_labels = allowance_labels + deduction_labels
        combined_values = allowance_values + deduction_values
        comparison_bar = px.bar(
            x=combined_labels,
            y=combined_values,
            title="Comparison of Allowances and Deductions",
            labels={"x": "Components", "y": "Amount (₹)"},
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(comparison_bar, use_container_width=True)

        # Detailed Data Table
        st.subheader("Detailed Salary Data")
        st.dataframe(data, use_container_width=True)

        # Download Data
        st.subheader("Download Filtered Data")
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name="salary_components.csv",
            mime="text/csv"
        )

        # Footer
        st.sidebar.markdown("---")
        st.sidebar.write("Developed by Satyam Nautiyal")
        st.sidebar.write("Contact: satyamnautiyal74@gmail.com")
    else:
        st.warning("Please upload a CSV file to begin.")


# Run the program
if __name__ == "__main__":
    db_path = "salary_data.db"
    main(db_path)

