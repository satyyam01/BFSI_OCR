import streamlit as st
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Custom CSS for styling
def add_custom_css():
    st.markdown(
        """
        <style>
        body {
            background-color: #f9fafb;
            font-family: 'Arial', sans-serif;
        }
        .main-title {
            font-family: 'Helvetica Neue', sans-serif;
            color: #1e3a8a; /* Bright blue */
            text-align: center;
            margin-top: 50px;
            font-size: 3.5em;
            font-weight: bold;
        }
        .sub-title {
            font-family: 'Arial', sans-serif;
            color: #3b82f6; /* Dark gray */
            text-align: center;
            font-size: 1.5em;
            margin-bottom: 40px;
        }
        .option-box {
            padding: 30px;
            border: 3px solid #60a5fa; /* Light blue */
            border-radius: 15px;
            background-color: #e0f2fe; /* Very light blue */
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
            text-align: center;
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }
        .option-box:hover {
            transform: scale(1.05);
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.15);
        }
        .footer {
            font-size: 1.1em;
            color: #3b82f6; /* Vibrant blue */
            text-align: center;
            margin-top: 50px;
            font-weight: bold;
            padding: 20px 0;
            border-top: 2px solid #3b82f6;
        }
        .stButton>button {
            background-color: #3b82f6 !important; /* Bright blue button */
            color: white !important;
            border-radius: 10px;
            padding: 12px 24px;
            font-size: 1.2em;
            font-weight: bold;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #2563eb !important; /* Darker blue on hover */
        }
        .stSelectbox>div {
            font-size: 1.1em;
            background-color: #f0f4f8;
            border: 2px solid #d1d5db;
            border-radius: 8px;
            padding: 10px;
        }
        .app-icon {
            width: 120px;
            height: 120px;
            margin-bottom: 20px;
            border-radius: 50%;
        }
        .banner {
            width: 100%;
            height: 250px;
            object-fit: cover;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
            margin-top: 30px;
            margin-bottom: 40px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Gemini API query function

def main():
    add_custom_css()

    # Page title and description with additional elements
    st.markdown("<h1 class='main-title'>Welcome to Document Analysis App</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='sub-title'>Analyze your financial documents with ease. Choose an option below to get started!</p>",
        unsafe_allow_html=True,
    )

    # Display a banner image for a pop effect
    #st.image('https://via.placeholder.com/1200x250.png?text=Document+Analysis+App', use_column_width=True, caption="Analyze your financial documents effortlessly!")

    # Display an app icon with custom CSS class
    #st.markdown(
        #'<img src="https://via.placeholder.com/120x120.png" alt="App Icon" class="app-icon">',
        #unsafe_allow_html=True
    #)

    # Options for navigation
    app_options = {
        "Bank Statement Analysis": "pdf2csv-preprocessing.py",
        "Pay Slip Analysis": "paySlipCombined.py",
        "Invoice Analysis": "invoiceCombined.py",
        "Stock Trend Analysis": "APIcombined.py",
        "LLM Transaction Classifier": r"LLM.py"
    }

    # Display options in a styled dropdown
    selected_option = st.selectbox(
        "Select an option",
        list(app_options.keys()),
        help="Choose the type of document you want to analyze.",
    )

    # Button for analysis with hover effect
    if st.button("Analyze", use_container_width=True):
        selected_app = app_options[selected_option]
        os.system(f"python -m streamlit run {selected_app}")

    # Footer section with a more prominent design
    st.markdown(
        """
        <div class="footer">
            © 2025 Document Analysis App | Developed by Satyam Nautiyal
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
