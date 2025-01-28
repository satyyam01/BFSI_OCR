import pandas as pd
import streamlit as st
import pdfplumber
import csv
import plotly.express as px



def clean_and_format_dataframe(df):
    """
    Cleans and formats the extracted DataFrame by removing unwanted rows,
    merging multi-line descriptions, and formatting columns.
    """
    # Ensure 'Transaction Date' column exists before processing
    if 'Transaction Date' in df.columns:
        for i in df.index:
            if isinstance(df["Transaction Date"][i], float):
                df["Transaction Date"][i] = df["Transaction Date"][i]
        df["Transaction Date"] = df["Transaction Date"]

    # Remove empty rows and header rows present in between
    delete = []
    headers = ["Date", "Description", "Credit", "Debit", "Balance"]
    for i in df.index:
        row = df.iloc[i, :].tolist()
        nan_count = sum(1 for val in row if pd.isna(val))
        if nan_count == len(df.columns):
            delete.append(i)
        if any(header in row for header in headers):
            delete.append(i)
    df.drop(delete, axis=0, inplace=True)

    # Merge multi-line descriptions
    last_valid_index = 0
    delete = []
    for i in df.index:
        if pd.isna(df["Value Date"][i]) and isinstance(df["Description"][i], str):
            df.at[last_valid_index, "Description"] += " " + df["Description"][i]
            delete.append(i)
        else:
            last_valid_index = i
    df.drop(delete, axis=0, inplace=True)

    # Format Credit and Debit columns
    for col in ["Credit", "Debit"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: str(x).replace(",", "") if x is not None else '0')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Remove the prefix from the 'Value Date' column
    if "Value Date" in df.columns:
        df["Value Date"] = df["Value Date"].apply(lambda x: x[3:] if isinstance(x, str) else x)

    # Reorganize columns
    df = df[["Transaction Date", "Value Date", "Description", "Debit", "Credit", "Balance"]]

    return df



def extract_transactions_from_pdf(pdf_file):
    """
    Extracts transactions from a PDF file and returns them as a cleaned DataFrame.
    """
    data = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines = text.split('\n')
                for line in lines:
                    data.append(line.split())

    # Dynamically determine the column structure
    max_columns = max(len(row) for row in data)
    columns = ['Transaction Date', 'Value Date', 'Description', 'Debit', 'Credit', 'Balance']
    if max_columns > len(columns):
        columns.extend([f'Extra Column {i}' for i in range(max_columns - len(columns))])

    df = pd.DataFrame(data, columns=columns)
    cleaned_df = clean_and_format_dataframe(df)

    return cleaned_df


def process_csv1(df):
    # Filter the DataFrame to include only rows 12–33 and 40–70 (adjusting for 0-based indexing)
    # data = df.iloc[12:34]
    data1 = df.iloc[12:34]

    # Step 2: Rename columns appropriately (if they exist in the header)
    data1.rename(
        columns={
            'Description/Narration': 'Description',
            'Chq./Ref. No.': 'Debit',
            'Unnamed: 13': 'Credit',
            'Unnamed: 17': 'Balance'
        }, inplace=True
    )

    # Step 3: Remove columns that are entirely empty or unnamed
    data1 = data1.dropna(how='all', axis=1)

    # Step 4: Filter rows to include only the 2 pages of transactions (adjust range as needed)
    # Assuming 20 rows per page for this example
    data1 = data1.head(60)

    # Step 5: Drop rows that are completely empty
    data1 = data1.dropna(how='all')

    # Step 1: Rename relevant columns
    rename_map1 = {
        'ACCOUNT STATEMENT': 'Date',
        'Unnamed: 2': 'Description',
        'Unnamed: 10': 'Debit',
        'Unnamed: 14': 'Credit',
        'Unnamed: 18': 'Balance'
    }

    data1.rename(columns=rename_map1, inplace=True)
    # df_modified.rename(columns=rename_map2, inplace=True)

    # Step 2: Drop empty or unnamed columns
    data1 = data1.loc[:, ~data1.columns.str.contains('^Unnamed') | data1.notna().any()]

    # Step 3: Filter only the first page of transactions (assumption: specific row range or logical filtering)
    # Example: Keeping first 10 rows as "first page" logic (you can adjust this as per your need)
    data1 = data1.head(30)

    # Drop the 'Unnamed: 7' and 'Unnamed: 9' columns
    columns_to_drop = ['Unnamed: 7', 'Unnamed: 9']
    data1.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

    # Replace all occurrences of '-' with '0'
    data1.replace('-', '0', inplace=True)

    # Convert the 'Date' column to the correct date format
    # Assuming the 'Date' column is in 'DD/MM/YYYY' or similar format
    #data1['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)

    return data1


def process_csv2(df):
    # Filter the DataFrame to include only rows 12–33 and 40–70 (adjusting for 0-based indexing)
    # data = df.iloc[12:34]
    data2 = df.iloc[40:71]

    # Step 2: Rename columns appropriately (if they exist in the header)
    data2.rename(
        columns={
            'ACCOUNT STATEMENT': 'Date',
            'Unnamed: 1': 'Description',
            'Unnamed: 7': 'Debit',
            'Unnamed: 9': 'Credit',
            'Unnamed: 10': 'Balance'
        }, inplace=True
    )

    # Step 3: Remove columns that are entirely empty or unnamed
    data2 = data2.dropna(how='all', axis=1)

    # Step 4: Filter rows to include only the 2 pages of transactions (adjust range as needed)
    # Assuming 20 rows per page for this example
    data2 = data2.head(30)

    # Step 5: Drop rows that are completely empty
    data2 = data2.dropna(how='all')

    # Drop the 'Unnamed: 7' and 'Unnamed: 9' columns
    columns_to_drop = ['Unnamed: 4', 'Unnamed: 6', 'Unnamed: 12']
    data2.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

    # Replace all occurrences of '-' with '0'
    data2.replace('-', '0', inplace=True)

    # Convert the 'Date' column to the correct date format
    # Assuming the 'Date' column is in 'DD/MM/YYYY' or similar format
    data2['Date'] = pd.to_datetime(data2['Date'], errors='coerce', dayfirst=True)

    return data2

def merge_df(df1, df2):
    # Concatenate the two DataFrames
    df = pd.concat([df1, df2], ignore_index=True)
    return df

# Custom CSS for styling
def add_custom_css():
    st.markdown(
        """
        <style>
        body {
            background-color: #ffffff;
        }
        .stSidebar {
            background-color: #3b82f6; /* Light blue sidebar */
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6 {
            color: #1e3a8a; /* Bright blue titles */
        }
        .stTitle {
            color: #1e3a8a; /* Bright blue title */
            font-size: 2.5em;
        }
        .stSubheader {
            color: #3b82f6; /* Dark gray subheaders */
            font-size: 1.4em;
        }
        .stMetric {
            background-color: #e0f2fe; /* Light blue metric cards */
            padding: 10px;
            border-radius: 10px;
        }
        .stDownloadButton>button {
            background-color: #3b82f6 !important; /* Bright blue button */
            color: white !important;
            border-radius: 5px;
            padding: 8px 16px;
            font-size: 1.1em;
        }
        .stDownloadButton>button:hover {
            background-color: #2563eb !important; /* Darker blue on hover */
        }
        .stPieChart {
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            border-radius: 10px;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Define the dictionary with labels and associated keywords
transaction_labels = {
    "UPI": ["MADHURI", "VIJAY", "SHRIYAM", "AJAY", "JEENA", "SHILA", "PRAKASH"],
    "Investment": ["MUTINDIAN", "CLEARING"],
    "Grocery": ["DILIP SINGH", "P H AND SONS", "KAMAL"],
    "Food": ["SUBWAY", "VINAYAK AGENCIES", "BEVERAGE FOR FRIENDS", "MAHESH KUMAR MEENA", "BAKERS", "DIALOG", "JAGDISH",
             "ZOMATO", "SWIGGY", "MOHANJI"],
    "Travel": ["HOTEL HIGHWAY KING", "HOTEL", "UBER", "OLA"],
    "Charges": ["SMS", "CHARGE"],
    "Education": ["MANIPAL", "GOOD HOST"],
    "Health": ["NYKAA", "SUNITA", "KOTHARI"],
    "Subscription": ["SPOTIFY"],
    "Rewards": ["REWARDS"]
}


def labeller(df):
    """
    Processes the input DataFrame, assigns categories based on transaction_labels,
    and returns a DataFrame with the required columns.

    Args:
        df (pd.DataFrame): Input DataFrame with columns 'Date', 'Description', 'Debit', 'Credit', 'Balance'.

    Returns:
        pd.DataFrame: Updated DataFrame with an additional 'Category' column.
    """
    # Ensure required columns exist
    required_columns = ['Date', 'Description', 'Debit', 'Credit', 'Balance']
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    # Handle missing or non-string values in the 'Description' column
    df['Description'] = df['Description'].fillna('').astype(str).str.strip().str.upper()

    # Assign categories based on transaction_labels
    def assign_category(description):
        for label, keywords in transaction_labels.items():
            if any(keyword in description for keyword in keywords):
                return label
        return "Others"

    # Apply the categorization function to the DataFrame
    df['Category'] = df['Description'].apply(assign_category)

    # Return the updated DataFrame
    return df




def visualize(df):
        # Ensure date is in the correct format
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Fill NaN values in Credit and Debit with 0 and ensure they are numeric
        df['Credit'] = pd.to_numeric(df['Credit'].fillna(0))
        df['Debit'] = pd.to_numeric(df['Debit'].fillna(0))

        # Calculate Net Amount (Credit - Debit) for easier analysis
        df['Net Amount'] = df['Credit'] - df['Debit']

        add_custom_css()

        # Sidebar Filters
        st.sidebar.header("Filter Options")
        start_date = st.sidebar.date_input("Start Date", value=df['Date'].min())
        end_date = st.sidebar.date_input("End Date", value=df['Date'].max())
        category_filter = st.sidebar.text_input("Search by Description (keyword)", "").upper()
        credit_debit_filter = st.sidebar.radio("Select Credit/Debit View", ["All", "Credits Only", "Debits Only"])

        # Apply Filters
        filtered_df = df[(df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))]
        if category_filter:
            filtered_df = filtered_df[filtered_df['Description'].str.contains(category_filter, na=False)]
        if credit_debit_filter == "Credits Only":
            filtered_df = filtered_df[filtered_df['Credit'] > 0]
        elif credit_debit_filter == "Debits Only":
            filtered_df = filtered_df[filtered_df['Debit'] > 0]

        # Main Dashboard
        st.markdown("<h1 class='stTitle'>Bank Statement Analysis Dashboard</h1>", unsafe_allow_html=True)

        # Display Filtered Data
        st.markdown("<h2 class='stSubheader'>Filtered Transactions</h2>", unsafe_allow_html=True)
        st.dataframe(filtered_df)

        # KPIs
        st.markdown("<h2 class='stSubheader'>Key Performance Indicators</h2>", unsafe_allow_html=True)
        total_credits = filtered_df['Credit'].sum()
        total_debits = filtered_df['Debit'].sum()
        net_balance = total_credits - total_debits

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Credits", f"₹{total_credits:,.2f}")
        col2.metric("Total Debits", f"₹{total_debits:,.2f}")
        col3.metric("Net Balance", f"₹{net_balance:,.2f}")

        # Pie Chart: Credit vs Debit
        st.markdown("<h2 class='stSubheader'>Credit vs Debit Distribution</h2>", unsafe_allow_html=True)
        credit_debit_pie = px.pie(
            names=["Credits", "Debits"],
            values=[total_credits, total_debits],
            title="Credit vs Debit Split",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(credit_debit_pie, use_container_width=True)

        # Category-Wise Spending Pie Chart
        st.markdown("<h2 class='stSubheader'>Category-Wise Spending (Debits)</h2>", unsafe_allow_html=True)
        category_spending = (
            filtered_df.groupby('Category')['Debit']
            .sum()
            .reset_index()
            .sort_values(by="Debit", ascending=False)
        )
        category_pie = px.pie(
            category_spending,
            names="Category",
            values="Debit",
            title="Category-Wise Spending",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(category_pie, use_container_width=True)

        # Bar Chart: Credit and Debit Over Time
        st.markdown("<h2 class='stSubheader'>Credits and Debits Over Time</h2>", unsafe_allow_html=True)
        credit_debit_bar = px.bar(
            filtered_df,
            x="Date",
            y=["Credit", "Debit"],
            title="Credits and Debits Over Time",
            labels={"value": "Amount", "variable": "Transaction Type"},
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(credit_debit_bar, use_container_width=True)

        # Time-Series Chart: Balance Trend
        st.markdown("<h2 class='stSubheader'>Balance Trend Over Time</h2>", unsafe_allow_html=True)
        balance_trend = px.line(
            filtered_df,
            x="Date",
            y="Balance",
            title="Balance Trend Over Time",
            markers=True,
            line_shape="spline",
            color_discrete_sequence=["#636EFA"]
        )
        st.plotly_chart(balance_trend, use_container_width=True)

        # Download Filtered Data
        st.markdown("<h2 class='stSubheader'>Download Filtered Data</h2>", unsafe_allow_html=True)
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Transactions as CSV",
            data=csv,
            file_name="filtered_transactions.csv",
            mime="text/csv"
        )

        # Footer
        st.sidebar.markdown("---")
        st.sidebar.write("Developed by Satyam Nautiyal")
        st.sidebar.write("For queries, contact: satyamnautiyal74@gmail.com")

    #else:
        #st.title("Bank Statement Analysis Dashboard")
        #st.write("Upload Bank Statement to visualize data.")


def main():

    data = pd.read_csv(r"C:\SatyamsFolder\projects\ML\Infosys-Springboard\BFSI-OCRv1\unsupervised\bank statements\transaction_labelled.csv")
    st.title("Bank Statement Analyzer")
    uploaded_pdf = st.file_uploader("Upload your Bank Statement as PDF", type=["pdf"])

    uploaded_csv = st.file_uploader("Upload your Bank Statement as CSV", type=["csv"])



    if uploaded_pdf :
        st.success("PDF uploaded successfully!")
        #df = extract_transactions_from_pdf(uploaded_pdf)
        #df_processed1 = process_csv1(df)
        #df_processed2 = process_csv2(df)
        #data = merge_df(df_processed1, df_processed2)
        #data = labeller(data)
        #visualize(data)

    elif uploaded_csv:
        uploaded_csv=pd.read_csv(uploaded_csv)
        df_processed1 = process_csv1(uploaded_csv)
        df_processed2 = process_csv2(uploaded_csv)
        data = merge_df(df_processed1, df_processed2)
        data = labeller(data)
        visualize(data)

if __name__ == "__main__":
    main()
