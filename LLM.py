import fitz  # PyMuPDF
import pytesseract
import pdfplumber
import re
import sqlite3
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
import torch

# Database connection
def create_connection(db_file):
    conn = sqlite3.connect(db_file)
    return conn

def create_table(conn):
    sql_create_files_table = """ CREATE TABLE IF NOT EXISTS files (
                                    id integer PRIMARY KEY,
                                    name text NOT NULL,
                                    data blob NOT NULL
                                ); """
    sql_create_results_table = """ CREATE TABLE IF NOT EXISTS results (
                                    id integer PRIMARY KEY,
                                    file_id integer NOT NULL,
                                    result blob NOT NULL,
                                    FOREIGN KEY (file_id) REFERENCES files (id)
                                ); """
    c = conn.cursor()
    c.execute(sql_create_files_table)
    c.execute(sql_create_results_table)

def insert_file(conn, file_path):
    with open(file_path, 'rb') as file:
        file_data = file.read()
    sql = ''' INSERT INTO files(name, data)
              VALUES(?, ?) '''
    cur = conn.cursor()
    cur.execute(sql, (os.path.basename(file_path), file_data))
    conn.commit()
    return cur.lastrowid

def insert_result(conn, file_id, result_data):
    sql = ''' INSERT INTO results(file_id, result)
              VALUES(?, ?) '''
    cur = conn.cursor()
    cur.execute(sql, (file_id, result_data))
    conn.commit()

# Read PDF using fitz and pytesseract
def read_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)  # first page
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

def ocr_image(image):
    return pytesseract.image_to_string(image, lang='eng')

# Transaction extraction with pdfplumber
def extract_transactions(pdf_path):
    data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            lines = text.split('\n')
            for line in lines:
                data.append(line.split())
    columns = ['Transaction Date', 'Value Date', 'Description', 'Debit', 'Credit', 'Balance']
    df = pd.DataFrame(data, columns=columns)
    df["Debit"] = pd.to_numeric(df["Debit"].apply(lambda x: str(x).replace(",", "") if x else '0'), errors='coerce').fillna(0)
    df["Credit"] = pd.to_numeric(df["Credit"].apply(lambda x: str(x).replace(",", "") if x else '0'), errors='coerce').fillna(0)
    return df

# Categorizing transaction descriptions
def categorize_transactions(df, model, tokenizer):
    df["Category"] = df["Description"].apply(lambda x: classify_transaction(x, model, tokenizer))
    return df

def classify_transaction(description, model, tokenizer):
    inputs = tokenizer(description, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions.item()

# Visualization function
def visualize_transactions(df):
    category_counts = df['Category'].value_counts()
    plt.figure(figsize=(10, 6))
    wedges, texts, autotexts = plt.pie(
        category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    for i, text in enumerate(texts):
        x, y = text.get_position()
        connection = ConnectionPatch((x, y), (1.2 * x, 1.2 * y), "data", "data", arrowstyle="->", color='black')
        plt.gca().add_artist(connection)
        text.set_position((1.2 * x, 1.2 * y))
        text.set_fontsize(10)
        text.set_bbox(dict(facecolor='white', edgecolor='none', pad=1))
    plt.show()

# Train the model
def train_model(df):
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)  # 5 categories

    train_texts, val_texts, train_labels, val_labels = train_test_split(df['Description'], df['Category'], test_size=0.2)
    train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
    val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True)

    class TransactionDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = TransactionDataset(train_encodings, train_labels.tolist())
    val_dataset = TransactionDataset(val_encodings, val_labels.tolist())

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()

    model.save_pretrained('./transaction_model')
    tokenizer.save_pretrained('./transaction_model')

    return model, tokenizer

# Example of usage
pdf_path = r"C:\SatyamsFolder\projects\ML\Infosys-Springboard\BFSI-OCRv1\unsupervised\bank statements\AU.pdf"
img = read_pdf(pdf_path)
text = ocr_image(img)

df = extract_transactions(pdf_path)

# Optionally, train a new model on your labeled dataset
model, tokenizer = train_model(df)

# Classify transactions with the trained model
df = categorize_transactions(df, model, tokenizer)

# Visualize the categorized transactions
visualize_transactions(df)
