import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import pandas as pd

# Define the dictionary with labels
category_dict = {
    "MADHURI":"UPI", "VIJAY":"UPI", "SHRIYAM":"UPI", "AJAY":"UPI", "JEENA":"UPI", "SHILA":"UPI", "PRAKASH":"UPI",
    "MUTINDIAN":"Investment", "CLEARING":"Investment",
    "DILIP SINGH":"Grocery", "P H AND SONS":"Grocery", "KAMAL":"Grocery",
    "SUBWAY":"Food", "VINAYAK AGENCIES":"Food", "BEVERAGE FOR FRIENDS":"Food", "MAHESH KUMAR MEENA":"Food", "BAKERS":"Food", "DIALOG":"Food", "JAGDISH":"Food",
             "ZOMATO":"Food", "SWIGGY":"Food", "MOHANJI":"Food",
    "HOTEL HIGHWAY KING":"Travel", "HOTEL":"Travel", "UBER":"Travel", "OLA":"Travel",
    "SMS":"Charges", "CHARGE":"Charges",
    "MANIPAL":"Education", "GOOD HOST":"Education",
    "NYKAA":"Health", "SUNITA":"Health", "KOTHARI":"Health",
    "SPOTIFY":"Subscription",
    "REWARDS":"Rewards"
}

# Create a mapping of transaction descriptions to categories
# For the sake of this example, we'll mock some transaction data
data = []
for category, items in category_dict.items():
    for item in items:
        data.append({"text": item, "label": category})

# Split data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2)

# Convert the data into a Dataset object
train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
test_dataset = Dataset.from_pandas(pd.DataFrame(test_data))

# Combine the datasets into a DatasetDict
dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Convert labels to integers
label_map = {category: idx for idx, category in enumerate(category_dict.keys())}
def map_labels(example):
    example['label'] = label_map[example['label']]
    return example

tokenized_datasets = tokenized_datasets.map(map_labels)

# Step 3: Initialize the model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label_map))

# Step 4: Define the Trainer and TrainingArguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    evaluation_strategy="epoch",     # evaluation strategy to use
    learning_rate=2e-5,              # learning rate
    per_device_train_batch_size=16,  # batch size for training
    per_device_eval_batch_size=64,   # batch size for evaluation
    num_train_epochs=5,              # number of training epochs
    weight_decay=0.01,                # strength of weight decay
    #warmup_steps=500
)

trainer = Trainer(
    model=model,                     # the model to train
    args=training_args,              # training arguments
    train_dataset=tokenized_datasets['train'],         # training dataset
    eval_dataset=tokenized_datasets['test'],           # evaluation dataset
    tokenizer=tokenizer,             # tokenizer to be used for encoding
)

# Step 5: Train the model
trainer.train()

# Step 6: Save the model
model.save_pretrained("./finetuned_distilbert_model")
tokenizer.save_pretrained("./finetuned_distilbert_model")