import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the data
data = pd.read_excel('qa_knowledge_base.xlsx')

# Extract the relevant columns
qa_data = data[['QA-問題', 'QA-答案']]
qa_data.dropna(subset=['QA-問題', 'QA-答案'], inplace=True)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')

# Function to encode questions
def encode_questions(questions):
    encoded_questions = []
    for question in questions:
        inputs = tokenizer(question, return_tensors='pt', padding=True, truncation=True)
        outputs = model(**inputs)
        encoded_questions.append(outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy())
    return encoded_questions

# Encode QA-問題
encoded_questions = encode_questions(qa_data['QA-問題'].tolist())

# Save BERT model
def save_model(model, tokenizer, model_path, tokenizer_path):
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(tokenizer_path)

# Save encoded questions and corresponding data
def save_encoded_data(encoded_data, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(encoded_data, file)

# Paths to save the model and data
model_path = 'bert_model'
tokenizer_path = 'bert_tokenizer'
encoded_data_path = 'encoded_questions.pkl'

# Save model and tokenizer
save_model(model, tokenizer, model_path, tokenizer_path)

# Save encoded questions and QA data
encoded_data = {
    'encoded_questions': encoded_questions,
    'qa_data': qa_data
}
save_encoded_data(encoded_data, encoded_data_path)
