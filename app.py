from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import pickle
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



class Chatbot:
    def __init__(self, model_path, tokenizer_path, encoded_data_path):
        self.model, self.tokenizer = self.load_model_and_tokenizer(model_path, tokenizer_path)
        encoded_data = self.load_encoded_data(encoded_data_path)
        self.encoded_questions = encoded_data['encoded_questions']
        self.qa_data = encoded_data['qa_data']
    
    def load_model_and_tokenizer(self, model_path, tokenizer_path):
        model = BertModel.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        return model, tokenizer

    def load_encoded_data(self, filepath):
        with open(filepath, 'rb') as file:
            encoded_data = pickle.load(file)
        return encoded_data

    def encode_questions(self, questions):
        encoded_questions = []
        for question in questions:
            inputs = self.tokenizer(question, return_tensors='pt', padding=True, truncation=True)
            outputs = self.model(**inputs)
            encoded_questions.append(outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy())
        return encoded_questions

    def find_top_similar_questions(self, user_input, top_k=3):
        user_input_encoded = self.encode_questions([user_input])[0].reshape(1, -1)
        similarities = cosine_similarity(user_input_encoded, np.array(self.encoded_questions))
        top_k_indices = np.argsort(similarities[0])[-top_k:][::-1]
        top_k_questions = self.qa_data['QA-問題'].iloc[top_k_indices].tolist()
        top_k_confidences = similarities[0, top_k_indices]
        return top_k_questions, top_k_confidences

    def get_answer(self, question):
        return self.qa_data[self.qa_data['QA-問題'] == question]['QA-答案'].values[0]

    def get_response(self, user_input, top_k=3):
        top_k_questions, top_k_confidences = self.find_top_similar_questions(user_input, top_k)
        top_k_answers = [self.get_answer(q) for q in top_k_questions]
        
        # Convert confidences to Python float
        top_k_confidences = [float(conf) for conf in top_k_confidences]

        return list(zip(top_k_questions, top_k_answers, top_k_confidences))
# Initialize Flask app
app = Flask(__name__, template_folder='.')
# app = Flask(__name__)
CORS(app)
# Initialize chatbot
chatbot = Chatbot('bert_model', 'bert_tokenizer', 'encoded_questions.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chatbot', methods=['POST'])
def chatbot_response():
    data = request.get_json()
    user_input = data.get("question", "")
    if user_input:
        top_responses = chatbot.get_response(user_input, top_k=3)
        response_data = []
        for i, (question, answer, confidence) in enumerate(top_responses, start=1):
            response_data.append({
                "question": question,
                "answer": answer,
                "confidence": confidence
            })
        return jsonify(response_data)
    return jsonify([])  



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
