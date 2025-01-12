Customer Question Answering Chatbot
This project is a fine-tuned BERT-based-Chinese chatbot designed to answer customer questions using a pre-existing knowledge base stored in an Excel file. The chatbot encodes customer queries and compares them with the most similar questions from the knowledge base using cosine similarity to provide relevant answers.

Project Structure:
>train.py: Script to fine-tune the bert-base-chinese pre-trained model , encode questions from the knowledge base, and save the trained model and data.
>QA_chatbot.py: Chatbot class implementation, which loads the trained BERT model and finds the top 3 similar questions to user input.
>app.py: Flask web application to serve the chatbot and handle API requests.
>index.html: Frontend interface to interact with the chatbot.
