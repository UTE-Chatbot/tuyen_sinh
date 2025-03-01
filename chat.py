# gemini-rag-chatbot.py
# - Generative AI Gemini Chatbot
# - RAG Chatbot using ChromaDB
# @robertluwang
# Aug 2024

import textwrap
import chromadb
import numpy as np
import pandas as pd
import datetime

import google.generativeai as genai

from chromadb import Documents, EmbeddingFunction, Embeddings

import os

from dotenv import load_dotenv

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        model = 'models/embedding-001'
        title = "Custom query"
        return genai.embed_content(model=model,
                                   content=input,
                                   task_type="retrieval_document",
                                   title=title)["embedding"]
class RAGChatBot:
    def __init__(self):
        self.envpath = '~'
        self.envfile = '.env'

        if self.envpath == '~':
            self.envpath = os.path.expanduser("~")

        load_dotenv(os.path.join(self.envpath, self.envfile))

        genai.configure(api_key="AIzaSyBNMDBIw8EgbJdVSR8_io747BJn-JssUiU")

        # Create persistent client with local storage
        persist_directory = "./chroma_db"
        os.makedirs(persist_directory, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)

        self.dbname = 'geminidb'
        self.db = None
        self.documents = []
        self.subject = ''
        self.embedding_function = GeminiEmbeddingFunction()
        self.query = ''
        self.passage = ''
        self.model_name = 'gemini-1.5-flash'
        self.model = genai.GenerativeModel(self.model_name)
        self.chat_history = []

    def create_chroma_db(self):
        # Get list of collection names
        collection_names = self.chroma_client.list_collections()
        
        # Check if collection exists and delete if it does
        if self.dbname in collection_names:
            self.chroma_client.delete_collection(name=self.dbname)
            
        # Create new collection
        self.db = self.chroma_client.create_collection(
            name=self.dbname, 
            embedding_function=self.embedding_function
        )

        # Add documents to collection
        for i, d in enumerate(self.documents):
            self.db.add(
              documents=d,
              ids=str(i)
            )
        return self.db

    def get_relevant_passage(self):
        # Get top 3 most relevant passages instead of just 1
        results = self.db.query(query_texts=[self.query], n_results=3)
        self.passage = "\n".join(results['documents'][0])
        return self.passage

    def make_prompt(self):
        escaped = self.passage.replace("'", "").replace('"', "").replace("\n", " ")
        prompt = """Bạn là trợ lý tư vấn tuyển sinh của trường Đại học Sư phạm Kỹ thuật TP.HCM (HCMUTE).
Hãy trả lời câu hỏi của sinh viên một cách thân thiện, rõ ràng và chính xác dựa trên thông tin dưới đây.
Chỉ sử dụng thông tin được cung cấp, không thêm thông tin từ bên ngoài.
Nếu không có đủ thông tin để trả lời, hãy thành thật nói rằng bạn không có thông tin về vấn đề đó.

Thông tin tham khảo:
{passage}

Câu hỏi: {query}

Trả lời:""".format(query=self.query, passage=escaped)

        return prompt

    def log_chat_history(self,logpath):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"chat-log-{timestamp}.txt"
        log_path = os.path.join(logpath, log_filename)

        os.makedirs(logpath, exist_ok=True)

        with open(log_path, "w") as f:
            for message in self.chat_history:
                f.write(f"{message}\n")

        print(f"chat log file: {log_path}")

    def ragchat(self):
        self.db = self.create_chroma_db()

        n=1 # input number
        print("Welcome to Gemini RAG Chatbot ! ('/q' to exit)")
        self.chat_history.append(f"Welcome to Gemini RAG Chatbot ! ('/q' to exit)")
        print(f"Based on doc set of subject: '{self.subject}'\n")
        self.chat_history.append(f"Based on doc set of subject: '{self.subject}'\n")
        while True:
            self.query = input(f"{n} You: ")

            if not self.query:  # Check if input is empty (only Enter pressed)
                continue  # Skip processing empty input

            self.chat_history.append(f"{n} You: {self.query}\n")

            if self.query.lower() == "/q":
                self.log_chat_history('./log')
                print("Chat history saved. Exiting.")
                break

            self.passage = self.get_relevant_passage()
            
            # Print retrieved relevant passages
            print("\nRelevant documents:")
            print("-" * 80) 
            print(self.passage)
            print("-" * 80)
            print()

            prompt = self.make_prompt()
            #print(f"prompt: {prompt}\n")  # Optional: comment out prompt printing

            response = self.model.generate_content(prompt)
            print(f"{n} RAG Chatbot: {response.text}")
            self.chat_history.append(f"{n} RAG Chatbot: {response.text}")
            n += 1

if __name__ == "__main__":
    ragchatbot = RAGChatBot()

    #ragchatbot.model_name = "gemini-1.5-flash"
    #ragchatbot.dbname = "geminidb"
    # DOCUMENT1 = "Gemini is the result of large-scale collaborative efforts by teams across Google, including our colleagues at Google Research. It was built from the ground up to be multimodal, which means it can generalize and seamlessly understand, operate across and combine different types of information including text, code, audio, image and video."
    # DOCUMENT2 = "We designed Gemini to be natively multimodal, pre-trained from the start on different modalities. Then we fine-tuned it with additional multimodal data to further refine its effectiveness. This helps Gemini seamlessly understand and reason about all kinds of inputs from the ground up, far better than existing multimodal models — and its capabilities are state of the art in nearly every domain."
    # DOCUMENT3 = "Gemini has the most comprehensive safety evaluations of any Google AI model to date, including for bias and toxicity. We’ve conducted novel research into potential risk areas like cyber-offense, persuasion and autonomy, and have applied Google Research’s best-in-class adversarial testing techniques to help identify critical safety issues in advance of Gemini’s deployment."
    # Load from demo4.csv
    df = pd.read_csv('demo4.csv')
    ragchatbot.documents = df['text'].tolist()
    ragchatbot.subject = 'Chatbot tư vấn tuyển sinh'

    ragchatbot.ragchat()