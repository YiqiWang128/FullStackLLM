"""
conda activate llm_fullstack_project
python chat_with_rag.py
"""

import os
import faiss
import numpy as np
import mysql.connector
from datetime import datetime
from openai import OpenAI
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels, DashScopeTextEmbeddingType

class VectorQueryService:
    def __init__(self, index_file_path, mysql_config):
        self.index_file_path = index_file_path
        self.mysql_config = mysql_config
        self.faiss_index = self.load_faiss_index()
        self.embedder = self.initialize_embedder()

    def load_faiss_index(self):
        return faiss.read_index(self.index_file_path)

    def initialize_embedder(self):
        return DashScopeEmbedding(
            model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
            text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
        )
    
    def connect_to_database(self):
        return mysql.connector.connect(**self.mysql_config)
    
    def query_knowledge_base(self, user_query, top_k=5):
        query_embedding = self.embedder.get_text_embedding_batch([user_query])[0]
        if query_embedding is None:
            raise ValueError("Query embedding failed.")
        query_embedding_np = np.array(query_embedding, dtype="float32").reshape(1, -1)
        distances, indices = self.faiss_index.search(query_embedding_np, top_k)
        results = self.fetch_results_from_database(indices[0])
        context = self.build_context(results)
        return context
    
    def fetch_results_from_database(self, indices):
        conn = self.connect_to_database()
        cursor = conn.cursor()
        results = []
        for idx in indices:
            if idx == -1:
                continue
            cursor.execute("SELECT text FROM SneakerStore_VectorMapping WHERE id = %s", (int(idx) + 1,))
            result = cursor.fetchone()
            if result:
                results.append(result[0])
        cursor.close()
        conn.close()
        return results
    
    def build_context(self, results):
        context = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(results))
        return f"以下是知识库中与您的问题相关的内容：\n{context}"

def main():

    INDEX_FILE_PATH = "SneakerStore_VectorMapping.index"
    MYSQL_CONFIG = {
        "host": "localhost",
        "user": "root",
        "password": "123456",
        "database": "FullStackLLM_DB",
    }
    API_KEY = os.getenv("DASHSCOPE_API_KEY")
    vector_service = VectorQueryService(INDEX_FILE_PATH, MYSQL_CONFIG)
    client = OpenAI(
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    msgs = [{"role": "system", "content": "You are a helpful assistant."}]
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit", "再见"]:
            print("感谢您的咨询，再见！")
            break
        try:
            context = vector_service.query_knowledge_base(user_input)
            msgs.append({"role": "user", "content": f"{user_input}\n{context}"})
        except Exception as e:
            print(f"知识库查询失败：{e}")
            continue
        try:
            completion = client.chat.completions.create(
                model="qwen-plus",
                messages=msgs,
                stream=True,
            )
            msg = ""
            for chunk in completion:
                print(chunk.choices[0].delta.content, end="", flush=True)
                msg += chunk.choices[0].delta.content
            msgs.append({"role": "assistant", "content": msg})
        except Exception as e:
            print(f"LLM 调用失败：{e}")

if __name__ == "__main__":
    main()