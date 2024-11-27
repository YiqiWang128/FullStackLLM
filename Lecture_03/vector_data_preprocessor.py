import pandas as pd
import numpy as np
import faiss
import mysql.connector
from datetime import datetime
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels, DashScopeTextEmbeddingType

with open('运动鞋店铺知识库.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
data = pd.DataFrame({'text': [line.strip() for line in lines if line.strip()]})
embedder = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
)
text_data = data['text'].tolist()
result_embeddings = embedder.get_text_embedding_batch(text_data)
valid_embeddings = []
valid_metadata = []

for emb, text in zip(result_embeddings, text_data):
    if emb is not None:
        valid_embeddings.append(emb)
        valid_metadata.append(text)
embedding_dim = 1536
faiss_index = faiss.IndexFlatL2(embedding_dim)
valid_embeddings_np = np.array(valid_embeddings, dtype="float32")
faiss_index.add(valid_embeddings_np)
index_file_path = "SneakerStore_VectorMapping.index"
faiss.write_index(faiss_index, index_file_path)

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="123456",
    database="FullStackLLM_DB"
)
cursor = conn.cursor()
cursor.execute("DELETE FROM SneakerStore_VectorMapping")
conn.commit()
cursor.execute("ALTER TABLE SneakerStore_VectorMapping AUTO_INCREMENT = 1")
conn.commit()

for text, embedding in zip(valid_metadata, valid_embeddings):
    cursor.execute(, (text, str(embedding), datetime.now(), datetime.now()))
conn.commit()
cursor.close()
conn.close()

print(f"FAISS index saved successfully at: {index_file_path}")