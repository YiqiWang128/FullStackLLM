import faiss
import numpy as np
import mysql.connector
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels, DashScopeTextEmbeddingType

index_file_path = "SneakerStore_VectorMapping.index"
faiss_index = faiss.read_index(index_file_path)
embedder = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
)

def query_knowledge_base(user_query, top_k=5):
    query_embedding = embedder.get_text_embedding_batch([user_query])[0]
    query_embedding_np = np.array(query_embedding, dtype="float32").reshape(1, -1)
    distances, indices = faiss_index.search(query_embedding_np, top_k)
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="123456",
        database="FullStackLLM_DB"
    )
    cursor = conn.cursor()
    results = []
    
    for idx in indices[0]:
        if idx == -1:
            continue
        cursor.execute("SELECT text FROM SneakerStore_VectorMapping WHERE id = %s", (int(idx) + 1,))
        result = cursor.fetchone()
        if result:
            results.append(result[0])
    cursor.close()
    conn.close()
    return results
user_input = input("请输入您想查询的信息：")
search_results = query_knowledge_base(user_input)

print("最相关的文本：")
for i, text in enumerate(search_results):
    print(f"{i + 1}. {text}")