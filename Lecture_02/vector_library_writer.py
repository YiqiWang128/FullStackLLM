"""
conda activate llm_fullstack_project
python vector_library_writer.py
"""


import pandas as pd
import faiss
import numpy as np
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels, DashScopeTextEmbeddingType
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.schema import Node  # 更新后的导入路径


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

print(f"Successfully vectorized {len(valid_embeddings)} out of {len(text_data)} texts.")


embedding_dim = 1536
faiss_index = faiss.IndexFlatL2(embedding_dim)


valid_embeddings_np = np.array(valid_embeddings, dtype="float32")
faiss_index.add(valid_embeddings_np)


vector_store = FaissVectorStore(faiss_index=faiss_index)


nodes = [
    Node(text=metadata, embedding=embedding)
    for metadata, embedding in zip(valid_metadata, valid_embeddings)
]


vector_store.add(nodes)


index_file_path = "ai_context.index"
faiss.write_index(faiss_index, index_file_path)

print(f"FAISS index saved successfully at: {index_file_path}")
