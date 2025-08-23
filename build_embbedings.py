import os
import json
import re
import time

from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

JSON_FILE_PATH = "sumulas.json"
PERSIST_DIR = "./storage"
EMBED_MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

def limpeza_texto_sumula(text):
    return re.sub(r'\s*\([^)]+\)$', '', text).strip()

def build_and_persist_index():
   
    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            sumulas_data = json.load(f)
    except FileNotFoundError:
        print(f"ERRO: Arquivo '{JSON_FILE_PATH}' não encontrado.")
        return

    documents = [
        Document(text=limpeza_texto_sumula(item['texto']), metadata={"id": item['id']})
        for item in sumulas_data
    ]

    print("\nConstruindo o índice vetorial")
    start = time.time()
    
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    
    end = time.time()
    print(f"Índice construído em {end - start:.2f} segundos.")

    print(f"Salvando o índice na pasta '{PERSIST_DIR}'...")
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    
    print("\nÍndice construído e salvo com sucesso!")



if __name__ == "__main__":
    print(f"Configurando o modelo de embedding global: '{EMBED_MODEL_NAME}'...")
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.llm = None 
    
    if not os.path.exists(PERSIST_DIR):
        build_and_persist_index()
    else:
        print(f"Índice já encontrado na pasta '{PERSIST_DIR}'.")
    
    