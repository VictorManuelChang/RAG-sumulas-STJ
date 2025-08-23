import os
from llama_index.core import (
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


PERSIST_DIR = "./storage"
EMBED_MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

def run_query(query, top_k=3):
  
    print("---------------------------------------------------------")
    print(f"Executando consulta: '{query}'")
    
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    
    query_engine = index.as_query_engine(similarity_top_k=top_k)
    
    response = query_engine.query(query)
    
    print("\n--- Súmulas Mais Relevantes Encontradas ---")
    if not response.source_nodes:
        print("Nenhum resultado encontrado.")
    else:
        for node in response.source_nodes:
            print(f"ID: {node.metadata['id']} | Similaridade: {node.score:.4f}")
            print(f"Texto: {node.get_content().strip()}\n")
    print("---------------------------------------------------------")


if __name__ == "__main__":
    if not os.path.exists(PERSIST_DIR):
        print(f"ERRO: Pasta do índice '{PERSIST_DIR}' não encontrada.")
        print("Execute o script 'build.py' primeiro para criar o índice.")
    else:
        Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
        Settings.llm = None
        
        queries_para_testar = [
            "impossibilidade de conversão de ofício da prisão em flagrante em preventiva",
            "interpretação de cláusula contratual não serve para recurso especial",
            "foro competente para ação de investigação de paternidade e alimentos",
            "compete à justiça estadual julgar processo eleitoral sindical",
        ]
        
        for query in queries_para_testar:
            run_query(query)