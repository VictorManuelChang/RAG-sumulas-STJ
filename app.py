import streamlit as st
import os
from dotenv import load_dotenv

from llama_index.core import (
    StorageContext,
    Settings,
    load_index_from_storage,
    PromptTemplate
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()
PERSIST_DIR = "./storage"
EMBED_MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

st.set_page_config(
    page_title="Análise de Ementas vs Súmulas do STJ",
    page_icon="⚖️",
    layout="centered"
)

@st.cache_resource
def load_rag_system():
    if not os.path.exists(PERSIST_DIR):
        st.error(f"Diretório do índice '{PERSIST_DIR}' não encontrado. Por favor, execute 'build.py' primeiro.")
        return None
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    return index


st.title("⚖️ Analisador de Ementas vs Súmulas do STJ")
st.markdown("""
    Esta aplicação utiliza um sistema RAG aprimorado para verificar se uma ementa jurídica
    está em consonância com alguma das Súmulas do STJ.
""")

index = load_rag_system()

if index:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("A chave da API da OpenAI (OPENAI_API_KEY) não foi encontrada. Por favor, configure seu arquivo .env.")
    else:
        user_query = st.text_area(
            "Cole o texto da ementa aqui:",
            height=250,
            placeholder="Pergunte sobre a ementa ou cole o texto completo aqui..."
        )

        if st.button("Analisar Ementa"):
            if not user_query.strip():
                st.warning("Por favor, insira o texto da ementa antes de analisar.")
            else:
                with st.spinner("Analisando... O assistente está processando a consulta e buscando as súmulas..."):
                    try:
                        llm = OpenAI(
                            model="gpt-3.5-turbo",
                            temperature=0,
                            api_key=os.getenv("OPENAI_API_KEY")
                        )

                        extracao_prompt_str = (
                            "Sua tarefa é identificar e extrair APENAS o texto que constitui uma ementa jurídica do texto fornecido. "
                            "Uma ementa jurídica é o resumo oficial de uma decisão judicial, geralmente em letras maiúsculas, "
                            "que contém palavras-chave como: PRISÃO, RECURSO, APELAÇÃO, HABEAS CORPUS, etc. "
                            "Ignore perguntas, instruções, saudações ou qualquer texto explicativo. "
                            "Se houver múltiplos blocos, extraia apenas o que parece ser a ementa oficial.\n\n"
                            "Texto do usuário: {full_query}\n\n"
                            "Ementa jurídica extraída (apenas o texto oficial):"
                        )
                        extracao_prompt = PromptTemplate(extracao_prompt_str)
                        
                        prompt_formatado = extracao_prompt.format(full_query=user_query)
                        response = llm.complete(prompt_formatado)
                        ementa_extraida = response.text.strip()
                    
                        query_generation_prompts = [
                            (
                                "Extraia os 3-4 termos jurídicos mais importantes desta ementa para busca: {ementa_text}\n"
                                "Responda APENAS com os termos separados por espaço:"
                            ),
                            (
                                "Qual é a tese jurídica principal desta ementa em uma frase de até 15 palavras? {ementa_text}\n"
                                "Tese:"
                            ),
                            (
                                "Qual o principal instituto jurídico tratado nesta ementa? {ementa_text}\n"
                                "Instituto:"
                            )
                        ]
                        
                        queries_otimizadas = []
                        for prompt_template in query_generation_prompts:
                            formatted_query_prompt = prompt_template.format(ementa_text=ementa_extraida)
                            response = llm.complete(formatted_query_prompt)
                            queries_otimizadas.append(response.text.strip())

                        all_retrieved_nodes = []
                        retriever = index.as_retriever(similarity_top_k=8) 
                        
                        for query in queries_otimizadas:
                            retrieved_nodes = retriever.retrieve(query)
                            for node_com_score in retrieved_nodes:
                                node_id = node_com_score.node.metadata.get('id', node_com_score.node.node_id)
                                if not any(
                                    existing.node.metadata.get('id', existing.node.node_id) == node_id
                                    for existing in all_retrieved_nodes
                                ):
                                    all_retrieved_nodes.append(node_com_score)
                        
                        all_retrieved_nodes.sort(key=lambda x: x.score, reverse=True)
                        final_retrieved_nodes = all_retrieved_nodes[:10]

                        with st.expander("🔍 Informações de Depuração (Processo de Otimização)"):
                            st.text_area("1. Ementa Extraída da Consulta:", value=ementa_extraida, height=150, disabled=True)
                            
                            st.write("2. Consultas Geradas:")
                            for i, query in enumerate(queries_otimizadas):
                                st.text_input(f"Consulta {i+1}:", value=query, disabled=True, key=f"query_{i}")
                            
                            st.info(f"Súmulas encontradas ({len(final_retrieved_nodes)} resultados):")
                            for node_com_score in final_retrieved_nodes:
                                node_obj = node_com_score.node 
                                sumula_id = node_obj.metadata.get('id', node_obj.node_id)
                                st.write(f"**Súmula {sumula_id}** | **Similaridade:** {node_com_score.score:.4f}")
                                st.text_area(
                                    label=f"Conteúdo da Súmula {sumula_id}",
                                    value=node_obj.get_content(), 
                                    height=80, 
                                    disabled=True, 
                                    key=f"content_{node_obj.node_id}",
                                    label_visibility="collapsed"
                                )
                        
                        Settings.llm = OpenAI(
                            model="gpt-4-turbo",
                            api_key=os.getenv("OPENAI_API_KEY")
                        )
                        qa_prompt_tmpl_str = (
                            "Você é um especialista em Súmulas do STJ e deve fazer uma análise precisa.\n"
                            "TAREFA: Analisar se a ementa apresentada está em consonância com alguma das Súmulas do STJ fornecidas.\n\n"
                            "SÚMULAS RELEVANTES ENCONTRADAS:\n"
                            "---------------------\n"
                            "{context_str}\n"
                            "---------------------\n\n"
                            "EMENTA PARA ANÁLISE:\n"
                            "---------------------\n"
                            "{query_str}\n"
                            "---------------------\n\n"
                            "INSTRUÇÕES PARA RESPOSTA:\n"
                            "1. Leia cuidadosamente cada Súmula fornecida\n"
                            "2. Identifique se alguma trata do mesmo tema da ementa\n"
                            "3. Se encontrar consonância:\n"
                            "   - Cite o número da Súmula\n"
                            "   - Transcreva o texto completo da Súmula\n"
                            "   - Explique detalhadamente a consonância\n"
                            "4. Se não encontrar:\n"
                            "   - Afirme claramente que não há consonância\n"
                            "   - Mencione brevemente os temas das Súmulas analisadas\n\n"
                            "ATENÇÃO: Priorize Súmulas com números altos (como 676, 675, etc.) que são mais recentes.\n"
                        )
                        qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
                        
                        response_synthesizer = get_response_synthesizer(
                            response_mode="compact",
                            text_qa_template=qa_prompt_tmpl,
                        )
                        
                        class CustomRetriever:
                            def __init__(self, nodes):
                                self.nodes = nodes
                            
                            def retrieve(self, query):
                                return self.nodes
                        
                        custom_retriever = CustomRetriever(final_retrieved_nodes)
                        
                        query_engine = RetrieverQueryEngine(
                            retriever=custom_retriever,
                            response_synthesizer=response_synthesizer,
                        )
                        
                        final_response = query_engine.query(user_query)

                        st.success("Análise Concluída!")
                        st.markdown("### Resposta do Assistente Jurídico:")
                        st.markdown(str(final_response))

                    except Exception as e:
                        st.error(f"Ocorreu um erro durante a análise: {e}")
