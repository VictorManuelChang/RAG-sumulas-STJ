import streamlit as st
import os
import json  
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
HISTORY_FILE = "history.json" 

st.set_page_config(
    page_title="Análise de Ementas vs Súmulas do STJ",
    page_icon="⚖️",
    layout="wide"
)

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return [] 
    return []

def save_history():
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(st.session_state.history, f, indent=4)

if "page" not in st.session_state:
    st.session_state.page = "home"

if "history" not in st.session_state:
    st.session_state.history = load_history()


@st.cache_resource(show_spinner="Carregando base de conhecimento das Súmulas...")
def load_rag_system():
    if not os.path.exists(PERSIST_DIR):
        return None
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    return index


def show_home():
    st.title("⚖️ Analisador de Ementas vs Súmulas do STJ")
    st.markdown("""
        ### Bem-vindo!
        Esta aplicação utiliza um sistema de Inteligência Artificial (RAG) para verificar se uma ementa jurídica
        está em consonância com alguma das **Súmulas do Superior Tribunal de Justiça (STJ)**.
    """)

    st.markdown("---")

    _, col2, _ = st.columns([2, 3, 2])
    with col2:
        if st.button("🚀 Iniciar Análise", use_container_width=True, type="primary"):
            st.session_state.page = "analysis"
            st.rerun()

def show_analysis():
    st.title("⚖️ Assistente Jurídico STJ")

    index = load_rag_system()

    if not index:
        st.error(f"Diretório do índice '{PERSIST_DIR}' não encontrado. Por favor, execute o script 'build.py' primeiro para criar a base de dados.")
        st.stop()

    if not os.getenv("OPENAI_API_KEY"):
        st.error("A chave da API da OpenAI (OPENAI_API_KEY) não foi encontrada. Por favor, configure seu arquivo .env.")
        st.stop()

    with st.sidebar:
        st.header("ℹ️ Instruções")
        st.markdown(
            """
            1. **Cole a ementa** na área de texto principal.
            2. Clique em **Analisar Ementa**.
            3. A IA irá extrair a ementa, buscar súmulas relevantes e gerar uma análise completa.
            """
        )

        st.header("📂 Histórico de Consultas")
        if st.button("🗑️ Limpar Histórico"):
            st.session_state.history = []
            save_history()
            st.rerun()

        if st.session_state.history:
            for i, item in enumerate(reversed(st.session_state.history)):
                with st.expander(f"Consulta {len(st.session_state.history) - i}"):
                    st.text_area("Ementa:", value=item["ementa"], height=100, disabled=True, key=f"hist_q_{i}")
                    st.markdown("**Resposta da IA:**")
                    st.markdown(item["resposta"], unsafe_allow_html=True)
        else:
            st.info("Nenhuma consulta foi realizada ainda.")

    user_query = st.text_area(
        "### 📄 Cole o texto da ementa jurídica aqui:",
        height=250,
        placeholder="AGRAVO REGIMENTAL NO HABEAS CORPUS. TRÁFICO DE DROGAS. DOSIMETRIA. PENA-BASE..."
    )

    if st.button("🔍 Analisar Ementa", use_container_width=True, type="primary"):
        if not user_query.strip():
            st.warning("⚠️ Por favor, insira o texto da ementa antes de analisar.")
            return

        with st.spinner("🤖 Analisando... A IA está processando a ementa e buscando as súmulas..."):
            try:
                llm_extrator = OpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0,
                    api_key=os.getenv("OPENAI_API_KEY")
                )
                extracao_prompt_str = (
                    "Sua tarefa é identificar e extrair APENAS o texto que constitui uma ementa jurídica do texto fornecido. "
                    "Ignore perguntas, instruções, saudações ou qualquer texto explicativo. "
                    "Se houver múltiplos blocos, extraia apenas o que parece ser a ementa oficial.\n\n"
                    "Texto do usuário: {full_query}\n\n"
                    "Ementa jurídica extraída (apenas o texto oficial):"
                )
                extracao_prompt = PromptTemplate(extracao_prompt_str)
                response = llm_extrator.complete(extracao_prompt.format(full_query=user_query))
                ementa_extraida = response.text.strip()

                query_generation_prompts = [
                    "Extraia os 3-4 termos jurídicos mais importantes desta ementa: {ementa_text}",
                    "Qual é a tese jurídica principal desta ementa em uma frase de até 15 palavras? {ementa_text}",
                    "Qual o principal instituto jurídico tratado nesta ementa? {ementa_text}"
                ]
                queries_otimizadas = [llm_extrator.complete(p.format(ementa_text=ementa_extraida)).text.strip() for p in query_generation_prompts]

                retriever = index.as_retriever(similarity_top_k=8)
                all_retrieved_nodes = []
                retrieved_ids = set()

                for query in queries_otimizadas:
                    retrieved_nodes = retriever.retrieve(query)
                    for node_with_score in retrieved_nodes:
                        node_id = node_with_score.node.metadata.get('id', node_with_score.node.node_id)
                        if node_id not in retrieved_ids:
                            all_retrieved_nodes.append(node_with_score)
                            retrieved_ids.add(node_id)

                all_retrieved_nodes.sort(key=lambda x: x.score, reverse=True)
                final_retrieved_nodes = all_retrieved_nodes[:10]

                Settings.llm = OpenAI(model="gpt-4-turbo", api_key=os.getenv("OPENAI_API_KEY"))

                qa_prompt_tmpl_str = (
                    "Você é um especialista em Súmulas do STJ e deve fazer uma análise precisa.\n"
                    "TAREFA: Analisar se a ementa apresentada está em consonância com alguma das Súmulas do STJ fornecidas.\n\n"
                    "SÚMULAS RELEVANTES ENCONTRADAS:\n---------------------\n{context_str}\n---------------------\n\n"
                    "EMENTA PARA ANÁLISE:\n---------------------\n{query_str}\n---------------------\n\n"
                    "INSTRUÇÕES PARA RESPOSTA:\n"
                    "1. Se encontrar consonância: Cite o número da Súmula, transcreva seu texto completo e explique detalhadamente a relação.\n"
                    "2. Se não encontrar: Afirme claramente que não há consonância direta e, se possível, mencione brevemente os temas das Súmulas encontradas para mostrar que foram analisadas.\n"
                    "3. Seja claro, objetivo e use formatação markdown para organizar a resposta."
                )
                qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

                response_synthesizer = get_response_synthesizer(
                    response_mode="compact",
                    text_qa_template=qa_prompt_tmpl,
                )

                class CustomRetriever:
                    def __init__(self, nodes): self.nodes = nodes
                    def retrieve(self, query): return self.nodes

                query_engine = RetrieverQueryEngine(
                    retriever=CustomRetriever(final_retrieved_nodes),
                    response_synthesizer=response_synthesizer,
                )

                final_response = query_engine.query(ementa_extraida)

                st.success("✅ Análise Concluída!")

                st.markdown("### 🧠 Resposta do Assistente Jurídico:")
                st.markdown(str(final_response))

                st.session_state.history.append({
                    "ementa": user_query,
                    "resposta": str(final_response)
                })
                save_history()

                with st.expander("🔍 Ver detalhes do processo de análise da IA"):
                    st.text_area("1. Ementa Extraída para Análise:", value=ementa_extraida, height=150, disabled=True)

                    st.write("2. Consultas Geradas para a Busca:")
                    for i, query in enumerate(queries_otimizadas):
                        st.text_input(f"Consulta {i+1}:", value=query, disabled=True, key=f"query_{i}")

                    st.write(f"3. Súmulas Encontradas ({len(final_retrieved_nodes)} resultados):")
                    for node_com_score in final_retrieved_nodes:
                        sumula_id = node_com_score.node.metadata.get('id', 'N/A')
                        st.markdown(f"**Súmula {sumula_id}** | Similaridade: `{node_com_score.score:.4f}`")
                        st.text_area(
                            label=f"Conteúdo Súmula {sumula_id}", value=node_com_score.node.get_content(),
                            height=80, disabled=True, key=f"content_{node_com_score.node.node_id}",
                            label_visibility="collapsed"
                        )

            except Exception as e:
                st.error(f"❌ Ocorreu um erro durante a análise: {e}")

if st.session_state.page == "home":
    show_home()
else:
    show_analysis()