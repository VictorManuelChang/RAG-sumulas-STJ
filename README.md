
# Analisador Jurídico de Ementas com RAG 

## 1\. Visão Geral

Este projeto é um assistente jurídico inteligente que utiliza uma arquitetura avançada de **Geração Aumentada por Recuperação (RAG)** para analisar ementas jurídicas e verificar sua consonância com as 676 Súmulas do Superior Tribunal de Justiça (STJ).

Indo além de uma simples busca semântica, esta aplicação implementa um pipeline de múltiplos estágios que extrai, analisa e otimiza a consulta do usuário para garantir a maior precisão possível na recuperação dos documentos relevantes, culminando em uma análise jurídica coesa e bem fundamentada gerada por um LLM de ponta.

## 2\. Funcionalidades Principais

  - **Interface Web Interativa:** Apresenta uma interface simples e intuitiva construída com Streamlit, permitindo que qualquer usuário utilize a ferramenta facilmente.
  - **Extração Inteligente de Ementa:** O sistema primeiro isola o texto jurídico relevante (a ementa) da entrada do usuário, ignorando instruções, perguntas ou formatação desnecessária.
  - **Geração de Múltiplas Consultas (Multi-Query Engine):** Em vez de depender de uma única interpretação, a aplicação analisa a ementa extraída sob diferentes perspectivas (termos-chave, tese principal, instituto jurídico) para gerar um conjunto diversificado de consultas de busca.
  - **Fusão e Reclassificação de Resultados (Result Fusion):** Os resultados da busca de cada consulta gerada são agregados, desduplicados e reclassificados com base na relevância (score de similaridade), garantindo que o contexto final seja o mais completo e preciso possível.
  - **Análise Jurídica com LLM:** Utiliza o GPT-4-Turbo para a etapa final de síntese, gerando uma análise detalhada e bem estruturada sobre a consonância entre a ementa do usuário e as súmulas recuperadas.
  - **Uso Otimizado de LLMs:** Emprega modelos de forma estratégica: `GPT-3.5-Turbo` para as tarefas rápidas de pré-processamento (extração e geração de consultas) e `GPT-4-Turbo` para a análise final.

## 3\. Fluxo do Sistema

O projeto é dividido em duas fases principais: uma preparação offline e a análise em tempo real.

### Fase A: Coleta e Indexação (Offline)

1.  **Web Scraping (`scrapper.py`):** Coleta o texto de todas as 676 Súmulas do site do STJ e os salva em `sumulas.json`.
2.  **Indexação Vetorial (`build.py`):**
      - Cada súmula é processada pelo modelo de embedding `neuralmind/bert-base-portuguese-cased`.
      - Os vetores resultantes são armazenados em um índice FAISS na pasta `./storage`.

### Fase B: Pipeline de Análise RAG (Tempo Real em `app.py`)

O fluxo de análise quando um usuário submete um texto é o seguinte:

1.  **Input do Usuário:** O usuário cola um texto, que pode conter a ementa junto com outras instruções.
2.  **Extração da Ementa:** Um LLM (GPT-3.5-Turbo) é usado para identificar e extrair apenas o bloco de texto correspondente à ementa jurídica oficial.
3.  **Geração de Múltiplas Consultas:** A ementa extraída é enviada novamente ao LLM para gerar 3 consultas otimizadas a partir de diferentes ângulos (ex: palavras-chave, resumo da tese, etc.).
4.  **Recuperação Paralela:** O sistema executa uma busca vetorial no índice FAISS para **cada uma** das 3 consultas geradas, recuperando um conjunto de súmulas para cada uma.
5.  **Fusão de Resultados:** Todos os resultados das buscas são combinados em uma única lista. Duplicatas são removidas e a lista é reordenada pela pontuação de similaridade, mantendo os 10 documentos mais relevantes.
6.  **Geração da Resposta Final:** A ementa **original** do usuário e as 10 súmulas **finais e mais relevantes** são enviadas a um LLM de alta capacidade (GPT-4-Turbo) junto com um prompt detalhado para gerar a análise jurídica final, que é então exibida na tela.

## 4\. Tecnologias Utilizadas

  - **Linguagem:** Python 3.10+
  - **Framework RAG:** LlamaIndex
  - **Interface Web:** Streamlit
  - **Modelos de Linguagem (LLMs):**
      - OpenAI GPT-3.5-Turbo (para extração e geração de múltiplas consultas)
      * OpenAI GPT-4-Turbo (para a geração da resposta final)
  - **Modelo de Embedding:** `neuralmind/bert-base-portuguese-cased`
  - **Banco de Dados Vetorial:** FAISS
  - **Coleta de Dados:** Requests & BeautifulSoup

## 5\. Como Executar o Projeto

Siga os passos abaixo para configurar e executar a aplicação localmente.

### Passo a Passo

1.  **Clone o Repositório**

    ```bash
    git clone [URL_DO_SEU_REPOSITORIO_GIT]
    cd [NOME_DA_PASTA_DO_PROJETO]
    ```


2.  **Instale as Dependências**
    Crie um arquivo `requirements.txt` com o conteúdo abaixo e execute o comando de instalação.

    ```text
    streamlit
    python-dotenv
    llama-index
    llama-index-llms-openai
    llama-index-embeddings-huggingface
    sentence-transformers
    faiss-cpu
    requests
    beautifulsoup4
    ```

    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure a Chave de API da OpenAI**

      - Crie um arquivo chamado `.env` na raiz do projeto.
      - Adicione sua chave de API da OpenAI a este arquivo:
        ```
        OPENAI_API_KEY="sk-..."
        ```

4.  **Execute os Scripts de Preparação (Apenas na Primeira Vez)**

      - **Coleta de Dados:**
        ```bash
        python3 scrapper.py
        ```
      - **Construção do Índice Vetorial:**
        ```bash
        python3 build.py
        ```

5.  **Execute a Aplicação Web**

    ```bash
    streamlit run app.py
    ```


## 6\. Como Usar a Aplicação

1.  Abra o URL fornecido pelo Streamlit no seu navegador.
2.  Cole o texto que contém a ementa jurídica na caixa de texto. Você pode incluir perguntas ou instruções adicionais, pois o sistema é projetado para extrair a ementa automaticamente.
3.  Clique no botão "Analisar Ementa".
4.  Aguarde a resposta. Para entender o processo de tomada de decisão do sistema, expanda a seção "🔍 Informações de Depuração" para ver a ementa extraída, as consultas geradas e a lista final de súmulas recuperadas.