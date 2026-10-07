# ------------------------------------------------------------------------------
# 本文件为 awesome-llm-apps 中 "Corrective RAG" 示例的修改版本（衍生作品）
#
#   原作者：Shubham Saboo
#   原项目：https://github.com/Shubhamsaboo/awesome-llm-apps
#   原始路径：rag_tutorials/corrective_rag/corrective_rag.py
#   许可证：Apache License 2.0
#
# 本版本所做修改：
#   - LLM 由 Anthropic Claude 更换为 DeepSeek（OpenAI 兼容接口）
#   - 向量化由 OpenAI embedding 更换为智谱 GLM embedding
#   - 依赖升级至 LangChain / LangGraph 1.x，并迁移对应导入路径
#   - 向量库初始化改为自动探测向量维度（force_recreate=True）
#   - 新增 .env 环境变量支持（python-dotenv）
#   - 修复最终生成结果的取值 bug；清理未使用的导入
#
# 完整改动清单见 NOTICE 与 README.md。
# ------------------------------------------------------------------------------
import json
import os
import re
import tempfile
from typing import Any, Dict, TypedDict
from urllib.parse import urlparse

import nest_asyncio
import pprint
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from tenacity import retry, stop_after_attempt, wait_exponential

# Optional: read keys from a local `.env` file so they do not have to be
# re-typed in the sidebar on every run.
load_dotenv()

nest_asyncio.apply()

retriever = None

# ---- Defaults (all overridable in the sidebar or via .env) -------------------
DEFAULT_DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEFAULT_DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEFAULT_ZHIPU_BASE_URL = os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
DEFAULT_ZHIPU_EMBEDDING_MODEL = os.getenv("ZHIPU_EMBEDDING_MODEL", "embedding-3")
DEFAULT_DOC_URL = "https://arxiv.org/pdf/2307.09288.pdf"


def initialize_session_state():
    """Initialize session state variables for API keys and configuration."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        # DeepSeek (LLM)
        st.session_state.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        st.session_state.deepseek_base_url = DEFAULT_DEEPSEEK_BASE_URL
        st.session_state.deepseek_model = DEFAULT_DEEPSEEK_MODEL
        # Zhipu (Embedding)
        st.session_state.zhipu_api_key = os.getenv("ZHIPU_API_KEY", "")
        st.session_state.zhipu_base_url = DEFAULT_ZHIPU_BASE_URL
        st.session_state.zhipu_embedding_model = DEFAULT_ZHIPU_EMBEDDING_MODEL
        # Qdrant (vector store)
        st.session_state.qdrant_url = os.getenv("QDRANT_URL", "")
        st.session_state.qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
        # Tavily (web search)
        st.session_state.tavily_api_key = os.getenv("TAVILY_API_KEY", "")
        st.session_state.doc_url = DEFAULT_DOC_URL


def setup_sidebar():
    """Setup sidebar for API keys and configuration."""
    with st.sidebar:
        st.subheader("API Configuration")
        st.caption("LLM: DeepSeek · Embedding: Zhipu (GLM) · Vector store: Qdrant")

        st.markdown("**DeepSeek (LLM)**")
        st.session_state.deepseek_api_key = st.text_input(
            "DeepSeek API Key", value=st.session_state.deepseek_api_key, type="password",
            help="https://platform.deepseek.com",
        )
        st.session_state.deepseek_base_url = st.text_input(
            "DeepSeek Base URL", value=st.session_state.deepseek_base_url,
        )
        st.session_state.deepseek_model = st.text_input(
            "DeepSeek Model", value=st.session_state.deepseek_model,
            help="deepseek-chat (V3) or deepseek-reasoner (R1)",
        )

        st.markdown("**Zhipu / GLM (Embedding)**")
        st.session_state.zhipu_api_key = st.text_input(
            "Zhipu API Key", value=st.session_state.zhipu_api_key, type="password",
            help="https://open.bigmodel.cn",
        )
        st.session_state.zhipu_base_url = st.text_input(
            "Zhipu Base URL", value=st.session_state.zhipu_base_url,
        )
        st.session_state.zhipu_embedding_model = st.text_input(
            "Zhipu Embedding Model", value=st.session_state.zhipu_embedding_model,
            help="embedding-3 or embedding-2",
        )

        st.markdown("**Qdrant (Vector store)**")
        st.session_state.qdrant_url = st.text_input(
            "Qdrant URL", value=st.session_state.qdrant_url,
        )
        st.session_state.qdrant_api_key = st.text_input(
            "Qdrant API Key", value=st.session_state.qdrant_api_key, type="password",
        )

        st.markdown("**Tavily (Web search, optional)**")
        st.session_state.tavily_api_key = st.text_input(
            "Tavily API Key", value=st.session_state.tavily_api_key, type="password",
            help="Optional - web search is skipped if left empty.",
        )
        st.session_state.doc_url = st.text_input("Document URL", value=st.session_state.doc_url)

        required = [
            st.session_state.deepseek_api_key,
            st.session_state.zhipu_api_key,
            st.session_state.qdrant_url,
            st.session_state.qdrant_api_key,
        ]
        if not all(required):
            st.warning("Please provide DeepSeek API Key, Zhipu API Key, Qdrant URL and Qdrant API Key.")
            st.stop()

        st.session_state.initialized = True


initialize_session_state()
setup_sidebar()


def get_llm():
    """Return the DeepSeek LLM through its OpenAI-compatible endpoint."""
    return ChatOpenAI(
        model=st.session_state.deepseek_model,
        api_key=st.session_state.deepseek_api_key,
        base_url=st.session_state.deepseek_base_url,
        temperature=0,
        max_tokens=1000,
    )


embeddings = OpenAIEmbeddings(
    model=st.session_state.zhipu_embedding_model,
    api_key=st.session_state.zhipu_api_key,
    base_url=st.session_state.zhipu_base_url,
)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def execute_tavily_search(tool, query):
    return tool.invoke({"query": query})


def web_search(state):
    """Web search based on the re-phrased question using Tavily API."""
    print("~-web search-~")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    progress_placeholder = st.empty()
    progress_placeholder.info("Initiating web search...")

    try:
        if not st.session_state.tavily_api_key:
            progress_placeholder.warning("Tavily API key not provided - skipping web search")
            return {"keys": {"documents": documents, "question": question}}

        progress_placeholder.info("Configuring search tool...")
        tool = TavilySearch(
            tavily_api_key=st.session_state.tavily_api_key,
            max_results=3,
            search_depth="advanced",
        )

        progress_placeholder.info("Executing search query...")
        try:
            raw_results = execute_tavily_search(tool, question)
        except Exception as search_error:
            progress_placeholder.error(f"Search failed after retries: {str(search_error)}")
            return {"keys": {"documents": documents, "question": question}}

        # TavilySearch returns {"query": ..., "results": [{"title", "content", ...}, ...]}
        results = raw_results.get("results", []) if isinstance(raw_results, dict) else (raw_results or [])

        if not results:
            progress_placeholder.warning("No search results found")
            return {"keys": {"documents": documents, "question": question}}

        progress_placeholder.info("Processing search results...")
        web_results = []
        for result in results:
            content = (
                f"Title: {result.get('title', 'No title')}\n"
                f"Content: {result.get('content', 'No content')}\n"
            )
            web_results.append(content)

        web_document = Document(
            page_content="\n\n".join(web_results),
            metadata={
                "source": "tavily_search",
                "query": question,
                "result_count": len(web_results),
            },
        )
        documents.append(web_document)
        progress_placeholder.success(f"Successfully added {len(web_results)} search results")

    except Exception as error:
        error_msg = f"Web search error: {str(error)}"
        print(error_msg)
        progress_placeholder.error(error_msg)
    finally:
        progress_placeholder.empty()
    return {"keys": {"documents": documents, "question": question}}


def load_documents(file_or_url: str, is_url: bool = True) -> list:
    try:
        if is_url:
            # A .pdf URL must be parsed as a PDF; WebBaseLoader would run an HTML
            # parser over the binary body and embed decoded garbage.
            if urlparse(file_or_url).path.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_or_url)
            else:
                loader = WebBaseLoader(file_or_url)
                loader.requests_per_second = 1
        else:
            file_extension = os.path.splitext(file_or_url)[1].lower()
            if file_extension == ".pdf":
                loader = PyPDFLoader(file_or_url)
            elif file_extension in [".txt", ".md"]:
                loader = TextLoader(file_or_url)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

        return loader.load()
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return []


st.subheader("Document Input")
input_option = st.radio("Choose input method:", ["URL", "File Upload"])

docs = None

if input_option == "URL":
    url = st.text_input("Enter document URL:", value=st.session_state.doc_url)
    if url:
        docs = load_documents(url, is_url=True)
else:
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "md"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            docs = load_documents(tmp_file.name, is_url=False)
        os.unlink(tmp_file.name)

# Re-ingest only when the source actually changes (Streamlit reruns the whole
# script on every widget interaction, e.g. when the user types a question).
source_key = url if input_option == "URL" else (uploaded_file.name if uploaded_file else None)

if docs and st.session_state.get("ingested_source") != source_key:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_splits = text_splitter.split_documents(docs)

    collection_name = "rag-qdrant"
    # force_recreate=True deletes/recreates the collection and auto-detects the
    # vector dimension from the embedding model, so re-ingestion is safe even if
    # the embedding model (and therefore its dimension) changes.
    vectorstore = QdrantVectorStore.from_documents(
        all_splits,
        embeddings,
        url=st.session_state.qdrant_url,
        api_key=st.session_state.qdrant_api_key,
        collection_name=collection_name,
        force_recreate=True,
    )
    retriever = vectorstore.as_retriever()
    st.session_state.ingested_source = source_key
    st.session_state.retriever = retriever
elif st.session_state.get("retriever") is not None:
    retriever = st.session_state.retriever


class GraphState(TypedDict):
    keys: Dict[str, Any]


def retrieve(state):
    print("~-retrieve-~")
    state_dict = state["keys"]
    question = state_dict["question"]

    if retriever is None:
        return {"keys": {"documents": [], "question": question}}

    documents = retriever.invoke(question)
    return {"keys": {"documents": documents, "question": question}}


def generate(state):
    """Generate an answer grounded in the retrieved (+ searched) documents."""
    print("~-generate-~")
    state_dict = state["keys"]
    question, documents = state_dict["question"], state_dict["documents"]
    try:
        prompt = PromptTemplate(
            template="""Based on the following context, please answer the question.
            Context: {context}
            Question: {question}
            Answer:""",
            input_variables=["context", "question"],
        )
        context = "\n\n".join(doc.page_content for doc in documents)

        rag_chain = (
            {"context": lambda x: context, "question": lambda x: question}
            | prompt
            | get_llm()
            | StrOutputParser()
        )
        generation = rag_chain.invoke({})

        return {
            "keys": {
                "documents": documents,
                "question": question,
                "generation": generation,
            }
        }
    except Exception as e:
        error_msg = f"Error in generate function: {str(e)}"
        print(error_msg)
        st.error(error_msg)
        return {
            "keys": {
                "documents": documents,
                "question": question,
                "generation": "Sorry, I encountered an error while generating the response.",
            }
        }


def grade_documents(state):
    """Determine whether the retrieved documents are relevant to the question."""
    print("~-check relevance-~")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    prompt = PromptTemplate(
        template="""You are grading the relevance of a retrieved document to a user question.
        Return ONLY a JSON object with a "score" field that is either "yes" or "no".
        Do not include any other text or explanation.

        Document: {context}
        Question: {question}

        Rules:
        - Check for related keywords or semantic meaning
        - Use lenient grading to only filter clear mismatches
        - Return exactly like this example: {{"score": "yes"}} or {{"score": "no"}}""",
        input_variables=["context", "question"],
    )

    chain = prompt | get_llm() | StrOutputParser()

    filtered_docs = []
    search = "No"

    for d in documents:
        try:
            response = chain.invoke({"question": question, "context": d.page_content})
            json_match = re.search(r"\{.*\}", response)
            if json_match:
                response = json_match.group()

            score = json.loads(response)
            if score.get("score") == "yes":
                print("~-grade: document relevant-~")
                filtered_docs.append(d)
            else:
                print("~-grade: document not relevant-~")
                search = "Yes"
        except Exception as e:
            print(f"Error grading document: {str(e)}")
            # On error, keep the document to be safe.
            filtered_docs.append(d)
            continue

    return {"keys": {"documents": filtered_docs, "question": question, "run_web_search": search}}


def transform_query(state):
    """Transform the query to produce a better search question."""
    print("~-transform query-~")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    prompt = PromptTemplate(
        template="""Generate a search-optimized version of this question by
        analyzing its core semantic meaning and intent.
        \n ------- \n
        {question}
        \n ------- \n
        Return only the improved question with no additional text:""",
        input_variables=["question"],
    )

    chain = prompt | get_llm() | StrOutputParser()
    better_question = chain.invoke({"question": question})

    return {"keys": {"documents": documents, "question": better_question}}


def decide_to_generate(state):
    print("~-decide to generate-~")
    state_dict = state["keys"]
    search = state_dict["run_web_search"]

    if search == "Yes":
        print("~-decision: transform query and run web search-~")
        return "transform_query"
    else:
        print("~-decision: generate-~")
        return "generate"


def format_document(doc: Document) -> str:
    return f"""
    Source: {doc.metadata.get('source', 'Unknown')}
    Title: {doc.metadata.get('title', 'No title')}
    Content: {doc.page_content[:200]}...
    """


def format_state(state: dict) -> str:
    formatted = {}
    for key, value in state.items():
        if key == "documents":
            formatted[key] = [format_document(doc) for doc in value]
        else:
            formatted[key] = value
    return formatted


workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search", web_search)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "web_search")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

st.title("🔄 Corrective RAG Agent")

st.text("A possible query: What are the experiment results and ablation studies in this research paper?")

user_question = st.text_input("Please enter your question:")

if user_question:
    inputs = {"keys": {"question": user_question}}

    final_generation = "No final generation produced."
    for output in app.stream(inputs):
        for key, value in output.items():
            with st.expander(f"Step '{key}':"):
                st.text(pprint.pformat(format_state(value["keys"]), indent=2, width=80))
            if "generation" in value["keys"]:
                final_generation = value["keys"]["generation"]

    st.subheader("Final Generation:")
    st.write(final_generation)