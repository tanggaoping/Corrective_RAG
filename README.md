# 🔄 Corrective RAG Agent

一个基于 **LangGraph + Streamlit** 实现的**纠正式检索增强生成（Corrective RAG, CRAG）** Agent。

它不再只是「检索 → 生成」的简单流程，而是在检索之后增加了一个**文档相关性评估**步骤：只有当检索到的文档确实相关时才直接生成答案；一旦发现不相关的文档，就会**改写查询**并触发**网络搜索**来补充信息，从而提升回答的准确性和可靠性。

## 📌 项目来源

本项目是 [awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps)（作者 [Shubham Saboo](https://github.com/Shubhamsaboo)）中 `rag_tutorials/corrective_rag` 示例的**衍生作品（Derivative Work）**，沿用原项目的 **Apache License 2.0** 分发。

- 原作者：Shubham Saboo
- 原仓库：https://github.com/Shubhamsaboo/awesome-llm-apps
- 原始文件：`rag_tutorials/corrective_rag/corrective_rag.py`
- 完整声明见 [NOTICE](NOTICE) 与 [LICENSE](LICENSE)

## 🔧 相对原版的改动

1. **LLM 更换**：Anthropic Claude 4.5 Sonnet → **DeepSeek**（`deepseek-chat` / `deepseek-reasoner`，OpenAI 兼容接口）。
2. **向量化模型更换**：OpenAI `text-embedding-3-small` → **智谱 GLM** `embedding-3`（OpenAI 兼容接口）。
3. **依赖升级至 LangChain 1.x**：`langchain` 0.3.x → 1.x、`langgraph` 0.2.x → 1.x，并迁移对应导入路径（`langchain_qdrant.QdrantVectorStore`、`langchain_tavily.TavilySearch` 等）。
4. **向量库初始化简化**：改为 `QdrantVectorStore.from_documents(force_recreate=True)`，自动探测向量维度，替换原先硬编码 `1536` 维度的手动建集合逻辑。
5. **新增 .env 环境变量支持**：引入 `python-dotenv`，启动时自动读取密钥（也可在侧边栏填写），并支持在侧边栏配置 Base URL 与模型名。
6. **修复最终输出取值 bug**：原版在 `app.stream()` 循环外直接引用 `value`，可能导致未定义或取错节点；改为在循环内捕获 `generation`。
7. **清理代码**：移除 `tiktoken` 依赖及未使用的导入（`ChatAnthropic`、`QdrantClient`、`yaml` 等）。

## ✨ 核心流程

```
用户提问
   │
   ▼
① retrieve        从 Qdrant 向量库检索相关文档
   │
   ▼
② grade_documents 逐条判断文档与问题是否相关（LLM 打分 yes/no）
   │
   ├── 全部相关 ──────────────────► ⑥ generate  基于上下文生成最终答案
   │
   └── 存在不相关 ──► ③ transform_query  改写/优化搜索问题
                         │
                         ▼
                     ④ web_search     调用 Tavily 网络搜索补充内容
                         │
                         ▼
                     ⑤ generate       基于「文档 + 搜索结果」生成最终答案
```

## 🧩 技术栈

| 组件 | 选型 | 说明 |
| --- | --- | --- |
| 编排框架 | [LangGraph](https://www.langchain.com/langgraph) | 定义并编译有状态的工作流 |
| 大语言模型 | [DeepSeek](https://platform.deepseek.com) | `deepseek-chat` / `deepseek-reasoner`，OpenAI 兼容接口 |
| 文本向量化 | [智谱 GLM](https://open.bigmodel.cn) | `embedding-3`，OpenAI 兼容接口 |
| 向量数据库 | [Qdrant](https://cloud.qdrant.io) | 存储与检索文档分块 |
| 网络搜索 | [Tavily](https://tavily.com) | 可选，未配置时自动跳过 |
| 前端界面 | [Streamlit](https://streamlit.io) | 侧边栏配置 + 对话式交互 |

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/tanggaoping/Corrective_RAG.git
cd Corrective_RAG
```

### 2. 安装依赖

> 建议使用 Python 3.10+ 并创建独立虚拟环境。

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. 配置 API 密钥

复制示例配置文件，并填入你自己的密钥：

```bash
cp .env.example .env
```

编辑 `.env`：

```dotenv
# DeepSeek（LLM）
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat

# 智谱 GLM（向量化）
ZHIPU_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx.xxxxxxxx
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4
ZHIPU_EMBEDDING_MODEL=embedding-3

# Qdrant（向量数据库）
QDRANT_URL=https://xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.us-east-2-0.aws.cloud.qdrant.io
QDRANT_API_KEY=eyJxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Tavily（网络搜索，可选）
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxxxxxx
```

> 密钥也可以在启动后于 Streamlit 侧边栏中手动填写，二者二选一即可。

### 4. 启动应用

```bash
streamlit run corrective_rag.py
```

浏览器会自动打开 `http://localhost:8501`。

## 📖 使用说明

1. 在左侧边栏填写所需 API 密钥（DeepSeek、智谱、Qdrant 为必填，Tavily 可选）。
2. 选择文档来源：
   - **URL**：支持 PDF 或网页链接（默认示例为 Llama 2 论文 `https://arxiv.org/pdf/2307.09288.pdf`）。
   - **File Upload**：支持上传 `pdf` / `txt` / `md` 文件。
3. 文档会被自动切分（chunk_size=500，overlap=100）并写入 Qdrant 向量库。
4. 在下方输入问题，即可查看每个节点的中间结果与最终回答。

示例问题：

> What are the experiment results and ablation studies in this research paper?

## 📁 项目结构

```
.
├── corrective_rag.py   # 主程序：Corrective RAG 工作流 + Streamlit 界面（已修改）
├── requirements.txt    # 依赖清单（langchain 1.x，国内可访问版本）
├── .env.example        # 环境变量示例（复制为 .env 后填写密钥）
├── NOTICE              # 原作者与改动声明
├── LICENSE             # Apache License 2.0
├── README.md           # 项目说明
└── .gitignore          # 忽略规则
```

## ⚠️ 注意事项

- **请勿将 `.env` 提交到仓库**，其中包含真实的 API 密钥。`.gitignore` 已默认排除 `.env`。
- 每次切换文档源时，会以 `force_recreate=True` 重建 Qdrant 集合（集合名 `rag-qdrant`），请确保该集合名没有被其他数据占用。
- Tavily 为可选依赖，未配置 `TAVILY_API_KEY` 时会自动跳过网络搜索步骤。
- 本项目默认使用 DeepSeek 与智谱 GLM，两者均提供 OpenAI 兼容接口，因此在代码中通过 `langchain-openai` 统一调用。

## 📄 License

本项目沿用原项目的 [Apache License 2.0](LICENSE) 分发，详见 [NOTICE](NOTICE)。
