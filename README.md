# 🦜🔗 LangChain Models Learning Journey

[![Python Version](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-LangChain-emerald.svg)](https://github.com/langchain-ai/langchain)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

Welcome to my personal hands-on laboratory for exploring the **LangChain** ecosystem. This repository documents my practical learning journey, featuring isolated, runnable code experiments that span foundational Large Language Models (LLMs), various Chat Interfaces, and Text Embedding configurations.

---

## 📂 Project Architecture

The codebase is modularly broken down by LangChain's core abstraction layers:

```text
.
├── 1.LLMs/
│   └── 1_llm_demo.py            # Legacy token-in/token-out completions (OpenAI)
├── 2.ChatModels/
│   ├── 1_chatmodels_openai.py   # OpenAI GPT-4 structural implementation
│   ├── 2_chatmodel_anthropic.py # Anthropic Claude 3.5 Sonnet pipeline
│   ├── 3_chatmodels_gemini.py    # Google Gemini 2.5 Flash execution
│   ├── 4_chatmodels_hf_api.py   # Cloud inference via Hugging Face Endpoint API
│   └── 5_chatmodels_hf_local.py # Offline inference using local pipelines (TinyLlama)
├── 3.EmbeddedModels/
│   └── [Pending/In-Progress]    # Dynamic vector embeddings and similarities
├── requirements.txt             # Integrated dependencies
└── README.md                    # Project blueprint


🛠️ Environment Configuration & Setup
Follow these steps to replicate this environment on your local system:
1. Clone the Workspace

git clone [https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git)
cd "LangChain Models"

2. Isolate Dependencies (Virtual Environment)
It is highly recommended to isolate these packages using a virtual environment.

# Create the virtual environment
python -m venv venv

# Activate on macOS/Linux:
source venv/bin/activate

# Activate on Windows:
# .\venv\Scripts\activate

3. Install Requirements
Install all framework bindings and LLM vendor client libraries simultaneously:

pip install -r requirements.txt

4. Inject Secret Keys (.env)
Create a .env file in the root of your project directory to map your credentials safely. Never commit this file to GitHub.

OPENAI_API_KEY="your_openai_secret_key"
ANTHROPIC_API_KEY="your_anthropic_secret_key"
GOOGLE_API_KEY="your_google_gemini_api_key"
HUGGINGFACEHUB_API_TOKEN="your_huggingface_access_token"

🚀 Execution & Verification
To verify that your installation wrappers are mapping perfectly, execute any script directly using your virtual environment's python interpreter:

# Example: Testing the Google Gemini 2.5 Flash pipeline
python "2.ChatModels/3_chatmodels_gemini.py"

💡 Key Conceptual Learnings
LLMs vs. Chat Models: Built scripts detailing the shift from unstructured string-in/string-out wrappers (OpenAI) to structured message-centric interfaces (ChatOpenAI, ChatAnthropic).
Compute Flexibility: Configured architectures running over cloud endpoints (Hugging Face API/Google AI Studio) as well as instances executing 100% locally on my machine utilizing pipeline abstractions (TinyLlama).
Environment Architecture: Hardened production security habits by decoupling strict system variables via python-dotenv.
📈 Roadmap & Core Targets
[x] Baseline LLM & Chat Architecture integrations.
[x] Hugging Face Local vs Cloud Pipeline processing.
[ ] Next Up: Vector Space Mechanics (Tokenizations, Vector Embeddings, and Matrix Similarity indexing).
[ ] Vector Storage Layering (ChromaDB / FAISS orchestration).
[ ] Retrieval Augmented Generation (RAG) implementation pipelines.
⭐ If you're also on a path to mastering AI Orchestration with LangChain, feel free to star this repository and follow along!
