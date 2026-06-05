# 🌿 PlantAI — Plant Identification & Care Assistant

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-ReAct_Agent-6B46C1)
![BioCLIP](https://img.shields.io/badge/BioCLIP-iNaturalist_10M-2E7D52)
![Groq](https://img.shields.io/badge/Groq-llama--3.3--70b-F55036?logo=groq)

---

![App screenshot](app_screenshot.png)

---

## How it works

```
Photo upload
    └─► BioCLIP vision model       (zero-shot, 58 species)
            └─► Web search          (DuckDuckGo / Tavily)
                    └─► ChromaDB    (RAG vector store)
                            └─► Llama 3 chat
```

A **LangGraph ReAct agent** orchestrates the identification pipeline autonomously — it decides when to call the vision model, when to fetch web articles, and when to answer. No hardcoded flow.

---

## Stack

| Layer | Tech |
|---|---|
| Vision | [BioCLIP](https://huggingface.co/imageomics/bioclip) — trained on 10M biological specimens |
| LLM | Llama 3.3 70B (agent) · Llama 3.1 8B (chat) via Groq |
| RAG | ChromaDB · `all-MiniLM-L6-v2` embeddings · LangChain |
| Agent | LangGraph ReAct with 3 tools |
| UI | Streamlit — sidebar upload, chat interface, session history |

---

## Features

- 🔍 **Zero-shot plant recognition** — no training data needed, ensemble of 3 prompt formats
- 🤖 **Autonomous agent** — identifies species, fetches care articles, builds knowledge base
- 💬 **RAG-powered chat** — answers grounded in real web sources, not hallucinated
- 📋 **Session history** — switch between multiple identified plants mid-conversation
