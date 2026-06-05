import streamlit as st
import os, torch
import open_clip
from PIL import Image
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from bs4 import BeautifulSoup
from ddgs import DDGS
import requests

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", os.getenv("TAVILY_API_KEY", ""))

st.set_page_config(page_title="PlantAI", page_icon="🌿", layout="wide")

# Każda para: (nazwa potoczna, nazwa naukowa)
# BioCLIP był trenowany na iNaturalist — nazwy naukowe dają lepsze wyniki
PLANT_SPECIES = [
    ("monstera deliciosa",  "Monstera deliciosa"),
    ("pothos",              "Epipremnum aureum"),
    ("snake plant",         "Dracaena trifasciata"),
    ("peace lily",          "Spathiphyllum wallisii"),
    ("spider plant",        "Chlorophytum comosum"),
    ("rubber plant",        "Ficus elastica"),
    ("fiddle leaf fig",     "Ficus lyrata"),
    ("aloe vera",           "Aloe vera"),
    ("cactus",              "Cactaceae"),
    ("succulent",           "Crassulaceae"),
    ("boston fern",         "Nephrolepis exaltata"),
    ("philodendron",        "Philodendron hederaceum"),
    ("dracaena",            "Dracaena marginata"),
    ("calathea",            "Calathea ornata"),
    ("orchid",              "Phalaenopsis amabilis"),
    ("african violet",      "Streptocarpus ionanthus"),
    ("jade plant",          "Crassula ovata"),
    ("chinese evergreen",   "Aglaonema commutatum"),
    ("ZZ plant",            "Zamioculcas zamiifolia"),
    ("prayer plant",        "Maranta leuconeura"),
    ("bird of paradise",    "Strelitzia reginae"),
    ("anthurium",           "Anthurium andraeanum"),
    ("begonia",             "Begonia rex"),
    ("bromeliad",           "Guzmania lingulata"),
    ("christmas cactus",    "Schlumbergera truncata"),
    ("croton",              "Codiaeum variegatum"),
    ("dieffenbachia",       "Dieffenbachia seguine"),
    ("english ivy",         "Hedera helix"),
    ("geranium",            "Pelargonium hortorum"),
    ("hibiscus",            "Hibiscus rosa-sinensis"),
    ("hosta",               "Hosta plantaginea"),
    ("hydrangea",           "Hydrangea macrophylla"),
    ("impatiens",           "Impatiens walleriana"),
    ("lavender",            "Lavandula angustifolia"),
    ("lemon tree",          "Citrus limon"),
    ("lily",                "Lilium candidum"),
    ("mint",                "Mentha spicata"),
    ("palm tree",           "Arecaceae"),
    ("pansy",               "Viola tricolor"),
    ("peperomia",           "Peperomia obtusifolia"),
    ("rose",                "Rosa hybrida"),
    ("rosemary",            "Salvia rosmarinus"),
    ("schefflera",          "Schefflera actinophylla"),
    ("sedum",               "Sedum spectabile"),
    ("syngonium",           "Syngonium podophyllum"),
    ("tradescantia",        "Tradescantia zebrina"),
    ("umbrella plant",      "Cyperus alternifolius"),
    ("venus flytrap",       "Dionaea muscipula"),
    ("wandering jew",       "Tradescantia fluminensis"),
    ("wisteria",            "Wisteria sinensis"),
    ("yucca",               "Yucca elephantipes"),
    ("zinnia",              "Zinnia elegans"),
    ("basil",               "Ocimum basilicum"),
    ("bamboo",              "Bambusoideae"),
    ("bonsai",              "Ficus retusa"),
    ("clivia",              "Clivia miniata"),
    ("echeveria",           "Echeveria elegans"),
    ("haworthia",           "Haworthiopsis attenuata"),
    ("maranta",             "Maranta leuconeura"),
    ("oxalis",              "Oxalis triangularis"),
]

COMMON_NAMES     = [p[0] for p in PLANT_SPECIES]
SCIENTIFIC_NAMES = [p[1] for p in PLANT_SPECIES]

CONFIDENCE_THRESHOLD = 0.10


@st.cache_resource
def load_bioclip():
    model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
    model.eval()
    return model, preprocess, tokenizer


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=400,
        groq_api_key=GROQ_API_KEY
    )


if "messages" not in st.session_state:
    st.session_state.messages = []
if "plant_name" not in st.session_state:
    st.session_state.plant_name = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


def classify_plant(image):
    model, preprocess, tokenizer = load_bioclip()

    # FIX 1: prompt templates zamiast gołych nazw
    # FIX 2: ensemble nazw potocznych + naukowych
    # FIX 3: hardkodowane 100.0 zamiast model.logit_scale (stabilniejsze dla BioCLIP)
    template_groups = [
        [f"a photo of a {s} plant"  for s in COMMON_NAMES],
        [f"a photo of {s}"          for s in SCIENTIFIC_NAMES],
        [f"a houseplant: {s}"       for s in COMMON_NAMES],
    ]

    img_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        img_f = model.encode_image(img_tensor)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)

        all_probs = []
        for texts in template_groups:
            tokens = tokenizer(texts)
            txt_f  = model.encode_text(tokens)
            txt_f  = txt_f / txt_f.norm(dim=-1, keepdim=True)
            probs  = (100.0 * img_f @ txt_f.T).softmax(dim=-1)[0]
            all_probs.append(probs)

    # Ensemble: uśrednij prawdopodobieństwa ze wszystkich szablonów
    probs = torch.stack(all_probs).mean(dim=0)

    top5 = {COMMON_NAMES[i]: round(probs[i].item(), 4) for i in probs.topk(5).indices}
    best = COMMON_NAMES[probs.argmax().item()]
    conf = probs.max().item()
    return (best if conf >= CONFIDENCE_THRESHOLD else None), conf, top5


def fetch_text(url, limit=5000):
    try:
        soup = BeautifulSoup(
            requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"}).text,
            "html.parser"
        )
        for t in soup(["nav", "footer", "header", "script", "style", "aside"]):
            t.decompose()
        return " ".join(soup.get_text(separator=" ", strip=True).split())[:limit]
    except Exception:
        return ""


def get_articles(plant_name):
    query = plant_name + " plant watering schedule light requirements soil"
    if TAVILY_API_KEY:
        from tavily import TavilyClient
        res = TavilyClient(api_key=TAVILY_API_KEY).search(
            query, max_results=3, include_raw_content=True
        )
        return [
            {
                "title": r.get("title", ""),
                "url":   r.get("url", ""),
                "text":  (r.get("raw_content") or r.get("content", ""))[:5000]
            }
            for r in res.get("results", [])
        ]
    docs = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=3):
            text = fetch_text(r.get("href", "")) or r.get("body", "")
            if text:
                docs.append({
                    "title": r.get("title", ""),
                    "url":   r.get("href", ""),
                    "text":  text
                })
    return docs


def build_vs(plant_name, articles):
    emb     = load_embeddings()
    persist = "./chroma_db/" + plant_name.replace(" ", "_")
    if os.path.exists(persist):
        return Chroma(persist_directory=persist, embedding_function=emb)
    docs = [
        Document(page_content=a["text"], metadata={"source": a["url"]})
        for a in articles if a.get("text")
    ]
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=30
    ).split_documents(docs)
    return Chroma.from_documents(chunks, embedding=emb, persist_directory=persist)


def get_answer(question, vs, plant_name):
    docs   = vs.as_retriever(search_kwargs={"k": 2}).invoke(question)
    ctx    = "\n\n".join(d.page_content for d in docs)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are PlantAI, an expert plant care assistant. "
         "Current plant: {plant_name}. "
         "Be practical and specific. Use exact numbers when available."
         "\n\nContext:\n{context}"),
        ("human", "{question}")
    ])
    return (prompt | get_llm() | StrOutputParser()).invoke({
        "context":    ctx,
        "plant_name": plant_name,
        "question":   question
    })


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌿 PlantAI")
    st.caption("Zidentyfikuj roślinę i zapytaj o pielęgnację")
    st.divider()
    uploaded = st.file_uploader(
        "Prześlij zdjęcie rośliny", type=["jpg", "jpeg", "png", "webp"]
    )
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True)
        if st.button("🔍 Rozpoznaj roślinę", use_container_width=True):
            with st.spinner("Analizuję zdjęcie (BioCLIP)..."):
                name, conf, top5 = classify_plant(image)
            if name:
                st.success(f"**{name.title()}** ({conf:.0%})")
                with st.spinner("Szukam artykułów i buduję bazę wiedzy..."):
                    arts = get_articles(name)
                    vs   = build_vs(name, arts)
                    st.session_state.plant_name  = name
                    st.session_state.vectorstore = vs
                st.success(f"Gotowe! Znaleziono {len(arts)} artykułów.")
                with st.expander("Top-5 predykcji BioCLIP"):
                    for sp, pr in top5.items():
                        st.progress(pr, text=f"{sp} ({pr:.1%})")
            else:
                st.error(f"Nie rozpoznano ({conf:.0%}). Spróbuj wyraźniejszego zdjęcia.")
    if st.session_state.plant_name:
        st.divider()
        st.info(f"🌱 **{st.session_state.plant_name.title()}**")
    if st.button("🗑️ Wyczyść czat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Chat ──────────────────────────────────────────────────────────────────────
st.title("🌿 PlantAI")
if not st.session_state.plant_name:
    st.info("← Prześlij zdjęcie rośliny w panelu bocznym, aby rozpocząć.")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
if q := st.chat_input("Zapytaj o pielęgnację..."):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        if not st.session_state.vectorstore:
            ans = "Najpierw prześlij zdjęcie i kliknij Rozpoznaj roślinę."
        else:
            with st.spinner("Szukam odpowiedzi..."):
                ans = get_answer(
                    q, st.session_state.vectorstore, st.session_state.plant_name
                )
        st.markdown(ans)
        st.session_state.messages.append({"role": "assistant", "content": ans})
