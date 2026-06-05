import streamlit as st
import os, torch, tempfile
from datetime import datetime
import open_clip
from PIL import Image
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from bs4 import BeautifulSoup
from ddgs import DDGS
import requests

GROQ_API_KEY   = st.secrets.get("GROQ_API_KEY",   os.getenv("GROQ_API_KEY",   ""))
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY",  os.getenv("TAVILY_API_KEY", ""))

st.set_page_config(page_title="PlantAI", page_icon="🌿", layout="wide")

# ── Species database ──────────────────────────────────────────────────────────
# Każda pozycja: (nazwa_potoczna, nazwa_naukowa, taksonomia_bioclip)
#
# BioCLIP był trenowany na iNaturalist z pełną hierarchią taksonomiczną jako
# etykietą tekstową. Format: "Kingdom Phylum Class Order Family Genus Species"
# To natywny format promptów dla tego modelu — znacznie lepsze wyniki niż
# gołe nazwy ("pothos") czy angielskie powszechne ("snake plant").
PLANT_SPECIES = [
    ("monstera deliciosa",  "Monstera deliciosa",
     "Plantae Tracheophyta Magnoliopsida Alismatales Araceae Monstera Monstera deliciosa"),
    ("pothos",              "Epipremnum aureum",
     "Plantae Tracheophyta Magnoliopsida Alismatales Araceae Epipremnum Epipremnum aureum"),
    ("snake plant",         "Dracaena trifasciata",
     "Plantae Tracheophyta Magnoliopsida Asparagales Asparagaceae Dracaena Dracaena trifasciata"),
    ("peace lily",          "Spathiphyllum wallisii",
     "Plantae Tracheophyta Magnoliopsida Alismatales Araceae Spathiphyllum Spathiphyllum wallisii"),
    ("spider plant",        "Chlorophytum comosum",
     "Plantae Tracheophyta Magnoliopsida Asparagales Asparagaceae Chlorophytum Chlorophytum comosum"),
    ("rubber plant",        "Ficus elastica",
     "Plantae Tracheophyta Magnoliopsida Rosales Moraceae Ficus Ficus elastica"),
    ("fiddle leaf fig",     "Ficus lyrata",
     "Plantae Tracheophyta Magnoliopsida Rosales Moraceae Ficus Ficus lyrata"),
    ("aloe vera",           "Aloe vera",
     "Plantae Tracheophyta Magnoliopsida Asparagales Asphodelaceae Aloe Aloe vera"),
    ("ZZ plant",            "Zamioculcas zamiifolia",
     "Plantae Tracheophyta Magnoliopsida Alismatales Araceae Zamioculcas Zamioculcas zamiifolia"),
    ("philodendron",        "Philodendron hederaceum",
     "Plantae Tracheophyta Magnoliopsida Alismatales Araceae Philodendron Philodendron hederaceum"),
    ("dracaena",            "Dracaena marginata",
     "Plantae Tracheophyta Magnoliopsida Asparagales Asparagaceae Dracaena Dracaena marginata"),
    ("calathea",            "Goeppertia ornata",
     "Plantae Tracheophyta Magnoliopsida Zingiberales Marantaceae Goeppertia Goeppertia ornata"),
    ("orchid",              "Phalaenopsis amabilis",
     "Plantae Tracheophyta Magnoliopsida Asparagales Orchidaceae Phalaenopsis Phalaenopsis amabilis"),
    ("african violet",      "Streptocarpus ionanthus",
     "Plantae Tracheophyta Magnoliopsida Lamiales Gesneriaceae Streptocarpus Streptocarpus ionanthus"),
    ("jade plant",          "Crassula ovata",
     "Plantae Tracheophyta Magnoliopsida Saxifragales Crassulaceae Crassula Crassula ovata"),
    ("chinese evergreen",   "Aglaonema commutatum",
     "Plantae Tracheophyta Magnoliopsida Alismatales Araceae Aglaonema Aglaonema commutatum"),
    ("prayer plant",        "Maranta leuconeura",
     "Plantae Tracheophyta Magnoliopsida Zingiberales Marantaceae Maranta Maranta leuconeura"),
    ("bird of paradise",    "Strelitzia reginae",
     "Plantae Tracheophyta Magnoliopsida Zingiberales Strelitziaceae Strelitzia Strelitzia reginae"),
    ("anthurium",           "Anthurium andraeanum",
     "Plantae Tracheophyta Magnoliopsida Alismatales Araceae Anthurium Anthurium andraeanum"),
    ("begonia",             "Begonia rex",
     "Plantae Tracheophyta Magnoliopsida Cucurbitales Begoniaceae Begonia Begonia rex"),
    ("bromeliad",           "Guzmania lingulata",
     "Plantae Tracheophyta Magnoliopsida Poales Bromeliaceae Guzmania Guzmania lingulata"),
    ("christmas cactus",    "Schlumbergera truncata",
     "Plantae Tracheophyta Magnoliopsida Caryophyllales Cactaceae Schlumbergera Schlumbergera truncata"),
    ("croton",              "Codiaeum variegatum",
     "Plantae Tracheophyta Magnoliopsida Malpighiales Euphorbiaceae Codiaeum Codiaeum variegatum"),
    ("dieffenbachia",       "Dieffenbachia seguine",
     "Plantae Tracheophyta Magnoliopsida Alismatales Araceae Dieffenbachia Dieffenbachia seguine"),
    ("english ivy",         "Hedera helix",
     "Plantae Tracheophyta Magnoliopsida Apiales Araliaceae Hedera Hedera helix"),
    ("geranium",            "Pelargonium hortorum",
     "Plantae Tracheophyta Magnoliopsida Geraniales Geraniaceae Pelargonium Pelargonium hortorum"),
    ("hibiscus",            "Hibiscus rosa-sinensis",
     "Plantae Tracheophyta Magnoliopsida Malvales Malvaceae Hibiscus Hibiscus rosa-sinensis"),
    ("hosta",               "Hosta plantaginea",
     "Plantae Tracheophyta Magnoliopsida Asparagales Asparagaceae Hosta Hosta plantaginea"),
    ("hydrangea",           "Hydrangea macrophylla",
     "Plantae Tracheophyta Magnoliopsida Cornales Hydrangeaceae Hydrangea Hydrangea macrophylla"),
    ("impatiens",           "Impatiens walleriana",
     "Plantae Tracheophyta Magnoliopsida Ericales Balsaminaceae Impatiens Impatiens walleriana"),
    ("lavender",            "Lavandula angustifolia",
     "Plantae Tracheophyta Magnoliopsida Lamiales Lamiaceae Lavandula Lavandula angustifolia"),
    ("lemon tree",          "Citrus limon",
     "Plantae Tracheophyta Magnoliopsida Sapindales Rutaceae Citrus Citrus limon"),
    ("lily",                "Lilium candidum",
     "Plantae Tracheophyta Magnoliopsida Liliales Liliaceae Lilium Lilium candidum"),
    ("mint",                "Mentha spicata",
     "Plantae Tracheophyta Magnoliopsida Lamiales Lamiaceae Mentha Mentha spicata"),
    ("pansy",               "Viola tricolor",
     "Plantae Tracheophyta Magnoliopsida Malpighiales Violaceae Viola Viola tricolor"),
    ("peperomia",           "Peperomia obtusifolia",
     "Plantae Tracheophyta Magnoliopsida Piperales Piperaceae Peperomia Peperomia obtusifolia"),
    ("rose",                "Rosa hybrida",
     "Plantae Tracheophyta Magnoliopsida Rosales Rosaceae Rosa Rosa hybrida"),
    ("rosemary",            "Salvia rosmarinus",
     "Plantae Tracheophyta Magnoliopsida Lamiales Lamiaceae Salvia Salvia rosmarinus"),
    ("schefflera",          "Schefflera actinophylla",
     "Plantae Tracheophyta Magnoliopsida Apiales Araliaceae Schefflera Schefflera actinophylla"),
    ("sedum",               "Sedum spectabile",
     "Plantae Tracheophyta Magnoliopsida Saxifragales Crassulaceae Sedum Sedum spectabile"),
    ("syngonium",           "Syngonium podophyllum",
     "Plantae Tracheophyta Magnoliopsida Alismatales Araceae Syngonium Syngonium podophyllum"),
    ("tradescantia",        "Tradescantia zebrina",
     "Plantae Tracheophyta Magnoliopsida Commelinales Commelinaceae Tradescantia Tradescantia zebrina"),
    ("umbrella plant",      "Cyperus alternifolius",
     "Plantae Tracheophyta Magnoliopsida Poales Cyperaceae Cyperus Cyperus alternifolius"),
    ("venus flytrap",       "Dionaea muscipula",
     "Plantae Tracheophyta Magnoliopsida Caryophyllales Droseraceae Dionaea Dionaea muscipula"),
    ("wandering jew",       "Tradescantia fluminensis",
     "Plantae Tracheophyta Magnoliopsida Commelinales Commelinaceae Tradescantia Tradescantia fluminensis"),
    ("wisteria",            "Wisteria sinensis",
     "Plantae Tracheophyta Magnoliopsida Fabales Fabaceae Wisteria Wisteria sinensis"),
    ("yucca",               "Yucca elephantipes",
     "Plantae Tracheophyta Magnoliopsida Asparagales Asparagaceae Yucca Yucca elephantipes"),
    ("zinnia",              "Zinnia elegans",
     "Plantae Tracheophyta Magnoliopsida Asterales Asteraceae Zinnia Zinnia elegans"),
    ("basil",               "Ocimum basilicum",
     "Plantae Tracheophyta Magnoliopsida Lamiales Lamiaceae Ocimum Ocimum basilicum"),
    ("bamboo",              "Bambusoideae",
     "Plantae Tracheophyta Magnoliopsida Poales Poaceae Bambusoideae"),
    ("clivia",              "Clivia miniata",
     "Plantae Tracheophyta Magnoliopsida Asparagales Amaryllidaceae Clivia Clivia miniata"),
    ("echeveria",           "Echeveria elegans",
     "Plantae Tracheophyta Magnoliopsida Saxifragales Crassulaceae Echeveria Echeveria elegans"),
    ("haworthia",           "Haworthiopsis attenuata",
     "Plantae Tracheophyta Magnoliopsida Asparagales Asphodelaceae Haworthiopsis Haworthiopsis attenuata"),
    ("maranta",             "Maranta leuconeura",
     "Plantae Tracheophyta Magnoliopsida Zingiberales Marantaceae Maranta Maranta leuconeura"),
    ("oxalis",              "Oxalis triangularis",
     "Plantae Tracheophyta Magnoliopsida Oxalidales Oxalidaceae Oxalis Oxalis triangularis"),
    ("cactus",              "Cactaceae",
     "Plantae Tracheophyta Magnoliopsida Caryophyllales Cactaceae"),
    ("succulent",           "Crassulaceae",
     "Plantae Tracheophyta Magnoliopsida Saxifragales Crassulaceae"),
    ("boston fern",         "Nephrolepis exaltata",
     "Plantae Tracheophyta Polypodiopsida Polypodiales Nephrolepidaceae Nephrolepis Nephrolepis exaltata"),
    ("palm tree",           "Arecaceae",
     "Plantae Tracheophyta Magnoliopsida Arecales Arecaceae"),
    ("bonsai",              "Ficus retusa",
     "Plantae Tracheophyta Magnoliopsida Rosales Moraceae Ficus Ficus retusa"),
]

COMMON_NAMES     = [p[0] for p in PLANT_SPECIES]
SCIENTIFIC_NAMES = [p[1] for p in PLANT_SPECIES]
TAXON_STRINGS    = [p[2] for p in PLANT_SPECIES]
CONFIDENCE_THRESHOLD = 0.10

# Wyniki narzędzi agenta — zwykły słownik modułowy (nie st.session_state)
# st.session_state jest niedostępny wewnątrz toolów LangGraph (inny wątek/kontekst)
_agent_results: dict = {}


# ── Model loading ─────────────────────────────────────────────────────────────
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


# ── Core classification & RAG functions ──────────────────────────────────────
def classify_plant_image(image: Image.Image):
    """BioCLIP zero-shot classification — ensemble 3 formatów promptów."""
    model, preprocess, tokenizer = load_bioclip()
    template_groups = [
        TAXON_STRINGS,                                      # natywny format BioCLIP
        [f"a photo of {s}"         for s in SCIENTIFIC_NAMES],
        [f"a photo of a {s} plant" for s in COMMON_NAMES],
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
    probs    = torch.stack(all_probs).mean(dim=0)
    top5     = {COMMON_NAMES[i]: round(probs[i].item(), 4) for i in probs.topk(5).indices}
    best_idx = probs.argmax().item()
    best, sci, conf = COMMON_NAMES[best_idx], SCIENTIFIC_NAMES[best_idx], probs.max().item()
    return (best if conf >= CONFIDENCE_THRESHOLD else None), conf, top5, sci


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


def get_articles(plant_name: str):
    query = plant_name + " plant watering schedule light requirements soil"
    if TAVILY_API_KEY:
        from tavily import TavilyClient
        res = TavilyClient(api_key=TAVILY_API_KEY).search(
            query, max_results=3, include_raw_content=True
        )
        return [{"title": r.get("title", ""), "url": r.get("url", ""),
                 "text": (r.get("raw_content") or r.get("content", ""))[:5000]}
                for r in res.get("results", [])]
    docs = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=3):
            text = fetch_text(r.get("href", "")) or r.get("body", "")
            if text:
                docs.append({"title": r.get("title", ""), "url": r.get("href", ""), "text": text})
    return docs


def build_vs(plant_name: str, articles: list):
    emb     = load_embeddings()
    persist = "./chroma_db/" + plant_name.replace(" ", "_")
    if os.path.exists(persist):
        return Chroma(persist_directory=persist, embedding_function=emb)
    docs   = [Document(page_content=a["text"], metadata={"source": a["url"]})
               for a in articles if a.get("text")]
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=30
    ).split_documents(docs)
    return Chroma.from_documents(chunks, embedding=emb, persist_directory=persist)


def rag_answer(question: str, vs, plant_name: str) -> str:
    docs   = vs.as_retriever(search_kwargs={"k": 2}).invoke(question)
    ctx    = "\n\n".join(d.page_content for d in docs)
    llm    = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3,
                      max_tokens=400, groq_api_key=GROQ_API_KEY)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are PlantAI, an expert plant care assistant. "
         "Current plant: {plant_name}. Be practical, use exact numbers when available."
         "\n\nContext:\n{context}"),
        ("human", "{question}")
    ])
    return (prompt | llm | StrOutputParser()).invoke(
        {"context": ctx, "plant_name": plant_name, "question": question}
    )


# ── Agent tools ───────────────────────────────────────────────────────────────
# Trzy narzędzia jak w notatniku (sekcja 6), teraz zintegrowane z Streamlit.
# Agent (LangGraph ReAct) samodzielnie decyduje, które wywołać i w jakiej kolejności.

@tool
def classify_plant_tool(image_path: str) -> str:
    """Identifies the plant species from an image file using BioCLIP.
    Input: absolute path to the image file (provided in the task message).
    Always call this first when a new plant image is available."""
    if not image_path or not os.path.exists(image_path):
        return f"Image file not found: '{image_path}'. Check if the path is correct."
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return f"Error opening image: {e}"
    name, conf, top5, sci = classify_plant_image(image)
    if name:
        # Zapisz wyniki do słownika modułowego (dostępny z każdego wątku)
        _agent_results["plant_name"] = name
        _agent_results["sci"]        = sci
        _agent_results["conf"]       = conf
        _agent_results["top5"]       = top5
        top3 = ", ".join(f"{k} ({v:.1%})" for k, v in list(top5.items())[:3])
        return (f"Plant identified: {name.title()} ({sci})\n"
                f"Confidence: {conf:.0%}\nTop-3: {top3}")
    else:
        return f"Could not identify plant (confidence too low: {conf:.0%})."


@tool
def search_plant_info_tool(plant_name: str) -> str:
    """Searches the internet for plant care articles and builds a ChromaDB vector knowledge base.
    Always call this after identifying the plant and before answering care questions.
    Input: common plant name (e.g. 'monstera deliciosa')."""
    arts = get_articles(plant_name)
    if not arts:
        return f"No articles found for '{plant_name}'. Try a more common name."
    vs = build_vs(plant_name, arts)
    _agent_results["vectorstore"] = vs
    _agent_results["arts_count"]  = len(arts)
    return (f"Knowledge base built for '{plant_name}': "
            f"{len(arts)} articles, chunked and indexed in ChromaDB.")


@tool
def answer_care_question_tool(question: str) -> str:
    """Answers plant care questions using the RAG knowledge base (ChromaDB + LLM).
    Use for questions about watering, light, soil, fertilizing, propagation, common problems.
    Requires the knowledge base to be built first via search_plant_info_tool."""
    # 1. Sprawdź słownik modułowy (zawsze dostępny, niezależnie od wątku)
    vs         = _agent_results.get("vectorstore")
    plant_name = _agent_results.get("plant_name", "the plant") or "the plant"
    # 2. Fallback: st.session_state (działa gdy agent w tym samym wątku)
    if vs is None:
        try:
            idx      = st.session_state.get("active_idx")
            sessions = st.session_state.get("sessions", [])
            if idx is not None and 0 <= idx < len(sessions):
                vs         = sessions[idx].get("vectorstore")
                plant_name = sessions[idx].get("plant_name", "the plant") or "the plant"
        except Exception:
            pass
    if vs is None:
        return "No knowledge base available yet. Please identify a plant first."
    return rag_answer(question, vs, plant_name)


# ── Agent factory ─────────────────────────────────────────────────────────────
def make_agent():
    """Creates a LangGraph ReAct agent with three plant tools.

    Model: llama-3.3-70b-versatile — znacznie lepszy od 8b w wywoływaniu narzędzi,
    wciąż dostępny na darmowym tierze Groq (do 1000 req/dzień).
    """
    # llama-3.3-70b-versatile: niezawodne tool-calling, darmowy Groq free tier
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=GROQ_API_KEY)
    system = SystemMessage(content="""You are PlantAI — an autonomous AI agent for plant identification and care.

You have exactly three tools. Use them proactively — do NOT answer from memory alone:
  1. classify_plant_tool   — ALWAYS call this first when a new image is uploaded
  2. search_plant_info_tool — ALWAYS call this right after classification (input: plant common name)
  3. answer_care_question_tool — call this for any care question (watering, light, soil, etc.)

MANDATORY workflow on image upload (follow strictly, in order):
  Step 1 → classify_plant_tool()
  Step 2 → search_plant_info_tool(plant_name from step 1)
  Step 3 → respond warmly with the plant name and an invitation to ask questions

For care questions in chat: call answer_care_question_tool(question).
Never skip tool calls. Never answer care questions from memory.""")
    return create_react_agent(llm, [classify_plant_tool, search_plant_info_tool,
                                     answer_care_question_tool], prompt=system)


def run_agent_turn(user_msg: str, sess: dict) -> str:
    """
    Runs one conversational turn of the ReAct agent.
    Maintains full message history in sess['agent_messages'] for multi-turn context.
    """
    agent   = make_agent()
    history = list(sess.get("agent_messages", [])) + [HumanMessage(content=user_msg)]
    try:
        result = agent.invoke({"messages": history}, config={"recursion_limit": 10})
        sess["agent_messages"] = list(result["messages"])
        # Return the last AI text (skip messages that only have tool_calls)
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content
        return result["messages"][-1].content
    except Exception as e:
        return f"Agent error: {e}"


# ── Streamlit session state init ──────────────────────────────────────────────
if "sessions"   not in st.session_state: st.session_state.sessions   = []
if "active_idx" not in st.session_state: st.session_state.active_idx = None


def active_session():
    idx = st.session_state.active_idx
    if idx is not None and 0 <= idx < len(st.session_state.sessions):
        return st.session_state.sessions[idx]
    return None


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
            # Zapisz obraz do pliku tymczasowego
            # Ścieżka przekazywana jako argument narzędzia (nie przez session_state)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                img_path = tmp.name
            image.save(img_path)   # zapisz po zamknięciu tmp (unika blokady pliku)
            st.session_state["_pending_image_path"] = img_path  # backup dla chatu

            # Wyczyść poprzednie wyniki narzędzi
            _agent_results.clear()

            # Utwórz slot sesji — agent go wypełni
            new_sess: dict = {
                "id":              datetime.now().strftime("%H%M%S%f"),
                "plant_name":      None,
                "scientific_name": None,
                "confidence":      None,
                "messages":        [],
                "agent_messages":  [],
                "vectorstore":     None,
                "timestamp":       datetime.now().strftime("%H:%M"),
                "top5":            None,
            }
            st.session_state.sessions.insert(0, new_sess)
            st.session_state.active_idx = 0

            # ── Agent autonomicznie: klasyfikuje → szuka → odpowiada ──
            with st.spinner("🤖 Agent PlantAI pracuje…"):
                trigger = (
                    f"A plant image has been saved to: '{img_path}'. "
                    f"Step 1: call classify_plant_tool with image_path='{img_path}'. "
                    "Step 2: call search_plant_info_tool(plant_name) with the result. "
                    "Step 3: greet the user and introduce the identified plant."
                )
                ai_intro = run_agent_turn(trigger, new_sess)

            # Pobierz wyniki z modułowego słownika (ustawionego przez narzędzia agenta)
            if _agent_results.get("plant_name"):
                new_sess["plant_name"]      = _agent_results["plant_name"]
                new_sess["scientific_name"] = _agent_results["sci"]
                new_sess["confidence"]      = _agent_results["conf"]
                new_sess["top5"]            = _agent_results["top5"]
                new_sess["vectorstore"]     = _agent_results.get("vectorstore")
                new_sess["messages"].append({"role": "assistant", "content": ai_intro})
                st.success(
                    f"**{new_sess['plant_name'].title()}** · "
                    f"*{new_sess['scientific_name']}* ({new_sess['confidence']:.0%})"
                )
            else:
                new_sess["messages"].append({"role": "assistant", "content": ai_intro})
                st.error("Nie rozpoznano rośliny. Spróbuj wyraźniejszego zdjęcia.")

            st.rerun()

    # ── Session history ──
    if st.session_state.sessions:
        st.divider()
        st.markdown("**📋 Historia czatów**")
        for i, sess in enumerate(st.session_state.sessions):
            is_active = (i == st.session_state.active_idx)
            plant     = (sess.get("plant_name") or "Identyfikacja…").title()
            n_msg     = sum(1 for m in sess["messages"] if m["role"] == "user")
            label     = (
                f"{'🟢' if is_active else '⚪'} **{plant}**\n"
                f"{sess['timestamp']}"
                + (f" · {n_msg} pytań" if n_msg else "")
            )
            col_btn, col_del = st.columns([5, 1])
            with col_btn:
                if st.button(label, key=f"sess_{sess['id']}", use_container_width=True,
                             type="primary" if is_active else "secondary"):
                    st.session_state.active_idx = i
                    st.rerun()
            with col_del:
                if st.button("🗑", key=f"del_{sess['id']}", help="Usuń sesję"):
                    st.session_state.sessions.pop(i)
                    n = len(st.session_state.sessions)
                    if n == 0:
                        st.session_state.active_idx = None
                    elif st.session_state.active_idx is not None and st.session_state.active_idx >= n:
                        st.session_state.active_idx = n - 1
                    elif st.session_state.active_idx is not None and st.session_state.active_idx > i:
                        st.session_state.active_idx -= 1
                    st.rerun()


# ── Main chat area ─────────────────────────────────────────────────────────────
sess = active_session()

if sess is None:
    st.title("🌿 PlantAI")
    st.info("← Prześlij zdjęcie rośliny w panelu bocznym, aby rozpocząć.")
else:
    plant_disp = (sess.get("plant_name") or "Identyfikacja w toku").title()
    sci_disp   = sess.get("scientific_name", "")
    conf_disp  = sess.get("confidence")

    st.title(f"🌿 {plant_disp}")
    if sci_disp:
        st.caption(f"*{sci_disp}*" + (f" · pewność: {conf_disp:.0%}" if conf_disp else ""))
    st.divider()

    # Show classification results for new (empty) session
    if not sess["messages"] and sess.get("top5"):
        with st.expander("🔬 Top-5 BioCLIP", expanded=True):
            for sp, pr in sess["top5"].items():
                st.progress(pr, text=f"{sp.title()} ({pr:.1%})")

    # Chat history
    for msg in sess["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input → agent
    if q := st.chat_input("Zapytaj o pielęgnację..."):
        sess["messages"].append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            if not sess.get("vectorstore"):
                ans = "Najpierw prześlij zdjęcie i kliknij Rozpoznaj roślinę."
            else:
                with st.spinner("🤖 Agent myśli…"):
                    ans = run_agent_turn(q, sess)
            st.markdown(ans)
            sess["messages"].append({"role": "assistant", "content": ans})
