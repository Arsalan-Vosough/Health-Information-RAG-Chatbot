"""
app.py
------
Flask chatbot that loads a pre-built FAISS vectorstore from disk.
Run embed.py first to create the index, then start this app.

Usage:
    python app.py
"""

import os
import json
from flask import Flask, render_template, request, redirect, url_for, jsonify
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langdetect import detect


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")  # set via environment variable or .env

# HuggingFace model used in embed.py — must match exactly so vectors are compatible
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Path to the FAISS index saved by embed.py
FAISS_INDEX_PATH = r"C:\pooshe kaar\Health-Information-RAG-Chatbot\sources\faiss_index"

# Temperature for the main chat LLM (0.0 = strict/factual, 1.0 = creative/flexible)
TEMPERATURE = 0.7

# ---------------------------------------------------------------------------
# App & global state
# ---------------------------------------------------------------------------
app = Flask(__name__)

vectorstore = None
retriever = None
chain = None
chat_history = []
lcel_history = []  # list of HumanMessage/AIMessage for the LCEL chain
last_user_question = ""
question_count = 0


# ---------------------------------------------------------------------------
# Load pre-built vectorstore from disk
# ---------------------------------------------------------------------------
def load_vectorstore():
    """Load the FAISS index that was saved by embed.py."""
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at '{FAISS_INDEX_PATH}'.\n"
            "Please run embed.py first to build and save the index."
        )
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"Loading FAISS index from: {FAISS_INDEX_PATH}")
    vs = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print("[OK] Vectorstore loaded successfully!")
    return vs


# ---------------------------------------------------------------------------
# Conversation chain (LCEL)
# ---------------------------------------------------------------------------
def get_conversation_chain(vs):
    llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=TEMPERATURE)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Olet avulias assistentti. Vastaat vain annettujen lähdemateriaalien perusteella.

SAANNOT:
- Jos kayttaja kysyy vertailua (esim. "difference between", "compare", "vs"), vastaa vertaillen
- Jos kysytaan vain yhdesta aiheesta, kerro vain siita - ala mainitse muita aiheita
- Jos asiayhteys sisaltaa tietoa kysytystä aiheesta/aiheista, vastaa sen perusteella
- Jos kysymys on englanniksi, vastaa englanniksi
- Jos kysymys on suomeksi, vastaa suomeksi
- Jos asiayhteys ei sisalla tietoa kysytystä aiheesta, sano: "Tietojeni perusteella en osaa vastata tahan kysymykseen."

Tassa asiayhteys:
{context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])

    ret = vs.as_retriever(search_type="similarity", search_kwargs={"k": 15})
    lcel_chain = prompt | llm | StrOutputParser()
    return ret, lcel_chain


# ---------------------------------------------------------------------------
# Translation helpers
# ---------------------------------------------------------------------------
def translate_to_finnish(question: str) -> str:
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    return llm.invoke(f"Translate the text below to Finnish. Return ONLY the translated text, no explanations:\n\n{question}").content.strip()


def translate_to_english(answer: str) -> str:
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    return llm.invoke(f"Translate the text below to English. Return ONLY the translated text, no explanations:\n\n{answer}").content.strip()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["GET", "POST"])
def chat():
    global retriever, chain, chat_history, lcel_history, last_user_question

    if request.method == "POST":
        user_question = request.form["user_question"]
        last_user_question = user_question
        lang = detect(user_question)

        # Append short follow-up questions to the previous question for context
        if len(user_question.strip().split()) < 3:
            last_user_msg = next(
                (m["content"] for m in reversed(chat_history) if m["type"] == "user"), ""
            )
            user_question = f"{last_user_msg}\n{user_question}"

        used_question = translate_to_finnish(user_question) if lang == "en" else user_question

        # Retrieve relevant docs
        sources = retriever.invoke(used_question)
        context = "\n\n".join(doc.page_content for doc in sources)
        source_urls = set()
        for doc in sources:
            if "source" in doc.metadata and doc.metadata["source"] != "PDF":
                clean_url = doc.metadata["source"].split("#")[0]
                source_urls.add(clean_url)

        if not sources or all(not doc.page_content.strip() for doc in sources):
            answer = "Tietojeni perusteella en osaa vastata tahan kysymykseen."
        else:
            answer = chain.invoke({
                "question": used_question,
                "context": context,
                "chat_history": lcel_history,
            })

        if lang == "en":
            answer = translate_to_english(answer)
            if source_urls:
                joined = "<br>".join(
                    f'<a href="{url}" target="_blank">Source: {url}</a>' for url in source_urls
                )
                answer += f"<br><br>{joined}"
        else:
            if source_urls:
                joined = "<br>".join(
                    f'<a href="{url}" target="_blank">Lahde: {url}</a>' for url in source_urls
                )
                answer += f"<br><br>{joined}"

        lcel_history.append(HumanMessage(content=used_question))
        lcel_history.append(AIMessage(content=answer))
        chat_history.append({"type": "user", "content": user_question})
        chat_history.append({"type": "assistant", "content": answer})

    return render_template("chat.html", chat_history=chat_history)


@app.route("/process_question", methods=["POST"])
def process_question():
    global retriever, chain, chat_history, lcel_history, last_user_question

    try:
        data = request.get_json()
        user_question = data.get("question", "").strip()
        if not user_question:
            return jsonify({"error": "Empty question"}), 400

        last_user_question = user_question
        lang = detect(user_question)

        if len(user_question.split()) < 3:
            last_user_msg = next(
                (m["content"] for m in reversed(chat_history) if m["type"] == "user"), ""
            )
            user_question = f"{last_user_msg}\n{user_question}"

        used_question = translate_to_finnish(user_question) if lang == "en" else user_question

        sources = retriever.invoke(used_question)
        context = "\n\n".join(doc.page_content for doc in sources)
        source_urls = set()
        for doc in sources:
            if "source" in doc.metadata and doc.metadata["source"] != "PDF":
                clean_url = doc.metadata["source"].split("#")[0]
                source_urls.add(clean_url)

        if not sources or all(not doc.page_content.strip() for doc in sources):
            answer = "Tietojeni perusteella en osaa vastata tahan kysymykseen."
        else:
            answer = chain.invoke({
                "question": used_question,
                "context": context,
                "chat_history": lcel_history,
            })

        if lang == "en":
            answer = translate_to_english(answer)

        lcel_history.append(HumanMessage(content=used_question))
        lcel_history.append(AIMessage(content=answer))
        chat_history.append({"type": "user", "content": user_question})
        chat_history.append({"type": "assistant", "content": answer})

        return jsonify({
            "answer": answer,
            "sources": list(source_urls),
            "language": lang,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/search_duodecim", methods=["POST"])
def search_duodecim_route():
    global chat_history

    if not DUODECIM_AVAILABLE:
        chat_history.append({"type": "assistant", "content": "<strong>Duodecim AI:</strong> duodecim_scraper module not available."})
        return redirect(url_for("chat"))

    last_question = next(
        (m["content"] for m in reversed(chat_history) if m["type"] == "user"), None
    )
    if not last_question:
        return redirect(url_for("chat"))

    try:
        result = search_duodecim(last_question, headless=True)
        duodecim_answer = f"<strong>Duodecim AI:</strong><br>{result['answer']}"

        unique_links = {}
        for src in result["sources"]:
            key = (src["title"], src["url"])
            if key not in unique_links:
                unique_links[key] = (
                    f'<a href="{src["url"]}" target="_blank">{src["title"]}: {src["url"]}</a>'
                )

        if unique_links:
            duodecim_answer += "<br><br><strong>SOURCES:</strong><br>" + "<br>".join(
                unique_links.values()
            )

        chat_history.append({"type": "assistant", "content": duodecim_answer})

    except Exception as e:
        chat_history.append(
            {"type": "assistant", "content": f"<strong>Duodecim AI:</strong> Error: {e}"}
        )

    return redirect(url_for("chat"))


@app.route("/clear_chat")
def clear_chat():
    global chat_history, lcel_history, question_count
    chat_history = []
    lcel_history = []
    question_count = 0
    return redirect(url_for("chat"))


@app.route("/reset_questions")
def reset_questions():
    global question_count
    question_count = 0
    return redirect(url_for("chat"))


# ---------------------------------------------------------------------------
# Startup — load vectorstore once, no embedding
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        vectorstore = load_vectorstore()
        retriever, chain = get_conversation_chain(vectorstore)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}\n")
        print("Starting the app anyway — visit /load to retry after running embed.py.")

    app.run(debug=False)
