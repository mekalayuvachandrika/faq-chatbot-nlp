import streamlit as st
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI FAQ Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>

/* Main Background */
.stApp {
    background-image: url('https://images.unsplash.com/photo-1516321318423-f06f85e504b3');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Dark Overlay */
.main::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.65);
    z-index: -1;
}

/* Main Container */
.block-container {
    padding-top: 2rem;
}

/* Title */
.title {
    text-align: center;
    font-size: 55px;
    font-weight: bold;
    color: white;
    margin-bottom: 10px;
    text-shadow: 2px 2px 10px black;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 20px;
    color: #e2e8f0;
    margin-bottom: 30px;
}

/* Chat Container */
.chat-container {
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(12px);
    border-radius: 25px;
    padding: 30px;
    box-shadow: 0px 8px 32px rgba(0,0,0,0.4);
}

/* User Message */
.user-msg {
    background: linear-gradient(to right, #2563eb, #3b82f6);
    color: white;
    padding: 14px;
    border-radius: 15px;
    margin: 12px 0;
    text-align: right;
    font-size: 17px;
}

/* Bot Message */
.bot-msg {
    background: linear-gradient(to right, #1e293b, #334155);
    color: white;
    padding: 14px;
    border-radius: 15px;
    margin: 12px 0;
    font-size: 17px;
}

/* Input Box */
.stTextInput > div > div > input {
    background-color: rgba(15, 23, 42, 0.85);
    color: white;
    border-radius: 12px;
    border: 1px solid #64748b;
    padding: 12px;
    font-size: 16px;
}

/* Buttons */
.stButton > button {
    width: 100%;
    background: linear-gradient(to right, #06b6d4, #3b82f6);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px;
    font-size: 18px;
    font-weight: bold;
    transition: 0.3s ease;
}

.stButton > button:hover {
    transform: scale(1.02);
    opacity: 0.95;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95);
}

.sidebar-title {
    color: white;
    text-align: center;
    font-size: 24px;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown('<div class="sidebar-title">🤖 AI FAQ Bot</div>', unsafe_allow_html=True)

    st.image(
        "https://cdn-icons-png.flaticon.com/512/4712/4712027.png",
        width=180
    )

    st.markdown("---")

    st.markdown("### 📌 Features")
    st.write("✅ NLP-based FAQ matching")
    st.write("✅ Cosine Similarity")
    st.write("✅ Smart Responses")
    st.write("✅ Beautiful Modern UI")

    st.markdown("---")
    st.write("Made with ❤️ using Streamlit")

# ================= TITLE =================
st.markdown('<div class="title">🤖 Smart FAQ Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask questions and get instant answers powered by NLP</div>', unsafe_allow_html=True)

# ================= LOAD FAQ DATA =================
with open("faq.json", "r") as f:
    faqs = json.load(f)

# ================= NLP PREPROCESSING =================
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

questions = [preprocess(faq["question"]) for faq in faqs]

# ================= TF-IDF =================
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words='english'
)

faq_vectors = vectorizer.fit_transform(questions)

# ================= CHATBOT FUNCTION =================
def get_answer(user_query):
    user_query = preprocess(user_query)

    user_vec = vectorizer.transform([user_query])

    similarity = cosine_similarity(user_vec, faq_vectors)

    index = similarity.argmax()
    score = similarity[0][index]

    if score < 0.3:
        return "❌ Sorry, I couldn't find a matching answer."

    return faqs[index]["answer"]

# ================= SESSION STATE =================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= CHAT SECTION =================
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    user_input = st.text_input("💬 Type your question")

    col1, col2 = st.columns([4, 1])

    with col1:
        send = st.button("Send 🚀")

    with col2:
        clear = st.button("🧹 Clear")

    if send:
        if user_input:
            answer = get_answer(user_input)

            st.session_state.messages.append(("user", user_input))
            st.session_state.messages.append(("bot", answer))

    if clear:
        st.session_state.messages = []
        st.rerun()

    # Display chat
    for sender, message in st.session_state.messages:
        if sender == "user":
            st.markdown(
                f'<div class="user-msg">🧑 {message}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="bot-msg">🤖 {message}</div>',
                unsafe_allow_html=True
            )

    st.markdown('</div>', unsafe_allow_html=True)
