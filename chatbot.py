import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load FAQ file
with open("faq.json", "r") as f:
    faqs = json.load(f)

# Preprocess function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

# Prepare questions
questions = [preprocess(faq["question"]) for faq in faqs]

# Convert to vectors
vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
faq_vectors = vectorizer.fit_transform(questions)

# Matching function
def get_answer(user_query):
    user_query = preprocess(user_query)
    user_vec = vectorizer.transform([user_query])

    similarity = cosine_similarity(user_vec, faq_vectors)

    index = similarity.argmax()
    score = similarity[0][index]
    print("DEBUG scores:", similarity)
    print("DEBUG best index:", index)
    print("DEBUG score:", score)
    if score < 0.5:
        return "Sorry, I don't understand your question."

    return faqs[index]["answer"]

# Chat loop
print("FAQ Chatbot is running... Type 'exit' to stop")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    response = get_answer(user_input)
    print("Bot:", response)