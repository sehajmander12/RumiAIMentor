import json
import time
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss

#########################################
# 1) LOAD & FLATTEN THE JSON
#########################################
def load_rumi_chapters(json_path):
    """
    Expects a JSON with structure like:
    [
      {
        "chapter_title": "...",
        "chapter_intro": "...",
        "poems": [
          {"title": "...", "poem": "..."},
          ...
        ]
      },
      ...
    ]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        chapters_data = json.load(f)

    # Flatten into docs (only poems, skipping chapter_intro)
    docs = []
    for chapter in chapters_data:
        for p in chapter["poems"]:
            docs.append({
                "type": "poem",
                "chapter_title": chapter["chapter_title"],
                "title": p["title"],
                "text": p["poem"]
            })
    return docs


#########################################
# 2) EMBEDDING MODEL & LLAMA SETUP
#########################################
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
modelLlama = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", device_map="auto"
)

#########################################
# 3) BUILD EMBEDDINGS
#########################################
def build_embeddings(docs):
    texts = [d["text"] for d in docs]
    doc_embs = embed_model.encode(texts, convert_to_tensor=True)
    return doc_embs


#########################################
# 4) BRUTE FORCE COSINE RETRIEVAL
#########################################
def retrieve_brute_force(query, docs, doc_embs, top_k=1):
    # Embed query
    q_emb = embed_model.encode(query, convert_to_tensor=True)
    # Cosine similarity
    sims = util.cos_sim(q_emb, doc_embs)  # shape: [1, #docs]
    top_results = torch.topk(sims, k=top_k, dim=1)
    
    relevant_docs = []
    for idx in top_results.indices[0]:
        relevant_docs.append(docs[idx.item()])
    return relevant_docs


#########################################
# 5) FAISS RETRIEVAL
#########################################
def build_faiss_index(doc_embs):
    """
    Build a FlatIP (Inner Product) index for approximate nearest neighbor.
    """
    emb_np = doc_embs.cpu().numpy().astype(np.float32)
    dim = emb_np.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb_np)
    return index

def retrieve_faiss(query, docs, faiss_index, top_k=1):
    q_emb = embed_model.encode(query, convert_to_tensor=False)  # returns numpy array
    q_emb_2d = np.expand_dims(q_emb, axis=0).astype(np.float32)
    scores, indices = faiss_index.search(q_emb_2d, k=top_k)
    
    relevant_docs = []
    for idx in indices[0]:
        relevant_docs.append(docs[idx])
    return relevant_docs


#########################################
# 6) MEASURE RETRIEVAL TIME
#########################################
def measure_retrieval_time(queries, retrieval_method, docs, top_k=1, n_runs=1):
    times = []
    for _ in range(n_runs):
        for q in queries:
            start_t = time.time()
            _ = retrieval_method(q, docs, top_k=top_k)
            end_t = time.time()
            times.append(end_t - start_t)
    avg_time = sum(times) / len(times)
    return avg_time


#########################################
# 7) CHECK IF LLM RESPONSE IS SATISFACTORY
#########################################
def is_satisfactory(response_text):
    """
    Check if the response includes all required sections:
    - Poem Excerpt
    - Brief Summary
    - Explanation
    Also, ensure the response meets a minimum word count.
    """
    required_sections = ["**Poem Excerpt:**", "**Brief Summary:**", "**Explanation:**"]
    if not response_text:
        return False
    if all(section in response_text for section in required_sections):
        if len(response_text.split()) >= 100:  # Adjust as needed
            return True
    return False



#########################################
# 8) GENERATE LLAMA RESPONSE
#########################################
def generate_llama_response(doc, user_query):
    """
    Generates a response based on the user's query and a relevant document.
    """
    chapter_title = doc.get("chapter_title", "")
    poem_title = doc.get("title", "No Title")
    poem_text = doc["text"]
    

    # =========================
    # FEW-SHOT EXAMPLES
    # =========================

    few_shot_examples = """
    ### Example 1

    **Question:** How can I discover who I truly am?

    **Answer:**

    **Poem Excerpt:**
    Not Christian or Jew or Muslim, not Hindu,
    Buddhist, sufi, or zen. Not any religion
    or cultural system. I am not from the East
    or the West, not out of the ocean or up
    from the ground, not natural or ethereal, not
    composed of elements at all. I do not exist,
    am not an entity in this world or the next,
    did not descend from Adam and Eve or any
    origin story. My place is placeless, a trace
    of the traceless. Neither body or soul.
    I belong to the beloved, have seen the two
    worlds as one and that one call to and know,
    first, last, outer, inner, only that
    breath breathing human being.

    **Brief Summary:**
    Rumi dissolves all labels of identity—religion, culture, nationality—to emphasize a higher, universal state of being. He describes a "placeless" place where the soul transcends definitions and belongs solely to Love (the Beloved).

    **Explanation:**
    Rumi’s verses suggest that true self-discovery happens when we let go of rigid identities and labels. By recognizing we are “not an entity” limited by culture, religion, or form, we connect to a boundless presence within—our true self.

    ---

    ### Example 2

    **Question:** How can I find joy in the small things in life?

    **Answer:**

    **Poem Excerpt:**
    Today, like every other day, we wake up empty
    and frightened. Don't open the door to the study
    and begin reading. Take down a musical instrument.
    Let the beauty we love be what we do.
    There are hundreds of ways to kneel and kiss the ground.

    **Brief Summary:**
    Rumi addresses the daily anxieties that weigh us down, encouraging us to engage with beauty and creativity instead of routine or worry. He reminds us there are countless ways to connect with the divine and find reverence in life’s simplicity.

    **Explanation:**
    This poem encourages embracing the present and finding delight in small, creative acts—like picking up a musical instrument or simply showing gratitude (“kneel and kiss the ground”). By seeking beauty in our daily actions, we uncover the joy hidden in everyday moments.

    ---

    ### Example 3

    **Question:** How do I quiet my mind to hear my inner voice?

    **Answer:**

    **Poem Excerpt:**
    There is a way between voice and presence
    where information flows.
    In disciplined silence it opens.
    With wandering talk it closes.

    **Brief Summary:**
    Rumi captures the essence of mindfulness and inner stillness. He suggests there is an inner stream of wisdom accessible only through silence and discipline, cautioning that idle chatter or mental noise drowns it out.

    **Explanation:**
    This verse shows that inner guidance emerges when we cultivate silence and attentiveness. By taking moments of quiet reflection—free of mental chatter—we can tune into our deeper voice, the wisdom that resides within.
    """
    # =========================
    # TEMPLATE PROMPT
    # =========================

    prompt = f"""You are a wise mentor who knows Rumi's poetry.

    Below are some examples of how to respond to user questions using Rumi's verses:

    {few_shot_examples}

    Now, here is the new inquiry:

    **Relevant Poem:**
    **Chapter:** {chapter_title}
    **Title:** {poem_title}

    {poem_text}

    **Question:** "{user_query}"

    **Answer:**
    """




    inputs = tokenizer(prompt, return_tensors="pt").to(modelLlama.device)
    outputs = modelLlama.generate(
        **inputs,
        max_length=2048,
        temperature=0.7
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Optional Cleanup of Prompt Leakage
    answer = answer.replace(prompt.strip(), "").strip()
    
    return answer



#########################################
# 9) ANSWER USER QUESTION (TOP_K=3)
#########################################
def answer_user_question(query, docs, doc_embs, top_k=3, retrieval_fn=None):
    """
    1. Retrieve top_k docs
    2. Generate an LLM response for each doc
    3. Return the first 'satisfactory' answer
    4. If none are satisfactory, fallback
    """
    if retrieval_fn is None:
        # default to brute force if not specified
        retrieval_fn = retrieve_brute_force

    candidates = retrieval_fn(query, docs, top_k=top_k)
    if not candidates:
        return "Sorry, I couldn't find any poems related to your query."

    # poem_text = candidates[0]["text"]
    # return(poem_text)

    for doc in candidates:
        answer_text = generate_llama_response(doc, query)
        if is_satisfactory(answer_text):
            return answer_text

    return (
        "I retrieved some poems, but none seemed suitable to answer your question. "
        "Try rephrasing or ask another question."
    )


#########################################
# MAIN DEMO
#########################################
def main():
    # 1) Load and flatten docs
    rumi_docs = load_rumi_chapters("filtered_rumi_poems.json")

    # 2) Build embeddings for brute force
    doc_embeddings = build_embeddings(rumi_docs)

    # 3) Build FAISS index
    faiss_index = build_faiss_index(doc_embeddings)

    # Some example queries
    queries = [
    "How can I find peace when life feels overwhelming?",
    "What does it mean to let go of control and surrender?",
    "How can I turn pain into growth?",
    ]

    print("Measuring FAISS retrieval time on these queries:")
    def faiss_method(q, docs, top_k=1):
        return retrieve_faiss(q, docs, faiss_index, top_k=top_k)
    faiss_time = measure_retrieval_time(queries, faiss_method, rumi_docs, top_k=1, n_runs=1)
    print(f"FAISS Average Time Per Query: {faiss_time:.4f} seconds\n")

    # Now answer each query using FAISS, top_k=3
    for user_query in queries:
        print(f"QUESTION:\n{user_query}\n")
        response = answer_user_question(
            user_query,
            rumi_docs,
            doc_embeddings,
            top_k=3,
            retrieval_fn=lambda q, docs, top_k: retrieve_faiss(q, docs, faiss_index, top_k)
        )
        print(f"ANSWER:\n{response}\n{'-'*60}\n")

if __name__ == "__main__":
    main()
