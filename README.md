# **Rumi Knowledge Base**

## **Overview**
This project is a custom-built knowledge base inspired by *The Essential Rumi*, one of the most cherished collections of Rumi's poetry. The knowledge base allows users to ask deep, philosophical, or spiritual questions and receive meaningful responses drawn directly from Rumi's verses. The system integrates state-of-the-art retrieval and language models to deliver thoughtful and contextually accurate answers.

## **Motivation**
While large language models like GPT are powerful, they are restricted from directly quoting certain texts due to ownership and copyright limitations. This project overcomes these barriers by creating a specialized knowledge base from scratch using one of my favorite books, *The Essential Rumi*. This ensures the poetry and its wisdom are faithfully represented and accessible to users.

## **Features**
- **Question-Answering System**: Users can ask questions like "How can I find peace when life feels overwhelming?" and receive Rumi-inspired responses.
- **Poetry Retrieval**: Retrieves the most relevant poem excerpts based on user queries.
- **Summary & Explanation**: Provides a brief summary and interpretation of the retrieved verses to help users connect with their meaning.
- **Efficient Retrieval**: Uses FAISS for fast and scalable similarity search.
- **Language Generation**: Combines embeddings from SentenceTransformers and language generation from the LLaMA model to craft complete answers.

## **How It Works**
1. **Load Poems**: The system parses a JSON file containing poems from *The Essential Rumi* and flattens them into documents.
2. **Build Embeddings**: Sentences are embedded using a pre-trained SentenceTransformer model for similarity comparison.
3. **Retrieve Relevant Poems**:
   - **Brute Force**: Computes cosine similarity for every query-poem pair.
   - **FAISS Index**: Enables faster, approximate nearest neighbor search for large datasets.
4. **Generate Answers**: The LLaMA language model combines retrieved poem excerpts with meaningful explanations to craft human-like responses.
5. **Satisfaction Check**: Ensures responses include all necessary sections (**Poem Excerpt**, **Summary**, and **Explanation**).

## **Examples**
Here are a few examples of the system in action:

### **Question:** How can I find peace when life feels overwhelming?
**Poem Excerpt:**
> It's a habit of yours to walk slowly.  
> You hold a grudge for years.  
> With such heaviness, how can you be modest?  
> ...  
> You are so weak. Give up to grace.  
> The ocean takes care of each wave till it gets to shore.

**Summary:**  
Rumi's poem encourages us to let go of burdens and attachments, turning to grace for support. Peace arises from recognizing our limitations and surrendering to a higher power.

**Explanation:**  
True peace is found not in control but in release—letting go of the weight we carry and trusting the ocean of grace to carry us safely to shore.

---

### **Question:** What does it mean to let go of control and surrender?
**Poem Excerpt:**
> Tell her, one surrendering bow is sweeter  
> than a hundred empires, is itself a kingdom.  
> Be dizzy and wandering like Ibrahim,  
> who suddenly left everything.

**Summary:**  
Rumi uses surrender as a metaphor for transformation, illustrating that letting go of control opens us to greater understanding and alignment with life’s flow.

**Explanation:**  
By surrendering attachments and embracing the unknown, we create space for profound personal growth and alignment with the wisdom of the universe.

---

## **Dependencies**
- **Python**: 3.8+
- **Libraries**:
  - `torch`
  - `transformers`
  - `sentence-transformers`
  - `faiss`
  - `numpy`
  - `json`

**Future Enhancements:**  
- Expand the knowledge base with additional poems and works by Rumi and encorporate more Sufi poets.
- Improve the system’s ability to explain verses with deeper and more nuanced interpretations.
- Create a user-friendly web interface for easier interaction and accessibility.

**Acknowledgments:**  
- Coleman Barks for his incredible translations of Rumi’s poetry in The Essential Rumi.
- OpenAI, Hugging Face, and the creators of FAISS for enabling cutting-edge AI tools.
