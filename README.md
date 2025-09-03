# Emotion-Aware AAC with Fine-Tuned LLMs and RAG

**Course:** AAC for Societal Good (NLP DUO)  
**Authors:** Sai Sri Kolanu, B. Meghana Chowdary  
**School of Engineering and Applied Sciences, University at Buffalo**  

---

## 📖 Overview
This project enhances **Augmentative and Alternative Communication (AAC)** systems for individuals with speech impairments by integrating **Large Language Models (LLMs)** with **Retrieval-Augmented Generation (RAG)**.  
The system generates **empathetic, expressive, and personalized responses**, addressing limitations of templated AAC systems.

---

## 📊 Datasets
- **EmpatheticDialogues** – ~25k conversations grounded in emotional situations (32 emotions).  
- **GoEmotions** – 58k+ Reddit comments annotated with 27 fine-grained emotions + neutral.  

---

## ⚙️ Methodology
1. **Fine-Tuned LLM**
   - Base model: **T5-Small (Hugging Face)**
   - Fine-tuned with **LoRA (Low-Rank Adaptation)** on hybrid emotional datasets.
   - Incorporates emotion tokens `[emotion]` for contextual response generation.

2. **Retrieval-Augmented Generation (RAG)**
   - FAISS + `all-MiniLM-L6-v2` embeddings.
   - Retrieves user-specific narratives to ground responses.
   - Reduces hallucinations and improves personalization.

3. **Interactive Streamlit Interface**
   - Emotion detection from user input.  
   - Style Controls: **formality, tone, response length, personality traits, greeting prefix**.  
   - Multiple generated responses with user selection.  
   - Simulates real AAC usage.

---

## 📈 Results
- **Automatic Evaluation** (500 samples):  
  - BLEU: 0.0081, ROUGE-1: 0.1183, ROUGE-L: 0.1086, METEOR: 0.0918, BERTScore: 0.8552  
- **Human Evaluation** (50 samples):  
  - Relevance: 1.84, Sincerity: 2.56, Fluency: 3.24, Personalization: 1.26  
- **Case Study**: LSTM detected “fear” in input “I’m scared” and generated warm, empathetic options.  

---

## 🛠️ Tech Stack
- **Python 3.x**
- **Hugging Face Transformers**
- **SentenceTransformers**
- **FAISS**
- **Streamlit**
- **LoRA (PEFT)**

---

## ▶️ How to Run
1. Clone repo:
   ```bash
   git clone https://github.com/saisrikolanu/Emotion-Aware-AAC-with-Fine-Tuned-LLMs-and-RAG.git
   cd Emotion-Aware-AAC-with-Fine-Tuned-LLMs-and-RAG
