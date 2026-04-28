# Calder AI/ML Internship - FAQ Chatbot System

## Overview
This repository contains a robust, interactive FAQ Chatbot developed for the Calder AI/ML Internship assignment. The system processes user queries and retrieves the most contextually relevant answers from a predefined knowledge base using Natural Language Processing (NLP) techniques.

## Features
- **Semantic Search Engine:** Utilizes **TF-IDF** (Term Frequency-Inverse Document Frequency) and **Cosine Similarity** to match user queries with the most relevant FAQ, going beyond simple keyword matching.
- **Graceful Fallback:** Implements a confidence threshold. If a user asks an out-of-scope question (e.g., similarity score < 30%), the bot gracefully redirects them to human support.
- **Object-Oriented Design:** The core retrieval logic is encapsulated in a highly reusable `FAQRetriever` class.
- **Interactive CLI:** A clean, infinite-loop Command Line Interface simulating a real chat environment.

##  Project Structure
```text
calder-faq-chatbot/
├── data/
│   └── faq_data.json       # Structured knowledge base
├── src/
│   ├── retriever.py        # Core NLP logic (TF-IDF & Cosine Similarity)
│   └── main.py             # Interactive Command Line Interface
├── requirements.txt        # Project dependencies
└── README.md               # Documentation