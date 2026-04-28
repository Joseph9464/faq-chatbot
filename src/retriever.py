import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

class FAQRetriever:
    def __init__(self, data_path: str):
        """
        Initialise le moteur de recherche en chargeant les données et en entraînant le vectoriseur.
        """
        self.data_path = data_path
        self.faqs = self._load_data()
        
        # Séparer les questions et les réponses
        self.questions = [faq['question'] for faq in self.faqs]
        self.answers = [faq['answer'] for faq in self.faqs]
        
        # Initialiser et entraîner le modèle TF-IDF sur nos questions FAQ
        # stop_words='english' permet d'ignorer les mots très communs (the, is, at...)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.question_vectors = self.vectorizer.fit_transform(self.questions)

    def _load_data(self) -> list:
        """Charge le fichier JSON contenant la FAQ."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Le fichier {self.data_path} est introuvable.")
            
        with open(self.data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data.get('faqs', [])

    def get_best_answer(self, user_query: str, threshold: float = 0.3) -> str:
        """
        Trouve la réponse la plus pertinente pour la question de l'utilisateur.
        
        :param user_query: La question posée par l'utilisateur.
        :param threshold: Le score de similarité minimum (entre 0 et 1) pour accepter une réponse.
        :return: La réponse correspondante ou un message d'erreur (fallback).
        """
        # 1. Transformer la question de l'utilisateur en vecteur
        query_vector = self.vectorizer.transform([user_query])
        
        # 2. Calculer la similarité cosinus avec toutes les questions de la FAQ
        similarities = cosine_similarity(query_vector, self.question_vectors).flatten()
        
        # 3. Trouver l'index du meilleur score
        best_match_index = np.argmax(similarities)
        best_score = similarities[best_match_index]
        
        # 4. Gérer le "fallback" (exigence du cahier des charges)
        if best_score < threshold:
            return "I'm sorry, I couldn't find an answer to your question. Would you like to speak with our support team at support@calder.com?"
            
        return self.answers[best_match_index]

# ==========================================
# Bloc de test (exécuté uniquement si on lance ce fichier directement)
# ==========================================
if __name__ == "__main__":
    # Ajuste le chemin selon l'endroit d'où tu lances le script
    path_to_json = "../data/faq_data.json" 
    
    try:
        retriever = FAQRetriever(path_to_json)
        
        print("--- Test du FAQ Retriever ---")
        # Test 1: Correspondance exacte (ou presque)
        print("\nQ: I forgot my password, how do I reset it?")
        print("R:", retriever.get_best_answer("I forgot my password, how do I reset it?"))
        
        # Test 2: Mots différents mais sens similaire
        print("\nQ: When is the support team online?")
        print("R:", retriever.get_best_answer("When is the support team online?"))
        
        # Test 3: Hors sujet (doit déclencher le fallback)
        print("\nQ: How do I cook a pizza?")
        print("R:", retriever.get_best_answer("How do I cook a pizza?"))
        
    except Exception as e:
        print(f"Erreur : {e}")