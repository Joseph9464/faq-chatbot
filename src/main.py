import sys
import os

# Cette ligne assure que Python trouve bien le fichier retriever.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from retriever import FAQRetriever

def main():
    # Construction dynamique du chemin vers faq_data.json
    # Cela permet au script de fonctionner peu importe d'où on le lance
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "faq_data.json")

    print("Initialisation du Chatbot Calder, veuillez patienter...")
    
    try:
        # Instanciation de notre moteur de recherche
        chatbot = FAQRetriever(data_path)
    except FileNotFoundError:
        print(f"Erreur ❌ : Le fichier de données est introuvable à l'emplacement {data_path}.")
        return

    print("\n" + "="*55)
    print("🤖 Bienvenue sur le Chatbot Support de Calder !")
    print("   Posez vos questions sur nos services.")
    print("   (Tapez 'quit' ou 'exit' pour quitter le chat)")
    print("="*55 + "\n")

    # Boucle de discussion infinie
    while True:
        try:
            user_input = input("👤 Vous : ")
            
            # Condition de sortie
            if user_input.lower().strip() in ['quit', 'exit']:
                print("\n🤖 Calder : Merci d'avoir utilisé notre service. À bientôt !")
                break
                
            # Ignorer les entrées vides
            if not user_input.strip():
                continue

            # Interroger le moteur TF-IDF
            response = chatbot.get_best_answer(user_input)
            
            # Afficher la réponse
            print(f"🤖 Calder : {response}\n")
            
        except KeyboardInterrupt:
            # Gérer proprement le "Ctrl+C"
            print("\n🤖 Calder : Interruption détectée. Au revoir !")
            break

if __name__ == "__main__":
    main()