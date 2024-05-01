import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

# Chargement du tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Chargement du modèle sauvegardé sur Hugging Face car assez lourd (419 Mo)
model = BertForSequenceClassification.from_pretrained('ThisPickles/PredicCat')

df = pd.read_csv('corpus.csv')
# Définition de la fonction de prédiction
def predict_genre(description):
    inputs = tokenizer(description, padding=True, truncation=True, max_length=512, return_tensors="pt")
    outputs = model(**inputs)
    predicted_label_id = torch.argmax(outputs.logits).item()
    predicted_genre = df['Genres'].iloc[predicted_label_id]
    return predicted_genre


descriptions = [
    "Plongez dans un monde fantastique rempli de mystères et de dangers. Explorez des donjons sombres, affrontez des monstres redoutables et découvrez des trésors cachés dans ce RPG épique.",
    "Prenez le volant des voitures les plus rapides du monde et affrontez vos adversaires sur des circuits spectaculaires. Avec des graphismes à couper le souffle et des sensations de vitesse incroyables, ce jeu de course vous fera vivre des moments inoubliables.",
    "Devenez le héros de votre propre aventure dans ce jeu d'action en monde ouvert. Parcourez des paysages magnifiques, accomplissez des quêtes épiques et affrontez des ennemis redoutables pour sauver le royaume.",
    "Construisez et gérez votre propre ville dans ce jeu de simulation addictif. Répondez aux besoins de vos citoyens, développez des infrastructures modernes et assurez-vous que votre ville prospère dans un environnement dynamique et réaliste.",
    "Plongez dans l'univers mystérieux de l'espace dans ce jeu de science-fiction captivant. Explorez des planètes lointaines, combattez des aliens hostiles et découvrez les secrets cachés de l'univers.",
    "Affrontez des hordes de zombies affamés dans ce jeu de survie palpitant. Utilisez vos compétences de combat et de stratégie pour rester en vie dans un monde post-apocalyptique rempli de dangers mortels.",
    "Prenez les commandes de puissants navires de guerre et engagez-vous dans des batailles navales épiques dans ce jeu de stratégie tactique. Dominez les mers et devenez le maître des océans.",
    "Explorez des jungles luxuriantes, escaladez des montagnes escarpées et découvrez des trésors perdus dans ce jeu d'aventure épique. Affrontez des pièges mortels et des ennemis redoutables dans votre quête de gloire et de fortune.",
    "Entrez dans l'arène et affrontez d'autres joueurs dans des combats PvP intenses. Utilisez vos compétences et votre intelligence pour devenir le champion ultime et remporter des récompenses fabuleuses.",
    "Construisez et personnalisez votre propre robot de combat dans ce jeu de science-fiction futuriste. Affrontez d'autres robots d'autres joueurs dans des arènes de combat mortelles et prouvez votre valeur en tant que pilote de robot d'élite.",
    "amoureux nature pêche sportive fait cette simulation réalisée passionnés retranscrit toutes subtilités cette pratique authenticité inégalée amateur désireux ’ apprendre pêcheur émérite expériences vivez the fisherman permettront ’ améliorer compétences réelles pêcheur ligne"
]

for descrip in descriptions:
    pre_genre = predict_genre(descrip)
    print("Description :", descrip)
    print("Genre prédit :", pre_genre)
    print()
