import re
import scrapy
from scrapy.crawler import CrawlerProcess
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string


class TextNormalizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

    # Fonction permettant la normalisation de la données
    def normalize_text(self, text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('french'))
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        normalized_text = ' '.join(tokens)
        return normalized_text

class Scrapper(scrapy.Spider):
    def __init__(self, *args, **kwargs):
        super(Scrapper, self).__init__(*args, **kwargs)
        self.text_normalizer = TextNormalizer()

    name = "scrapper"
    data = []
    start_urls = [
                     f'https://www.instant-gaming.com/fr/rechercher/?platform%5B0%5D=&type%5B0%5D=&sort_by=&min_reviewsavg=10&max_reviewsavg=100&noreviews=1&min_price=0&max_price=200&noprice=1&gametype=all&search_tags=0&query=&page={x}'
                     for x in range(1, 217)
                 ]

    def parse(self, response):
        # Récupération des liens à scrapper
        links = response.css('div.item.force-badge a.cover.video::attr(href)').getall()
        for link in links:
            yield response.follow(link, callback=self.parse_game)

    def parse_game(self, response):
        # Nettoyage des données
        raw_title = response.css('title::text').get()
        clean_title = re.sub(r'\b(?:Acheter|Steam|GOG.com|Epic Games|(PS4 / PS5)|PS5|Playstation Store|Switch Nintendo Eshop)\b', '', raw_title.strip())

        raw_description = response.css('div.text.readable span::text').getall()
        cleaned_description = '\n'.join(raw_description).strip()
        cleaned_description = '\n'.join(line.strip() for line in cleaned_description.split('\n') if line.strip())
        cleaned_description = cleaned_description.strip('[]"')
        cleaned_description = '\n'.join(
            line for line in cleaned_description.split('\n') if not line.strip().startswith('•'))

        genres_list = response.css('div.genres a::text').getall()
        cleaned_genres = [genre.strip() for genre in genres_list if genre.strip().lower() != 'jeux solo']
        cleaned_genres = ', '.join(cleaned_genres)

        # Normalisation des données récuperer
        clean_title = self.text_normalizer.normalize_text(clean_title)
        cleaned_description = self.text_normalizer.normalize_text(cleaned_description)
        cleaned_genres = self.text_normalizer.normalize_text(cleaned_genres)

        # Ajout des éléments dans un dictionnaire
        item = {
            'Titre': clean_title,
            'Description': cleaned_description,
            'Genres': cleaned_genres
        }

        # Ajouts du dictionnaire dans une liste
        self.data.append(item)

    def closed(self, reason):
        # Convertir les données en DataFrame
        df = pd.DataFrame(self.data)

        # Normaliser les descriptions, titres et genres
        df['Titre'] = df['Titre'].apply(self.text_normalizer.normalize_text)
        df['Description'] = df['Description'].apply(self.text_normalizer.normalize_text)
        df['Genres'] = df['Genres'].apply(self.text_normalizer.normalize_text)


        # Enregistrer la données dans un fichier CSV
        df.to_csv('corpus.csv', index=False, encoding='utf-8')

if __name__ == "__main__":
    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    })

    process.crawl(Scrapper)
    process.start()
