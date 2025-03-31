from transformers import AutoTokenizer, AutoModel
from tensorflow.keras.models import load_model
from deep_translator import GoogleTranslator
from flask import Flask, request, jsonify
import xml.etree.ElementTree as ET
import tensorflow as tf
import numpy as np
import joblib
import torch
import spacy
import bz2

app = Flask(__name__)

# Завантажуємо StandardScaler
scaler = joblib.load('scaler.pkl')

# Завантажуємо модель
model = tf.keras.models.load_model('model.keras')
model_2 = load_model("model_2.h5")

# Завантажуємо Spacy-модель
nlp = spacy.load('uk_core_news_lg')

# Завантажуємо GloVe-словник
glove_path = "./news.lowercased.lemmatized.glove.300d.bz2"
word_vectors = {}

        
# Load the pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
bert_model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")

# Define unique labels
unique_labels = ['Заперечення', 'Виправдовування', 'Заклик', 
                 'Розпалювання ворожнечі та ненависті', 
                 'Приниження національної честі та гідності', 
                 'Без наративу']

label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
id_to_label = {idx: label for label, idx in label_to_id.items()}

with bz2.open(glove_path, "rt", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        word = parts[0]
        vector = np.array(parts[1:], dtype=np.float32)
        word_vectors[word] = vector
        
def preprocess_text_2(text):
    nlp = spacy.load("ru_core_news_sm")
    translated_text = GoogleTranslator(source='uk', target='ru').translate(text)
    doc = nlp(str(translated_text).lower()) 
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def get_avg_w2v(text, tokenizer, bert_model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    vector = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return vector

def count_named_entities(text):
    doc = nlp(text)
    location_count = sum(1 for ent in doc.ents if ent.label_ == "LOC")  # Count locations
    organization_count = sum(1 for ent in doc.ents if ent.label_ == "ORG")  # Count organizations
    return location_count, organization_count

def avg_noun_verb_ratio(text):
    doc = nlp(text)
    ratios = []
    for sent in doc.sents:  # Process each sentence separately
        nouns = sum(1 for token in sent if token.pos_ == "NOUN")
        verbs = sum(1 for token in sent if token.pos_ == "VERB")
        if verbs > 0:
            ratios.append(nouns / verbs)  # Compute noun/verb ratio
        else:
            ratios.append(0)  # Avoid division by zero
    return sum(ratios) / len(ratios) if ratios else 0  # Compute the average

def calculate_subj(text):
    subj_dict_synt = {}
    tree = ET.parse("translated_output.xml")
    root = tree.getroot()
    # Extract words and polarity
    for word in root.findall("word"):
        word_form = word.get("form")
        polarity = float(word.get("subjectivity", 0))  # Default polarity = 0 if not present
        subj_dict_synt[word_form] = polarity
    if not isinstance(text, str):
        return 0.0  # Return neutral score for missing values
    words = text.split()  # Tokenize text
    score = sum(subj_dict_synt.get(word, 0) for word in words)  # Sum word polarities
    return score

def calculate_sentiment(text):
    sentiment_dict_synt = {}
    tree = ET.parse("translated_output.xml")
    root = tree.getroot()
    # Extract words and polarity
    for word in root.findall("word"):
        word_form = word.get("form")
        polarity = float(word.get("polarity", 0))  # Default polarity = 0 if not present
        sentiment_dict_synt[word_form] = polarity
    if not isinstance(text, str):
        return 0.0  # Return neutral score for missing values
    words = text.split()  # Tokenize text
    score = sum(sentiment_dict_synt.get(word, 0) for word in words)  # Sum word polarities
    return score


def preprocess_text(text):
    doc = nlp(text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc])
    
    has_colons = 1 if ':' in text else 0
    has_hyphens = 1 if '-' in text else 0
    has_quotmarks = 1 if '"' in text else 0
    
    # sentiment = 0.0  # Тут має бути твоя функція
    # subjectiveness = 0.0
    # noun_verb_ratio = 0.0
    # location_count, organization_count = 0, 0  # Аналогічно
    
    sentiment = calculate_sentiment(lemmatized_text)
    subjectiveness = calculate_subj(lemmatized_text)
    noun_verb_ratio = avg_noun_verb_ratio(lemmatized_text)
    location_count, organization_count = count_named_entities(text)
    
    vectors = [word_vectors[token.lemma_.lower()] for token in doc if token.lemma_.lower() in word_vectors]
    vector = np.mean(vectors, axis=0) if vectors else np.zeros(300)

    numeric_features = np.array([sentiment, subjectiveness, noun_verb_ratio, location_count, organization_count]).reshape(1, -1)
    scaled_features = scaler.transform(numeric_features).flatten()

    return np.concatenate([vector, [has_colons, has_hyphens, has_quotmarks], scaled_features])



def multilabel(text):
    # Preprocess the input sentence
    processed_text = preprocess_text_2(text)
    
    # Convert the processed text to a vector (наприклад, Word2Vec або будь-який інший)
    vector = get_avg_w2v(processed_text, tokenizer, bert_model)
    
    vector = vector[:767]

    # Передбачення ймовірностей
    probabilities = model_2.predict(np.array([vector]))

    # Нормалізовані ймовірності (якщо необхідно)
    probabilities_dict = {
        id_to_label[i]: float(probabilities[0][i]) for i in range(len(probabilities[0]))
    }

    # Найвірогідніший клас
    predicted_label = id_to_label[np.argmax(probabilities)]

    return {
        "text": text,
        "processed_text": processed_text,
        "label": predicted_label,
        "probabilities": probabilities_dict,
    }

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    text = data['text']

    # First-level classification
    features = preprocess_text(text)
    prediction = model.predict(np.expand_dims(features, axis=0))[0]
    label = "Propaganda" if prediction > 0.5 else "Not propaganda"

    response = {"class": label}

    # If classified as "Propaganda", apply second-level classification
    if prediction > 0.5:
        multilabel_result = multilabel(text)
        response["multilabel"] = multilabel_result  # Attach multilabel results

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
