from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from googletrans import Translator
from langdetect import detect
import base64


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

app = Flask(__name__)

model = load_model("API/kerasmodelLSTMEquals32New.h5")

tokenizer = Tokenizer()

def decode_64(comment):
    bytes_decoded = base64.b64decode(comment)
    string_decoded = bytes_decoded.decode('utf-8')
    return string_decoded

# Função para detectar o idioma e traduzir o texto
def translate_to_english(comment):
    try:
        lang = detect(comment)
        if lang != 'en':
            translator = Translator()
            print("Texto original:", comment)
            translated_text = translator.translate(comment, src=lang, dest='en').text
            return translated_text
        else:
            return comment
    except:
        return None

def preprocess_comment(comment):
    
    # Remover quebras de linha
    comment = comment.replace('\n', ' ').replace('\r', ' ')

    # Converter para minúsculas
    comment = comment.lower()
    
    # Remover URLs
    comment = re.sub(r'http\S+|www\S+|https\S+', '', comment, flags=re.MULTILINE)
    
    # Remover pontuação
    comment = comment.translate(str.maketrans('', '', string.punctuation))
    
    # Remover números
    comment = re.sub(r'\d+', '', comment)
    
    # Tokenizar palavras
    tokens = word_tokenize(comment)
    
    # Filtrar apenas verbos, substantivos, adjetivos e advérbios
    filtered_tokens = [word for word, pos in pos_tag(tokens) if pos.startswith('J') or pos.startswith('N') or pos.startswith('R') or pos.startswith('V')]
    
    # Juntar tokens de volta em uma string
    comment = ' '.join(filtered_tokens)
    
    # Remover espaços excessivos
    comment = comment.strip()
        
    # Adicionar o texto ao Tokenizer
    tokenizer.fit_on_texts([comment])
    
    # Sequenciar o texto
    sequence = tokenizer.texts_to_sequences([comment])
    
    # Preencher sequência
    max_length = 72
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    
    return padded_sequence

@app.route('/predict', methods=['GET'])
def predict():
    comment = request.args.get('text', '')

    if not comment:
        return jsonify({'error': 'Nenhum comentário fornecido'}), 400

    # Decode texto
    decode_comment = decode_64(comment)
    
    # Traduzir textos
    translate_comment = translate_to_english(decode_comment)
    print(translate_comment)
    
    # Preprocessar o comentário
    preprocessed_comment = preprocess_comment(translate_comment)
    print(preprocessed_comment)
    
    # Gerar uma predição falsa
    prediction = model.predict(preprocessed_comment)
    
    # Converter a previsão de ndarray para lista
    prediction_list = prediction.tolist()
    
    # Obter o índice da maior probabilidade na lista
    predicted_class = np.argmax(prediction_list)
    
    # Converter o tipo de dado para int
    predicted_class = int(predicted_class)
    
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082)
