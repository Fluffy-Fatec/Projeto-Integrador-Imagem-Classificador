from flask import Flask, request
import os
import pandas as pd

from service import SentimentAnalysisService
from repository import SentimentRepository
from database.connections import conn_params

app = Flask(__name__)

model_path = 'modelo_analise_sentimento.h5'

service = SentimentAnalysisService(model_path)
repository = SentimentRepository(conn_params)

@app.route('/processar_csv', methods=['POST'])
def processar_csv():
    csv_file = request.files['file']
    df = pd.read_csv(csv_file)

    # Pré-processamento dos dados
    processed_data = service.preprocess_data(df['texto'])

    # Análise de sentimento
    sentiments = service.analyze_sentiment(processed_data)

    # Salvar no banco de dados
    repository.save_sentiments(df['texto'], sentiments)

    return "CSV processado e resultados salvos no banco de dados com sucesso!"

if __name__ == '__main__':
    app.run(debug=True)
