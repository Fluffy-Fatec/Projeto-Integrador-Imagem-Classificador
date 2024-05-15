import psycopg2
from connections import conn_params

class SentimentRepository:
    def __init__(self):
        self.conn_params = conn_params

    def save_sentiments(self, texts, sentiments):
        conn = psycopg2.connect(**self.conn_params)
        cursor = conn.cursor()

        for text, sentiment in zip(texts, sentiments):
            cursor.execute("INSERT INTO tabela_sentimentos (texto, sentimento) VALUES (%s, %s)", (text, sentiment))

        conn.commit()
        conn.close()
