from model_utils import SentimentClassifier
import os
import time

def train_main():
    dataset_path = 'training.1600000.processed.noemoticon.csv'
    
    if not os.path.exists(dataset_path):
        print(f"Erro: O dataset '{dataset_path}' não foi encontrado.")
        return

    print("Iniciando o treinamento do modelo (isso pode levar alguns minutos)...")
    start_time = time.time()
    
    classifier = SentimentClassifier()
    # Treinando com 200k para ser rápido como no app.py original
    classifier.train(dataset_path, sample_size=200000)
    
    print("Treinamento concluído. Salvando modelo...")
    classifier.save()
    
    duration = time.time() - start_time
    print(f"Concluído em {duration:.2f} segundos. Arquivos salvos com sucesso.")

if __name__ == "__main__":
    train_main()
