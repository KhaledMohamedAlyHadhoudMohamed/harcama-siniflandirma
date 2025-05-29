import argparse
import gensim
import pandas as pd
from utils import preprocess_text, average_word2vec
from sklearn.metrics.pairwise import cosine_similarity

def classify_word2vec(input_text, model_path, dataset_path):
    model = gensim.models.Word2Vec.load(model_path)
    df = pd.read_csv(dataset_path)
    texts = df["text"].tolist()

    input_words = preprocess_text(input_text)
    input_vec = average_word2vec(input_words, model)
    if input_vec is None:
        print("❌ Giriş kelimeleri modelde bulunamadı.")
        return

    similarities = []
    for text in texts[:1000]:
        words = preprocess_text(text)
        vec = average_word2vec(words, model)
        if vec is not None:
            sim = cosine_similarity([input_vec], [vec])[0][0]
            similarities.append((text, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    print("\n📌 En benzer 5 açıklama:")
    for text, score in similarities[:5]:
        print(f"{text} → {score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", default="harcama_aciklamalari_raw.csv")
    args = parser.parse_args()

    if args.model.endswith(".model"):
        classify_word2vec(args.input, args.model, args.dataset)