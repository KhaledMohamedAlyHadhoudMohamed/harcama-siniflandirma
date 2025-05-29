# Harcama Açıklaması Sınıflandırma Projesi

Bu proje, banka işlem açıklamalarını harcama kategorilerine göre otomatik sınıflandırır. Word2Vec ve TF-IDF modelleri ile benzerlik analizleri yapılır.

## 💻 Gereksinimler

```
pip install -r requirements.txt
```

## 🚀 Kullanım

### 1. Model Eğitimi
```
python train.py
```

### 2. Açıklama Sınıflandırma
```
python classify.py --model word2vec_cbow_50.model --input "SHELL PETROL İSTANBUL"
```

## 📁 Dosyalar

- `train.py`: Word2Vec & TF-IDF model eğitimi
- `classify.py`: Açıklama benzerliği ve kategori tahmini
- `utils.py`: Metin temizleme ve vektörleştirme
- `harcama_aciklamalari_raw.csv`: 200K satırlık veri
- `requirements.txt`: Gerekli Python paketleri