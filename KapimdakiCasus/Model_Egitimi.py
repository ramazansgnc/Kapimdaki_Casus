import pickle
import numpy as np
from sklearn.svm import SVC

# Embeddings ve etiketleri yükleyin
with open('embeddings.pkl', 'rb') as f:
    embeddings, labels = pickle.load(f)

# Embeddings'leri ve etiketleri numpy array'e çevir
embeddings = np.array(embeddings)
labels = np.array(labels)

# SVM sınıflandırıcısını oluştur ve eğit
clf = SVC(kernel='linear', probability=True)
clf.fit(embeddings, labels)

# Modeli ve etiketleri birlikte kaydedin
with open('face_recognition_model.pkl', 'wb') as f:
    pickle.dump((clf, labels), f)

print("Model ve etiketler kaydedildi.")
