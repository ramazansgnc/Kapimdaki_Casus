import cv2
import dlib
import numpy as np
import pickle
import time
import os
import firebase_admin
from firebase_admin import credentials, storage, db

# Firebase ayarları
cred = credentials.Certificate("C:\\Users\\RAMAZAN\\Desktop\\en son halli\\kapimdakicasus-c26f2-firebase-adminsdk-f3f2g-0664f4fb50.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'kapimdakicasus-c26f2.appspot.com',
    'databaseURL': 'https://kapimdakicasus-c26f2-default-rtdb.firebaseio.com/'
})
bucket = storage.bucket()

# Model ve shape predictor yollarını ayarla
model_path = 'dlib_face_recognition_resnet_model_v1.dat'
shape_predictor_path = 'shape_predictor_68_face_landmarks.dat'

# Dlib modellerini yükle
model = dlib.face_recognition_model_v1(model_path)
shape_predictor = dlib.shape_predictor(shape_predictor_path)
face_detector = dlib.get_frontal_face_detector()

# Eğitimli SVM modelini ve etiketleri yükle
with open('face_recognition_model.pkl', 'rb') as f:
    clf, labels = pickle.load(f)

# "Tanınmayan kişiler" klasörünü oluştur
unknown_faces_dir = "Bilinmeyenler"
if not os.path.exists(unknown_faces_dir):
    os.makedirs(unknown_faces_dir)

# VideoCapture nesnesi oluştur
video_capture = cv2.VideoCapture(0)

# Yüz tanımlanamama sürelerini takip etmek için bir sözlük oluştur
unknown_faces = {}

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Kameradan görüntü alınamadı.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray_frame)

    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        face_image = frame[y:y+h, x:x+w]

        if face_image.size > 0:
            shape = shape_predictor(rgb_frame, face)
            face_descriptor = model.compute_face_descriptor(rgb_frame, shape)
            face_descriptor = np.array(face_descriptor).reshape(1, -1)

            # Tahmin yap ve olasılık değerlerini al
            probabilities = clf.predict_proba(face_descriptor)[0]
            max_prob = max(probabilities)
            prediction = clf.classes_[np.argmax(probabilities)]
            p = max_prob * 100

            if p < 75:
                name = "Bilinmiyor"
                percentage = max_prob * 100

                # Yüzün tanımlanamama süresini takip et
                face_id = (x, y, w, h)  # Yüzü tanımlamak için koordinatları kullan
                current_time = time.time()
                if face_id not in unknown_faces:
                    unknown_faces[face_id] = current_time
                else:
                    elapsed_time = current_time - unknown_faces[face_id]
                    if elapsed_time > 5:
                        # Ekran görüntüsü al ve kaydet
                        screenshot_name = os.path.join(unknown_faces_dir, f"screenshot_{int(current_time)}.png")
                        cv2.imwrite(screenshot_name, frame)
                        print(f"Ekran görüntüsü kaydedildi: {screenshot_name}")

                        blob = bucket.blob(f"Bilinmeyenler/{os.path.basename(screenshot_name)}")
                        blob.upload_from_filename(screenshot_name)
                        print(f"Ekran görüntüsü Firebase'e yüklendi: {blob.public_url}")

                        # Veritabanına yeni resim kaydını ekle
                        ref = db.reference('unknown_faces')
                        ref.push({
                            'timestamp': current_time,
                            'image_url': blob.public_url
                        })

                        # Kaydedildikten sonra yüzü sözlükten kaldır
                        del unknown_faces[face_id]

            else:
                name = prediction
                percentage = max_prob * 100

                # Tanımlanan yüzleri sözlükten çıkar
                face_id = (x, y, w, h)
                if face_id in unknown_faces:
                    del unknown_faces[face_id]

            label = f"{name} ({percentage:.2f}%)"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
