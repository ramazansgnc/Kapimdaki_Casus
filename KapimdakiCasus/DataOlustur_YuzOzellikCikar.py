import os
import cv2
import dlib
import numpy as np
import pickle

def capture_faces_from_camera(user_name, num_samples=100, base_dir='dataset'):
    user_dir = os.path.join(base_dir, user_name)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    video_capture = cv2.VideoCapture(0)
    count = 0

    while count < num_samples:
        ret, frame = video_capture.read()
        if not ret:
            print("Kameradan görüntü alınamadı.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detector = dlib.get_frontal_face_detector()
        faces = face_detector(gray)

        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            face_image = frame[y:y+h, x:x+w]
            if face_image.size == 0:  # Geçersiz yüz bölgelerini kontrol edin
                continue
            face_image = cv2.resize(face_image, (300, 300))
            cv2.imwrite(os.path.join(user_dir, f'{user_name}_{count}.jpg'), face_image)
            count += 1

            # Yüzü çerçevele ve ekranda göster
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def get_face_embeddings_from_image(image_path, model, shape_predictor):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_detector = dlib.get_frontal_face_detector()
    faces = face_detector(gray)

    if len(faces) > 0:
        shape = shape_predictor(image, faces[0])
        face_descriptor = model.compute_face_descriptor(image, shape)
        return np.array(face_descriptor)
    return None

def create_embeddings_dataset(base_dir, model, shape_predictor):
    embeddings = []
    labels = []
    for user_name in os.listdir(base_dir):
        user_dir = os.path.join(base_dir, user_name)
        if os.path.isdir(user_dir):
            for image_name in os.listdir(user_dir):
                image_path = os.path.join(user_dir, image_name)
                embedding = get_face_embeddings_from_image(image_path, model, shape_predictor)
                if embedding is not None:
                    embeddings.append(embedding)
                    labels.append(user_name)
    return embeddings, labels

# Dlib model ve shape predictor yükle
model_path = 'dlib_face_recognition_resnet_model_v1.dat'
shape_predictor_path = 'shape_predictor_68_face_landmarks.dat'
model = dlib.face_recognition_model_v1(model_path)
shape_predictor = dlib.shape_predictor(shape_predictor_path)

# Kullanıcı ekleme işlemi
while True:
    user_name = input("Yeni kullanıcı adı girin: ")
    capture_faces_from_camera(user_name, num_samples=100)

    more_users = input("Başka bir kullanıcı için yüz resimleri toplamak ister misiniz? (e/h): ")
    if more_users.lower() != 'e':
        break

embeddings, labels = create_embeddings_dataset('dataset', model, shape_predictor)

with open('embeddings.pkl', 'wb') as f:
    pickle.dump((embeddings, labels), f)

print("Embeddings dataset 'embeddings.pkl' olarak olusturuldu ve kaydedildi.")
