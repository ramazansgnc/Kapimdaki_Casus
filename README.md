DataOlustur_YuzOzellikCikar.py kullanılarak kullanıcı için giriş istenir. Her bir profil için klasör oluşturulur. Bu profilin yüz fotoğrafları kamera ile alınır. Kişinin yüz özellikleri çıkartılıp embeddings.pkl dosyasına kaydedilir.(en az iki kişi kaydedin)

Model Eğitimi embeddings.pkl dosyası kullanılarak bir model oluşturulur ve face_recognition_model.pkl olarak model kaydedilir.

Son olarak Yuz_Tanima.py dosyası çalıştırılarak kişinin profil ismi ya da bilinmiyor olarak kamerada yüz çerçeve içerisinde tanımlanır. Eğer tanınma olasılığı %75 altında ise bilinmiyor olarak etiketlenir ve anlık o fotoğraf karesi Bilinmeyenler klasörüne kaydedilir.

NOT:Bu projede "face_recognition_models-master" klasör içerisinde yüz algılama ve tanıma için gereken model dosyaları bulunuyor.
