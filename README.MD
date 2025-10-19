# 🎬 Kişisel Film ve Dizi Asistanı Chatbot'u

Bu proje, Akbank & Global AI Hub Generative AI Bootcamp'i için geliştirilmiş, RAG (Retrieval-Augmented Generation) mimarisine sahip bir chatbot'tur.

## 🎯 Projenin Amacı

Bu chatbot, kullanıcının sorduğu sorulara, IMDb'nin en iyi filmlerinden oluşan bir bilgi tabanını kullanarak doğal dilde cevaplar verir. Amacımız, bir film hakkında anında, sohbet formatında bilgi almayı sağlamaktır.

## 📂 Veri Seti

Bilgi tabanı olarak IMDb'nin en yüksek puanlı 5 filminin özetleri, yönetmenleri ve tür bilgileri kullanılmıştır. Harici bir dosya sistemine bağımlılığı ortadan kaldırmak ve "hayalet hataları" engellemek için, veri seti doğrudan `app.py` kodunun içine bir Python listesi olarak gömülmüştür.

Kullanılan filmler:
* The Dark Knight
* The Shawshank Redemption
* The Godfather
* 12 Angry Men
* Pulp Fiction

## 🛠️ Kullanılan Teknolojiler

* **Programlama Dili:** Python
* **Web Arayüzü:** Streamlit
* **RAG Mimarisi:** LangChain
* **LLM (Büyük Dil Modeli):** Google Gemini (`models/gemini-pro-latest`)
* **Embedding Modeli:** Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`)
* **Vektör Veritabanı:** ChromaDB

## 📁 Proje Yapısı

Proje, basit ve anlaşılır bir dosya yapısına sahiptir:

├── app.py             # 🚀 Ana Streamlit uygulamasını ve RAG mantığını içerir
├── requirements.txt   # 📦 Python bağımlılıklarının listesi
├── README.md          # 📄 Bu bilgilendirme dosyası
├── .env               # 🔑 API anahtarını saklayan dosya (versiyon kontrolüne dahil edilmez)
└── .gitignore         # 🚫 Versiyon kontrolü dışında tutulacak dosyaları belirtir

## 🚀 Çalıştırma Kılavuzu

1.  Proje klonlandıktan sonra, sanal bir ortam oluşturun:
    `python -m venv venv`
2.  Sanal ortamı aktif edin:
    `.\venv\Scripts\activate`
3.  Gerekli kütüphaneleri yükleyin:
    `pip install -r requirements.txt`
4.  `.env` dosyasını oluşturup içine `GOOGLE_API_KEY="AIzaSy..."` şeklinde API anahtarınızı ekleyin.
5.  Uygulamayı çalıştırın:
    `streamlit run app.py`

## 🎯 Örnek Sorular

Chatbot'a aşağıdaki gibi sorular sorabilirsiniz:

- "Pulp Fiction filminin konusu nedir?"
- "Baba filminin yönetmeni kim?"
- "Nolan'ın filmi hakkında bilgi verir misin?"
- "Dram türündeki filmler hangileri?"

## 🖼️ Ekran Görüntüsü
![alt text](image.png)

(Buraya chatbot'un çalışan halinin bir ekran görüntüsünü ekleyebilirsin)

## Web Linki: https://huggingface.co/spaces/dorukcelik/film-oneri-chatbot

![alt text](image-1.png)