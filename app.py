# app.py

import streamlit as st
from dotenv import load_dotenv
import os
from core.rag_pipeline import create_rag_chain # <-- Yeni fonksiyonumuzu import ediyoruz

# API anahtarını .env dosyasından yükle
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Eğer anahtar bulunamazsa hata ver ve dur
if not api_key:
    st.error("GOOGLE_API_KEY bulunamadı. Lütfen .env dosyasını kontrol edin.")
else:
    # --- Sayfa Ayarları ve Başlık ---
    st.set_page_config(page_title="Film Öneri Chatbotu", page_icon="🎬")
    st.title("🎬 Kişisel Film ve Dizi Asistanı")
    st.subheader("IMDb'nin en iyileri hakkında sohbet edelim!")

    # RAG zincirini sadece bir kere oluşturup hafızaya al
    @st.cache_resource
    def load_chain():
        return create_rag_chain(api_key)

    rag_chain = load_chain()
    st.success("Film veritabanı başarıyla yüklendi!")

    # --- Sohbet Arayüzü ---
    user_question = st.text_input("Bir film hakkında soru sorun:")

    if st.button("Gönder"):
        if user_question:
            with st.spinner("Cevap hazırlanıyor..."):
                response = rag_chain.invoke(user_question)
                st.info(response)
        else:
            st.warning("Lütfen bir soru sorun.")