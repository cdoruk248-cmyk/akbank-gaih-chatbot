# core/rag_pipeline.py

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def create_rag_chain(api_key):
    """
    Veriyi koda gömerek RAG zincirini oluşturan ana fonksiyon.
    """
    # --- VERİYİ DOĞRUDAN KODUN İÇİNE YAZIYORUZ ---
    film_texts = [
        # ... (Film metinleri burada, aynen kalacak) ...
        """Film Adı: The Dark Knight (Kara Şövalye)
        Yönetmen: Christopher Nolan
        Tür: Süper Kahraman, Aksiyon, Bilim Kurgu, Gerilim
        Konu Özeti: Gotham şehrini Joker'in anarşist planlarından korumaya çalışan Batman'in mücadelesi.
        Notlar: Heath Ledger'ın ikonik Joker performansı, harika senaryosu ve oyuncuları ile bilinir.""",
        """Film Adı: The Shawshank Redemption (Esaretin Bedeli)
        Yönetmen: Frank Darabont
        Tür: Dram
        Konu Özeti: Haksız yere hapse atılan bankacı Andy Dufresne'in, Shawshank hapishanesindeki umut dolu hayatta kalma mücadelesi.
        Notlar: IMDb'nin en yüksek puanlı filmi. Umut, dostluk ve sabır temalarını işler. Morgan Freeman'ın anlatımıyla öne çıkar.""",
        """Film Adı: The Godfather (Baba)
        Yönetmen: Francis Ford Coppola
        Tür: Suç, Dram
        Konu Özeti: Güçlü bir İtalyan mafya ailesi olan Corleone'lerin reisi Don Vito Corleone'nin, işlerini oğlu Michael'a devretme sürecindeki güç savaşları.
        Notlar: Sinema tarihinin en iyi filmlerinden biri olarak kabul edilir. Marlon Brando ve Al Pacino'nun unutulmaz performansları vardır.""",
        """Film Adı: 12 Angry Men (12 Öfkeli Adam)
        Yönetmen: Sidney Lumet
        Tür: Dram, Gerilim
        Konu Özeti: Bir cinayet davasında, bir jüri üyesinin, sanığın masum olabileceğine dair makul şüpheleri olduğunu belirterek diğer 11 üyeyi ikna etme çabası.
        Notlar: Neredeyse tamamı tek bir odada geçer. Adalet, önyargı ve insan psikolojisi üzerine yoğunlaşan diyaloglarıyla ünlüdür.""",
        """Film Adı: Pulp Fiction
        Diğer Adı: Ucuz Roman
        Yönetmen: Quentin Tarantino
        Tür: Suç, Dram, Kara Mizah
        Konu Özeti: Los Angeles'ın suç dünyasında iki tetikçi, bir boksör ve bir gangsterin karısının kesişen hayatlarını anlatan, doğrusal olmayan kurguya sahip bir film.
        Notlar: Quentin Tarantino'nun kendine has diyalogları ve ikonik sahneleriyle bir kült filmdir. Kurgusal yapısıyla sinemada çığır açmıştır."""
    ]

    documents = [Document(page_content=text) for text in film_texts]

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = Chroma.from_documents(documents=documents, embedding=embeddings)

    # --- LangChain ile RAG Zinciri Oluşturma ---
    template = """
    Sen filmler hakkında bilgi veren yardımcı bir asistansın. 
    Kullanıcının sorusunu cevaplamak için aşağıdaki film özetlerini kullan.
    Eğer cevabı bilmiyorsan, sadece bilmediğini söyle, tahmin yürütme.

    BAĞLAM: {context}
    SORU: {question}
    CEVAP:
    """
    prompt = PromptTemplate.from_template(template)

    llm = ChatGoogleGenerativeAI(model="models/gemini-pro-latest", google_api_key=api_key)

    rag_chain = (
        {"context": vector_store.as_retriever(search_kwargs={'k': 2}), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain