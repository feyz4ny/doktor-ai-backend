from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os  # Port ve dosya yönetimi için gerekli

app = FastAPI()

# Tüm dünyadan (buluttan) erişim için izinler
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. VERİSETİNİ YÜKLE
print("Veriseti yükleniyor ve AI motoru hazırlanıyor...")

# Dosya yolunu güvenli hale getirelim
base_path = os.path.dirname(__file__)
csv_path = os.path.join(base_path, 'veriseti.csv')

try:
    df = pd.read_csv(csv_path, sep=';')
    sikayetler = df['Birleşik Belirtiler'].astype(str).tolist()
    cevaplar = df['Triyaj Cevap'].astype(str).tolist()

    # 2. ANLAMSAL EŞLEŞTİRME (AI MOTORU)
    vectorizer = TfidfVectorizer()
    sikayet_vektorleri = vectorizer.fit_transform(sikayetler)
    print("Sistem hazır! Sunucu bulut için başlatılıyor...")
except Exception as e:
    print(f"Hata: veriseti.csv dosyası işlenemedi! {e}")

class Soru(BaseModel):
    text: str

@app.post("/tahmin")
async def tahmin_yap(item: Soru):
    kullanici_vektoru = vectorizer.transform([item.text])
    
    benzerlikler = cosine_similarity(kullanici_vektoru, sikayet_vektorleri)
    en_yakin_index = benzerlikler.argmax()
    benzerlik_skoru = benzerlikler[0][en_yakin_index]
    
    # GÜVENLİK EŞİĞİ (THRESHOLD)
    if benzerlik_skoru < 0.45:
        cevap = "⚪ BELİRSİZ: Şikayetinizi tam anlayamadım. Teşhis koyabilmem için lütfen şikayetinizi, ağrının tam yerini ve ne kadar süredir devam ettiğini daha detaylı yazar mısınız?"
    else:
        cevap = cevaplar[en_yakin_index]
    
    # Log kayıtları
    print(f"\nİSTEK: {item.text} | SKOR: %{int(benzerlik_skoru*100)}")
    
    return {"sonuc": cevap}

# --- BULUT SUNUCUSU İÇİN KRİTİK AYAR ---
if __name__ == "__main__":
    # Render veya diğer bulut servisleri portu otomatik atar.
    # Eğer atamazsa varsayılan olarak 8000 portunu kullanır.
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)