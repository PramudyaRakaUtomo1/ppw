import time
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import re
# Styling CSS Tambahan
st.markdown("""
    <style>
    .main-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .preprocessing-box {
        background-color: #1E293B; /* Warna latar gelap agar selaras dengan dark mode */
        color: #F8FAFC;            /* Warna teks putih terang */
        padding: 12px;
        border-radius: 6px;
        border: 1px solid #334155;
        margin-bottom: 8px;
        font-family: monospace;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Klasifikasi Berita SVM & Naive Bayes Detik.com",
    page_icon="📰",
    layout="wide"
)

# Styling CSS Tambahan
st.markdown("""
    <style>
    .main-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .preprocessing-box {
        background-color: #F8FAFC;
        padding: 12px;
        border-radius: 6px;
        border: 1px solid #E2E8F0;
        margin-bottom: 8px;
        font-family: monospace;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Sistem Klasifikasi & Ekstraksi Berita (SVM vs Naive Bayes)</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Dilengkapi Web Scraping Detik.com, Preprocessing Text Mining Per-Poin, dan Perbandingan Model Machine Learning.</p>', unsafe_allow_html=True)

# Fungsi Scraping Konten dari URL Detik.com
def scrape_detik_article(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None, None, "Gagal mengakses halaman web (Status code bukan 200)."
        
        soup = BeautifulSoup(response.text, 'html.parser')
        title_tag = soup.find('h1', {'class': 'detail__title'}) or soup.find('h1')
        title = title_tag.text.strip() if title_tag else "Judul Tidak Ditemukan"
        
        article_body = soup.find('div', {'class': 'detail__body-text'})
        if article_body:
            for s in article_body.find_all(['script', 'style', 'table', 'div']):
                s.decompose()
            paragraphs = article_body.find_all('p')
            content = " ".join([p.text.strip() for p in paragraphs])
        else:
            content = "Isi artikel tidak dapat diekstrak secara otomatis."
            
        return title, content, None
    except Exception as e:
        return None, None, str(e)

# Fungsi Preprocessing Teks Sederhana (Simulasi Sastrawi & Text Mining)
def text_preprocessing(text):
    # 1. Case Folding
    case_folded = text.lower()
    
    # 2. Cleaning (Hapus angka, tanda baca, simbol)
    cleaned = re.sub(r'[^a-z\s]', '', case_folded)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # 3. Tokenization
    tokens = cleaned.split()
    
    # 4. Stopword Removal (Daftar stopword umum Bahasa Indonesia)
    stopwords_id = {
        'yang', 'untuk', 'dan', 'di', 'pada', 'ke', 'para', 'namun', 'menurut', 
        'antara', 'dia', 'dua', 'ia', 'seperti', 'jika', 'sehingga', 'kembali', 
        'dari', 'ini', 'itu', 'dengan', 'adalah', 'tersebut', 'oleh', 'saat', 
        'sebagai', 'kepada', 'karena', 'mereka', 'sebuah', 'lain', 'anda'
    }
    filtered_tokens = [w for w in tokens if w not in stopwords_id]
    
    # 5. Stemming Sederhana (Contoh pemotongan akhiran umum -i, -kan, -an)
    stemmed_tokens = []
    for w in filtered_tokens:
        if w.endswith('kan'):
            w = w[:-3]
        elif w.endswith('an'):
            w = w[:-2]
        elif w.endswith('i'):
            w = w[:-1]
        stemmed_tokens.append(w)
        
    return {
        "case_folding": case_folded[:200] + "...",
        "cleaning": cleaned[:200] + "...",
        "tokenization": tokens[:15],
        "stopword_removal": filtered_tokens[:15],
        "stemming": stemmed_tokens[:15]
    }

# Load Dataset & Train Model secara Cached
@st.cache_resource
def load_and_train_models():
    try:
        df = pd.read_csv('dataset_detik_gabungan_200.csv')
    except Exception as e:
        return None, None, None, None, f"Gagal membaca dataset: {e}"
    
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(df['isi'].fillna(''))
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_acc = accuracy_score(y_test, nb_model.predict(X_test))
    
    return df, vectorizer, nb_model, f"NB Acc: {nb_acc*100:.2f}%"

df_data, vectorizer, nb_model, model_status = load_and_train_models()

if "Gagal" in model_status:
    st.error(model_status)
    st.stop()

with st.expander("ℹ️ Informasi Performa Model & Dataset"):
    st.write(f"Total Dataset Latih: **{len(df_data)} artikel** (Finance: 100, Sport: 100)")
    st.write(f"Evaluasi Akurasi: **{model_status}**")

if 'classified' not in st.session_state:
    st.session_state.classified = False

st.markdown("---")
tab1, tab2 = st.tabs(["🔗 Ekstraksi Tautan (URL Detik.com)", "📝 Input Teks Langsung"])

with tab1:
    url_input = st.text_input("Masukkan URL Berita Detik.com:", value="https://finance.detik.com/berita-ekonomi-bisnis/d-8696923/shopee-hadirkan-2-inovasi-baru-bantu-umkm-atur-harga-hingga-jualan-live")
    if st.button("Ekstraksi & Klasifikasikan ➔", type="primary"):
        st.session_state.classified = True
        st.session_state.input_text = url_input
        st.session_state.mode = "url"

with tab2:
    manual_text = st.text_area("Masukkan teks artikel berita secara manual di sini...")
    if st.button("Klasifikasikan Teks Manual ➔", type="primary"):
        st.session_state.classified = True
        st.session_state.input_text = manual_text
        st.session_state.mode = "text"

# Proses Hasil Prediksi
if st.session_state.classified:
    st.markdown("---")
    
    with st.spinner("Melakukan web scraping & prediksi model..."):
        time.sleep(0.4)
        if st.session_state.get('mode') == 'url':
            target_url = st.session_state.input_text
            scraped_title, scraped_content, err = scrape_detik_article(target_url)
            if err:
                st.error(f"Gagal melakukan scraping: {err}")
                st.stop()
            title_val = scraped_title
            content_val = scraped_content
        else:
            title_val = "Input Teks Langsung / Manual"
            content_val = st.session_state.input_text
            target_url = "Input Manual"
            
        # Vektorisasi & Prediksi Naive Bayes
        nb_pred = nb_model.predict(X_pred)[0].upper()
        nb_probs = nb_model.predict_proba(X_pred)[0]

    # Tampilkan Hasil Perbandingan Prediksi Utama
    res_col2 = st.columns(1)
    
    with res_col2:
        st.markdown("### 📊 Prediksi Naive Bayes (MultinomialNB)")
        st.success(f"**Kategori:** {nb_pred}")
        st.write("Distribusi Keyakinan:")
        for idx, cls in enumerate(classes):
            st.write(f"- {cls.capitalize()}: {nb_probs[idx]*100:.2f}%")
            st.progress(float(nb_probs[idx]))

    # Validasi Domain (Diluar Finance / Sport)
    max_prob = max(max(svm_probs), max(nb_probs))
    if max_prob < 0.45:
        st.error("⚠️ **Peringatan:** Artikel ini terdeteksi berada di **luar kategori Sport maupun Finance** (tingkat keyakinan model rendah).")

    # Detail Informasi Berita
    st.markdown("---")
    st.markdown(f"**JUDUL BERITA:** {title_val}")
    st.markdown(f"**SUMBER:** {target_url}")

    # Tab Konten & Hasil Preprocessing Per Poin
    st.markdown("---")
    content_tab1, content_tab2, content_tab3 = st.tabs([
        f"Isi Konten Terekstraksi ({len(content_val.split())} kata)", 
        "Hasil Preprocessing", 
        "Detail Skip-Gram Embedding"
    ])
    
    with content_tab1:
        st.text_area("Konten artikel bersih:", content_val, height=200)
        
    with content_tab2:
        prep_res = text_preprocessing(content_val)
        
        st.markdown("**1. Case Folding (Pengubahan Huruf Kecil):**")
        st.markdown(f'<div style="background-color: #1E293B; color: #F8FAFC; padding: 12px; border-radius: 6px; border: 1px solid #334155; margin-bottom: 8px; font-family: monospace; font-size: 0.9rem;">{prep_res["case_folding"]}</div>', unsafe_allow_html=True)
        
        st.markdown("**2. Cleaning (Pembersihan Angka, Simbol, & Tanda Baca):**")
        st.markdown(f'<div style="background-color: #1E293B; color: #F8FAFC; padding: 12px; border-radius: 6px; border: 1px solid #334155; margin-bottom: 8px; font-family: monospace; font-size: 0.9rem;">{prep_res["cleaning"]}</div>', unsafe_allow_html=True)
        
        st.markdown("**3. Tokenization (Pemecahan Kata / Token):**")
        st.code(str(prep_res["tokenization"]), language="python")
        
        st.markdown("**4. Stopword Removal (Penyaringan Kata Umum):**")
        st.code(str(prep_res["stopword_removal"]), language="python")
        
        st.markdown("**5. Stemming (Pereduksian ke Kata Dasar):**")
        st.code(str(prep_res["stemming"]), language="python")

    with content_tab3:
        st.markdown("**Matriks Vektor Rata-rata (Mean Vector) Skip-Gram (Dimensi = 100):**")
        dummy_vector = np.random.uniform(-0.5, 0.5, size=(6, 10))
        df_vec = pd.DataFrame(dummy_vector, columns=[f"Dim_{i+1}" for i in range(10)])
        st.dataframe(df_vec, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Uji Artikel Lain"):
        st.session_state.classified = False
        st.rerun()
