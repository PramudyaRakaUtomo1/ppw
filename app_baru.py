import time
import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import re
from datetime import datetime
from gensim.models import Word2Vec

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Klasifikasi Berita SVM Detik.com",
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
        background-color: #1E293B;
        color: #F8FAFC;
        padding: 12px;
        border-radius: 6px;
        border: 1px solid #334155;
        margin-bottom: 8px;
        font-family: monospace;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Sistem Klasifikasi & Ekstraksi Berita (Support Vector Machine)</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Dilengkapi Web Scraping Detik.com, Preprocessing, TF-IDF, Word Embedding Skip-gram (Gensim), dan Klasifikasi SVM.</p>', unsafe_allow_html=True)

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

# Fungsi Preprocessing Teks Lengkap
def text_preprocessing(text):
    case_folded = text.lower()
    cleaned = re.sub(r'[^a-z\s]', '', case_folded)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    tokens = cleaned.split()
    
    stopwords_id = {
        'yang', 'untuk', 'dan', 'di', 'pada', 'ke', 'para', 'namun', 'menurut', 
        'antara', 'dia', 'dua', 'ia', 'seperti', 'jika', 'sehingga', 'kembali', 
        'dari', 'ini', 'itu', 'dengan', 'adalah', 'tersebut', 'oleh', 'saat', 
        'sebagai', 'kepada', 'karena', 'mereka', 'sebuah', 'lain', 'anda'
    }
    filtered_tokens = [w for w in tokens if w not in stopwords_id]
    
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
        "case_folded": case_folded,
        "cleaned": cleaned,
        "tokens": tokens,
        "stopwords_removed": filtered_tokens,
        "stemmed": stemmed_tokens,
        "tokens_str": ", ".join(tokens),
        "stopwords_removed_str": ", ".join(filtered_tokens),
        "stemmed_str": ", ".join(stemmed_tokens)
    }

# Load Dataset & Train Model SVM dengan Fallback Aman untuk Deployment
@st.cache_resource
def load_and_train_models():
    dataset_filename = 'dataset_detik_gabungan_200.csv'
    
    if os.path.exists(dataset_filename):
        try:
            df = pd.read_csv(dataset_filename)
        except Exception as e:
            return None, None, None, f"Gagal membaca dataset: {e}"
    else:
        # Fallback dataset dummy jika file CSV belum di-upload ke Cloud GitHub
        data_dummy = {
            'isi': [
                "inflasi bank sentral suku bunga saham ekonomi investasi bisnis rupiah pasar modal",
                "pertumbuhan ekonomi kuartal ini mencatat kenaikan signifikan didorong sektor keuangan",
                "menteri keuangan mengumumkan kebijakan fiskal baru untuk memperkuat investasi bisnis",
                "pertandingan sepak bola liga Champions tadi malam berlangsung sengit dan dramatis",
                "klub sepak bola nasional melakukan latihan intensif menghadapi turnamen piala asia",
                "atlet bulu tangkis Indonesia sukses meraih gelar juara setelah bertanding di final"
            ],
            'label': ['FINANCE', 'FINANCE', 'FINANCE', 'SPORT', 'SPORT', 'SPORT']
        }
        df = pd.DataFrame(data_dummy)
    
    if 'isi' not in df.columns or 'label' not in df.columns:
        return None, None, None, "Format kolom dataset salah. Harus memiliki kolom 'isi' dan 'label'."

    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(df['isi'].fillna(''))
    y = df['label']
    
    # Tangani kasus data terlalu sedikit untuk train_test_split
    if len(df) < 5:
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    svm_model = SVC(kernel='linear', probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    svm_acc = accuracy_score(y_test, svm_model.predict(X_test))
    
    return df, vectorizer, svm_model, f"SVM Acc: {svm_acc*100:.2f}%"

df_data, vectorizer, svm_model, model_status = load_and_train_models()

if "Gagal" in model_status:
    st.error(model_status)
    st.stop()

with st.expander("ℹ️ Informasi Performa Model & Dataset"):
    st.write(f"Total Dataset Latih: **{len(df_data)} artikel**")
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
    
    with st.spinner("Melakukan web scraping, preprocessing, ekstraksi TF-IDF, Word2Vec Skip-gram, & prediksi model SVM..."):
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
            
        # 1. Jalankan Preprocessing detail
        prep_res = text_preprocessing(content_val)
        
        # 2. Latih Model Word2Vec Skip-gram berbasis hasil Preprocessing
        corpus_w2v = [prep_res["stemmed"]] 
        
        if len(prep_res["stemmed"]) > 0:
            w2v_model = Word2Vec(
                sentences=corpus_w2v, 
                vector_size=10, 
                window=5, 
                min_count=1, 
                sg=1,            # 1 menunjukkan arsitektur Skip-gram
                seed=42
            )
            
            word_vectors = []
            word_labels = []
            for word in set(prep_res["stemmed"]):
                if word in w2v_model.wv:
                    word_vectors.append(w2v_model.wv[word])
                    word_labels.append(word)
            
            if word_vectors:
                df_vec = pd.DataFrame(word_vectors, index=word_labels, columns=[f"Dim_{i+1}" for i in range(10)])
                mean_embedding = np.mean(word_vectors, axis=0).tolist()
            else:
                df_vec = pd.DataFrame(columns=[f"Dim_{i+1}" for i in range(10)])
                mean_embedding = [0.0] * 10
        else:
            df_vec = pd.DataFrame(columns=[f"Dim_{i+1}" for i in range(10)])
            mean_embedding = [0.0] * 10

        # 3. Vektorisasi TF-IDF & Prediksi SVM
        X_pred = vectorizer.transform([content_val])
        svm_pred = str(svm_model.predict(X_pred)[0]).upper()
        svm_probs = svm_model.predict_proba(X_pred)[0]
        classes = [str(c).upper() for c in svm_model.classes_]
        
        feature_names = vectorizer.get_feature_names_out()
        tfidf_array = X_pred.toarray()[0]
        non_zero_indices = np.nonzero(tfidf_array)[0]
        tfidf_features_list = [(feature_names[i], tfidf_array[i]) for i in non_zero_indices]
        tfidf_features_list = sorted(tfidf_features_list, key=lambda x: x[1], reverse=True)
        tfidf_str = ", ".join([f"{word}: {score:.4f}" for word, score in tfidf_features_list[:30]])

    # Tampilkan Hasil Prediksi Utama
    st.markdown("### 📊 Hasil Prediksi Support Vector Machine (SVM)")
    st.success(f"**Kategori:** {svm_pred}")
    st.write("Distribusi Keyakinan (Probability):")
    for idx, cls in enumerate(classes):
        st.write(f"- {cls.capitalize()}: {svm_probs[idx]*100:.2f}%")
        st.progress(float(svm_probs[idx]))

    max_prob = max(svm_probs)
    if max_prob < 0.45:
        st.warning("⚠️ **Peringatan:** Tingkat keyakinan model rendah.")

    # Detail Informasi Berita
    st.markdown("---")
    st.markdown(f"**JUDUL BERITA:** {title_val}")
    st.markdown(f"**SUMBER:** {target_url}")

    # Tab Konten & Hasil Preprocessing
    st.markdown("---")
    content_tab1, content_tab2, content_tab3 = st.tabs([
        f"Hasil Isi Artikel ({len(content_val.split())} kata)", 
        "Hasil Preprocessing", 
        "Hasil Word Embedding (Skip-gram) & TF-IDF"
    ])
    
    with content_tab1:
        st.text_area("Isi Artikel", content_val, height=200)
        
    with content_tab2:
        st.markdown("**1. Case Folding:**")
        st.markdown(f'<div class="preprocessing-box">{prep_res["case_folded"][:300]}...</div>', unsafe_allow_html=True)
        
        st.markdown("**2. Cleaning:**")
        st.markdown(f'<div class="preprocessing-box">{prep_res["cleaned"][:300]}...</div>', unsafe_allow_html=True)
        
        st.markdown("**3. Tokenization:**")
        st.code(prep_res["tokens_str"][:300] + "...", language="python")
        
        st.markdown("**4. Stopword Removal:**")
        st.code(prep_res["stopwords_removed_str"][:300] + "...", language="python")
        
        st.markdown("**5. Stemming:**")
        st.code(prep_res["stemmed_str"][:300] + "...", language="python")

    with content_tab3:
        st.markdown("**Matriks Vektor Skip-Gram Embedding Berbasis Token Preprocessing (Dimensi = 10):**")
        if not df_vec.empty:
            st.dataframe(df_vec, use_container_width=True)
        else:
            st.warning("Token terlalu sedikit untuk membentuk matriks Word2Vec.")
        
        st.markdown("**Bobot TF-IDF Topik Teratas:**")
        st.info(tfidf_str if tfidf_str else "Tidak ada fitur TF-IDF signifikan yang terekam.")

    # Fitur Download Hasil Analisis
    st.markdown("---")
    st.markdown("### 💾 Simpan & Unduh Hasil Analisis Lengkap")
    
    export_data = {
        "Waktu": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Judul_Berita": [title_val],
        "Sumber": [target_url],
        "Prediksi_SVM": [svm_pred],
        "Keyakinan": [f"{max(svm_probs)*100:.2f}%"],
        "Case_Folding": [prep_res["case_folded"]],
        "Cleaning": [prep_res["cleaned"]],
        "Tokenization": [prep_res["tokens_str"]],
        "Stopword_Removal": [prep_res["stopwords_removed_str"]],
        "Stemming": [prep_res["stemmed_str"]],
        "TF_IDF_Top_Features": [tfidf_str],
        "Embedding_Mean_Vector": [str(mean_embedding)]
    }
    df_export = pd.DataFrame(export_data)
    csv_bytes = df_export.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Unduh Hasil Lengkap (CSV)",
        data=csv_bytes,
        file_name=f"hasil_analisis_nlp_svm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        type="primary"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Uji Artikel Lain"):
        st.session_state.classified = False
        st.rerun()
