import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

st.set_page_config(page_title="KNN Classification", layout="wide")

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("KNN Classification - Data Kesehatan Pasien")
st.markdown("Klasifikasi kondisi pasien berdasarkan **tinggi badan** dan **berat badan** menggunakan algoritma K-Nearest Neighbor.")
st.divider()

# ─── Sidebar: Parameter ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("Pengaturan Model")
    k_value = st.slider("Jumlah K (tetangga)", min_value=1, max_value=20, value=5)
    metric  = st.selectbox("Metrik Jarak", ["minkowski", "euclidean", "manhattan"])
    test_size = st.slider("Ukuran Data Test (%)", min_value=10, max_value=40, value=20, step=5)
    n_data  = st.slider("Jumlah Data", min_value=100, max_value=500, value=200, step=50)
    seed    = st.number_input("Random Seed", value=42)

# ─── Generate & Cache Dataset ─────────────────────────────────────────────────
@st.cache_data
def generate_data(n, seed):
    np.random.seed(seed)
    tinggi = np.concatenate([
        np.random.normal(170, 5, n // 3),
        np.random.normal(160, 6, n // 3),
        np.random.normal(155, 7, n // 3),
    ])
    berat = np.concatenate([
        np.random.normal(60, 5, n // 3),
        np.random.normal(75, 7, n // 3),
        np.random.normal(90, 8, n // 3),
    ])
    label = [0]*(n//3) + [1]*(n//3) + [2]*(n//3)
    df = pd.DataFrame({"tinggi_badan": tinggi, "berat_badan": berat, "kondisi": label})
    df.to_csv("data_kesehatan.csv", index=False)
    return df

df = generate_data(n_data, int(seed))

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Dataset", "Hasil Model", "Prediksi Data Baru"])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1: Dataset
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Preview Dataset")
        st.dataframe(df.head(20), use_container_width=True)
    with col2:
        st.subheader("Ringkasan")
        st.metric("Total Data", len(df))
        dist = df["kondisi"].value_counts().sort_index()
        nama = {0: "Sehat", 1: "Berisiko", 2: "Sakit"}
        for k, v in dist.items():
            st.metric(nama[k], v)

    st.subheader("Distribusi Data")
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#4CAF50", "#FF9800", "#F44336"]
    for kls, warna in zip([0, 1, 2], colors):
        mask = df["kondisi"] == kls
        ax.scatter(df.loc[mask, "tinggi_badan"], df.loc[mask, "berat_badan"],
                   c=warna, label=nama[kls], alpha=0.6, edgecolors="k", linewidths=0.4)
    ax.set_xlabel("Tinggi Badan (cm)")
    ax.set_ylabel("Berat Badan (kg)")
    ax.set_title("Scatter Plot Dataset")
    ax.legend()
    st.pyplot(fig)
    plt.close()

# ────────────────────────────────────────────────────────────────────────────
# TAB 2: Hasil Model
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    X = df[["tinggi_badan", "berat_badan"]].values
    y = df["kondisi"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=k_value, metric=metric)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    # simpan ke session state untuk tab 3
    st.session_state["model"]  = model
    st.session_state["scaler"] = scaler

    # Metrik
    acc = accuracy_score(y_test, y_pred)
    col1, col2, col3 = st.columns(3)
    col1.metric("Akurasi", f"{acc:.2%}")
    col2.metric("K", k_value)
    col3.metric("Metrik Jarak", metric.capitalize())

    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred,
                                   target_names=["Sehat","Berisiko","Sakit"],
                                   output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(2), use_container_width=True)

    # Confusion Matrix + Decision Boundary
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(3)); ax.set_xticklabels(["Sehat","Berisiko","Sakit"])
        ax.set_yticks(range(3)); ax.set_yticklabels(["Sehat","Berisiko","Sakit"])
        ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i,j], ha="center", va="center", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.subheader("Decision Boundary")
        x_min, x_max = X_test_s[:,0].min()-1, X_test_s[:,0].max()+1
        y_min, y_max = X_test_s[:,1].min()-1, X_test_s[:,1].max()+1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                              np.linspace(y_min, y_max, 200))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.contourf(xx, yy, Z, alpha=0.2, cmap="RdYlGn_r")
        colors = ["#4CAF50", "#FF9800", "#F44336"]
        for kls, warna, label_n in zip([0,1,2], colors, ["Sehat","Berisiko","Sakit"]):
            mask = y_test == kls
            ax.scatter(X_test_s[mask,0], X_test_s[mask,1],
                       c=warna, edgecolors="k", label=label_n, s=40, linewidths=0.4)
        ax.set_xlabel("Tinggi (normalized)"); ax.set_ylabel("Berat (normalized)")
        ax.legend(fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ────────────────────────────────────────────────────────────────────────────
# TAB 3: Prediksi Data Baru
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Input Data Pasien Baru")
    st.markdown("Masukkan tinggi dan berat badan pasien untuk diprediksi kondisinya.")

    col1, col2 = st.columns(2)
    with col1:
        tinggi_input = st.number_input("Tinggi Badan (cm)", min_value=100.0, max_value=220.0, value=165.0, step=0.5)
    with col2:
        berat_input  = st.number_input("Berat Badan (kg)",  min_value=30.0,  max_value=150.0, value=70.0,  step=0.5)

    if st.button("Prediksi", type="primary"):
        if "model" not in st.session_state:
            st.warning("Jalankan tab Hasil Model terlebih dahulu.")
        else:
            data_baru = np.array([[tinggi_input, berat_input]])
            data_scaled = st.session_state["scaler"].transform(data_baru)
            hasil = st.session_state["model"].predict(data_scaled)[0]
            proba = st.session_state["model"].predict_proba(data_scaled)[0]

            nama_hasil = {0: "Sehat", 1: "Berisiko", 2: "Sakit"}
            warna_hasil = {0: "green", 1: "orange", 2: "red"}

            st.divider()
            st.markdown(f"### Hasil Prediksi")
            st.markdown(
                f"<h2 style='color:{warna_hasil[hasil]}'>{nama_hasil[hasil]}</h2>",
                unsafe_allow_html=True
            )

            st.markdown("**Probabilitas per Kelas:**")
            for i, (nm, pb) in enumerate(zip(["Sehat","Berisiko","Sakit"], proba)):
                st.progress(float(pb), text=f"{nm}: {pb:.1%}")