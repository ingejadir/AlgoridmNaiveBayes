import streamlit as st
import pandas as pd
import arff
import io
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from pandas.api import types as ptypes

# Colores fijos
COLOR_NEG = "#1f77b4"   # Azul = No contagiado
COLOR_POS = "#d62728"   # Rojo = Contagiado

st.set_page_config(page_title="Dashboard Contagios", layout="wide")
st.title("📊 Dashboard Visual + Naive Bayes (Opcional)")

# --------------------------------------------------
# Función para leer ARFF o CSV
# --------------------------------------------------
def leer_archivo(uploaded):
    if uploaded.name.lower().endswith(".arff"):
        raw = uploaded.getvalue().decode("utf-8")
        data = arff.load(io.StringIO(raw))
        df = pd.DataFrame(data["data"], columns=[a[0] for a in data["attributes"]])
    else:
        df = pd.read_csv(uploaded)
    return df

# Subida del archivo
uploaded = st.file_uploader("Sube un archivo .arff o .csv", type=["arff", "csv"])

if uploaded is None:
    st.info("Sube un archivo para comenzar.")
    st.stop()

df = leer_archivo(uploaded)

# --------------------------------------------------
# Vista previa
# --------------------------------------------------
st.markdown("## 📌 Vista previa de los datos")
st.dataframe(df.head())

# Variable objetivo detectada
objetivo = "clase"
NEG_VALUE = "Negativo"
POS_VALUE = "Positivo"

# --------------------------------------------------
# Crear la columna binaria
# --------------------------------------------------
def binarizar(val):
    val = str(val).strip().lower()
    return 1 if val == POS_VALUE.lower() else 0

df["_contagiado_bin"] = df[objetivo].apply(binarizar)

# --------------------------------------------------
# Métricas simples
# --------------------------------------------------
total = len(df)
n_pos = df["_contagiado_bin"].sum()
n_neg = total - n_pos
pct_pos = (n_pos / total) * 100 if total > 0 else 0

c1, c2, c3 = st.columns(3)
c1.metric("Total registros", total)
c2.metric("Contagiados (Positivo)", n_pos)
c3.metric("Porcentaje contagiados", f"{pct_pos:.1f}%")

# --------------------------------------------------
# Gráfico barras
# --------------------------------------------------
st.markdown("## 📊 Contagiados vs No contagiados")
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(["No contagiado", "Contagiado"], [n_neg, n_pos], color=[COLOR_NEG, COLOR_POS])

for bar in bars:
    h = bar.get_height()
    ax.annotate(str(int(h)), (bar.get_x() + bar.get_width()/2, h),
                ha="center", va="bottom", fontsize=12)

st.pyplot(fig)

# --------------------------------------------------
# Pie chart
# --------------------------------------------------
st.markdown("## 🥧 Porcentaje por clase")
fig2, ax2 = plt.subplots(figsize=(5, 4))
ax2.pie([n_neg, n_pos], labels=["No contagiado", "Contagiado"],
        autopct="%1.1f%%", startangle=90, colors=[COLOR_NEG, COLOR_POS])
ax2.axis("equal")
st.pyplot(fig2)

# --------------------------------------------------
# Visualiza 1 o 2 variables
# --------------------------------------------------
st.markdown("## 🔍 Visualiza 1 o 2 variables")
cols_available = [c for c in df.columns if c not in [objetivo, "_contagiado_bin"]]
sel_feats = st.multiselect("Selecciona 1 o 2 variables", options=cols_available, max_selections=2)

if sel_feats:

    st.dataframe(df[sel_feats + ["_contagiado_bin"]].head())

    # -------- 1 VARIABLE --------
    if len(sel_feats) == 1:

        feat = sel_feats[0]
        st.markdown(f"### 📌 Distribución de `{feat}`")

        if ptypes.is_numeric_dtype(df[feat]):
            fig3, ax3 = plt.subplots(figsize=(7, 4))
            ax3.hist(df[df["_contagiado_bin"] == 0][feat], bins=20, alpha=0.6, color=COLOR_NEG, label="No contagiado")
            ax3.hist(df[df["_contagiado_bin"] == 1][feat], bins=20, alpha=0.6, color=COLOR_POS, label="Contagiado")
            ax3.legend()
            st.pyplot(fig3)

        else:
            cross = pd.crosstab(df[feat].astype(str), df["_contagiado_bin"])
            if 0 not in cross.columns: cross[0] = 0
            if 1 not in cross.columns: cross[1] = 0
            cross = cross.sort_values(by=0, ascending=False)

            fig4, ax4 = plt.subplots(figsize=(8, 4))
            ind = np.arange(len(cross))
            width = 0.4

            ax4.bar(ind - width/2, cross[0], width, color=COLOR_NEG, label="No contagiado")
            ax4.bar(ind + width/2, cross[1], width, color=COLOR_POS, label="Contagiado")
            ax4.set_xticks(ind)
            ax4.set_xticklabels(cross.index, rotation=45)
            ax4.legend()
            st.pyplot(fig4)


    # --------------------------------------------------
    # -------- 2 VARIABLES → HEATMAP --------
    # --------------------------------------------------
    else:
        f1, f2 = sel_feats
        st.markdown(f"### 🔥 Heatmap: `{f1}` vs `{f2}` (proporción de contagiados)")

        df[f1] = df[f1].astype(str)
        df[f2] = df[f2].astype(str)

        tabla = pd.crosstab(
            df[f2], df[f1],
            values=df["_contagiado_bin"],
            aggfunc="mean"
        ).fillna(0)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(tabla, cmap="viridis")

        ax.set_xticks(np.arange(len(tabla.columns)))
        ax.set_xticklabels(tabla.columns, rotation=45, ha="right")

        ax.set_yticks(np.arange(len(tabla.index)))
        ax.set_yticklabels(tabla.index)

        ax.set_xlabel(f1)
        ax.set_ylabel(f2)
        ax.set_title("Proporción de contagiados por combinación")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Proporción contagiados (0-1)")

        st.pyplot(fig)

# --------------------------------------------------
# Naive Bayes opcional
# --------------------------------------------------
st.markdown("---")
usar_nb = st.checkbox("🔘 Activar predicción con Naive Bayes")

if usar_nb:

    st.markdown("## 🤖 Modelo Naive Bayes")

    features = [c for c in df.columns if c not in [objetivo, "_contagiado_bin"]]
    X = df[features].copy()
    y = df["_contagiado_bin"].copy()

    X = X.apply(lambda col: pd.factorize(col)[0] if col.dtype == "object" else col)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    st.metric("Exactitud (Accuracy)", f"{accuracy_score(y_test, y_pred):.2f}")

    st.markdown("### 📈 Distribución de probabilidades de contagio")

    fig6, ax6 = plt.subplots(figsize=(7, 4))

    prob_neg = y_proba[y_pred == 0]
    prob_pos = y_proba[y_pred == 1]

    ax6.hist(prob_neg, bins=20, color=COLOR_NEG, alpha=0.7, label="No contagiado")
    ax6.hist(prob_pos, bins=20, color=COLOR_POS, alpha=0.7, label="Contagiado")

    ax6.set_xlabel("Probabilidad de contagio")
    ax6.set_ylabel("Frecuencia")
    ax6.legend()

    st.pyplot(fig6)
