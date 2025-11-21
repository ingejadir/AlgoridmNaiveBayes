# app.py
import streamlit as st
import pandas as pd
import arff
import io
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import base64
from pandas.api import types as ptypes

st.set_page_config(page_title="Dashboard Contagios - Visual", layout="wide")
st.title(" Dashboard Visual - Contagios (Simple)")

uploaded = st.file_uploader("Sube un archivo .arff o .csv", type=["arff", "csv"])

# Colores fijos: negativo (blue) y positivo (red)
COLOR_NEG = "#1f77b4"
COLOR_POS = "#d62728"

def leer_archivo(uploaded):
    if uploaded.name.lower().endswith(".arff"):
        raw = uploaded.getvalue().decode("utf-8")
        data = arff.load(io.StringIO(raw))
        df = pd.DataFrame(data["data"], columns=[a[0] for a in data["attributes"]])
    else:
        df = pd.read_csv(uploaded)
    return df

def to_str(x): 
    return str(x) if pd.notna(x) else "NaN"

def make_download_link(df, filename="subset.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f"data:file/csv;base64,{b64}"

if uploaded is None:
    st.info("Carga un archivo .arff o .csv para comenzar. La app mostrará solo visualizaciones sencillas.")
    st.stop()

# ------ Leer y mostrar preview ------
try:
    df = leer_archivo(uploaded)
except Exception as e:
    st.error(f"Error leyendo archivo: {e}")
    st.stop()

st.markdown("##  Vista previa de los datos")
st.dataframe(df.head())

# Seleccionar variable objetivo
st.markdown("###  Selecciona la columna objetivo (contagiado / no contagiado)")
objetivo = st.selectbox("Variable objetivo", df.columns)

if not objetivo:
    st.stop()

# Mostrar conteo original
raw_counts = df[objetivo].value_counts(dropna=False)
st.write("Conteo por valor (raw):")
st.write(raw_counts)

# Si hay más de 2 clases, permitir elegir cuáles usar para binarizar
unique_vals = df[objetivo].dropna().unique().tolist()
unique_vals_str = [to_str(v) for v in unique_vals]

if len(unique_vals) > 2:
    st.warning("Tu variable objetivo tiene más de 2 clases. Elige 2 clases para tratar como 'No contagiado' y 'Contagiado'.")
    chosen_two = st.multiselect("Selecciona exactamente 2 valores (orden: primero = No, segundo = Sí)", options=unique_vals_str, default=unique_vals_str[:2])
    if len(chosen_two) != 2:
        st.info("Selecciona exactamente 2 valores para continuar con la visualización binaria.")
        st.stop()
    clase_no, clase_si = chosen_two[0], chosen_two[1]
else:
    # si hay 2 o 1 clases, preseleccionar
    vals = unique_vals_str
    if len(vals) == 1:
        # único valor: lo tratamos como "No contagiado" y creamos un valor "Otro" vacío
        clase_no = vals[0]
        clase_si = "OTRO"
        st.info(f"Solo se detectó un valor en la columna objetivo: {vals[0]}. Puedes interpretar como No contagiado o subir otro dataset.")
    else:
        # dos valores: intentar detectar cuál es "positivo"
        v0, v1 = vals[0], vals[1]
        # heurística: si alguno es 1/'1'/'si' etc, ese será pos
        pos_keywords = {"1","'1'","si","sí","yes","true","positivo","infectado","pos"}
        if str(v0).strip().lower() in pos_keywords:
            clase_si, clase_no = v0, v1
        elif str(v1).strip().lower() in pos_keywords:
            clase_si, clase_no = v1, v0
        else:
            # por defecto: el segundo como positivo (solo por conveniencia)
            clase_no, clase_si = v0, v1
        st.write(f"Mapeo detectado: No contagiado = **{clase_no}**, Contagiado = **{clase_si}**")

# Permitir que el usuario ajuste manualmente qué valor es "Contagiado"
st.markdown("**Ajusta manualmente** si el mapeo anterior no coincide:")
colA, colB = st.columns(2)
with colA:
    sel_no = st.selectbox("Valor que representarás como NO contagiado", options=[to_str(x) for x in unique_vals], index=0 if clase_no in unique_vals_str else 0)
with colB:
    # si el usuario seleccionó manualmente algo no en unique_vals (caso OTRO), mantener
    sel_si = st.selectbox("Valor que representarás como CONTAGIADO", options=[to_str(x) for x in unique_vals], index=1 if clase_si in unique_vals_str else (0 if len(unique_vals)>1 else 0))

# convertir en columna binaria 'contagiado' (True/False)
def is_pos(val, positivo_str):
    # comparamos como strings insensibles a mayúsculas y espacios
    return to_str(val).strip().lower() == to_str(positivo_str).strip().lower()

df["_contagiado_bin"] = df[objetivo].apply(lambda x: is_pos(x, sel_si))

# Métricas simples
total = len(df)
n_pos = int(df["_contagiado_bin"].sum())
n_neg = total - n_pos
pct_pos = n_pos / total * 100 if total>0 else 0

k1, k2, k3 = st.columns(3)
k1.metric("Total registros", f"{total}")
k2.metric("Contagiados", f"{n_pos}")
k3.metric("Porcentaje contagiados", f"{pct_pos:.1f}%")

# ==== Gráfico barras: Contagiados vs No contagiados (2 colores) ====
st.markdown("##  Contagiados vs No contagiados (2 colores)")
fig, ax = plt.subplots(figsize=(6,4))
counts = [n_neg, n_pos]
labels = ["No contagiado", "Contagiado"]
bars = ax.bar(labels, counts, color=[COLOR_NEG, COLOR_POS])
# anotar
for bar in bars:
    h = bar.get_height()
    ax.annotate(f"{int(h)}", xy=(bar.get_x()+bar.get_width()/2, h), xytext=(0,5), textcoords="offset points", ha="center")
ax.set_ylabel("Cantidad de registros")
ax.set_title("Distribución binaria por clase")
st.pyplot(fig)

# ==== Pie chart con porcentajes ====
st.markdown("##  Porcentaje por clase")
fig2, ax2 = plt.subplots(figsize=(5,4))
ax2.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90, colors=[COLOR_NEG, COLOR_POS])
ax2.axis("equal")
st.pyplot(fig2)

# ==== Visualización por características: elegir 1 o 2 variables ====
st.markdown("##  Visualiza variables (elige 1 o 2 variables para ver cómo se distribuyen según contagio)")
cols_available = [c for c in df.columns if c != objetivo and c != "_contagiado_bin"]
sel_feats = st.multiselect("Selecciona 1 o 2 variables", options=cols_available, max_selections=2)

if sel_feats:
    subset = df[sel_feats + ["_contagiado_bin", objetivo]].copy()
    st.markdown("#### Vista rápida de los datos seleccionados")
    st.dataframe(subset.head(30))

    # 1 variable seleccionada
    if len(sel_feats) == 1:
        feat = sel_feats[0]
        st.markdown(f"### Visualización de `{feat}` por estado de contagio")

        if ptypes.is_numeric_dtype(df[feat]):
            # Histogramas superpuestos (dos colores)
            fig3, ax3 = plt.subplots(figsize=(7,4))
            data_pos = df[df["_contagiado_bin"]][feat].dropna()
            data_neg = df[~df["_contagiado_bin"]][feat].dropna()
            # Si hay muy pocas observaciones, usar rug/jitter no histogram
            if len(data_pos) < 2 or len(data_neg) < 2:
                ax3.plot(data_neg, np.zeros_like(data_neg)-0.02, '|', markersize=10, label="No contagiado")
                ax3.plot(data_pos, np.ones_like(data_pos)*0.02, '|', markersize=10, label="Contagiado")
            else:
                ax3.hist([data_neg, data_pos], bins=20, label=["No contagiado","Contagiado"], stacked=False, alpha=0.6, color=[COLOR_NEG, COLOR_POS])
            ax3.legend()
            ax3.set_xlabel(feat)
            ax3.set_ylabel("Frecuencia")
            st.pyplot(fig3)

        else:
            # Categórica: barras agrupadas por clase (dos colores)
            cross = pd.crosstab(df[feat].astype(str), df["_contagiado_bin"])
            # asegurar columnas False, True presentes
            if False not in cross.columns: cross[False] = 0
            if True not in cross.columns: cross[True] = 0
            cross = cross.sort_values(by=False, ascending=False)
            fig4, ax4 = plt.subplots(figsize=(8,4))
            ind = np.arange(len(cross))
            width = 0.4
            ax4.bar(ind - width/2, cross[False].values, width, label="No contagiado", color=COLOR_NEG)
            ax4.bar(ind + width/2, cross[True].values, width, label="Contagiado", color=COLOR_POS)
            ax4.set_xticks(ind)
            ax4.set_xticklabels(cross.index, rotation=45, ha="right")
            ax4.set_ylabel("Cantidad")
            ax4.legend()
            st.pyplot(fig4)

    # 2 variables seleccionadas
    else:
        f1, f2 = sel_feats
        st.markdown(f"### Visualización de `{f1}` vs `{f2}` (coloreado por contagio)")
        # caso: ambos num -> scatter
        if ptypes.is_numeric_dtype(df[f1]) and ptypes.is_numeric_dtype(df[f2]):
            fig5, ax5 = plt.subplots(figsize=(7,5))
            neg = df[~df["_contagiado_bin"]]
            pos = df[df["_contagiado_bin"]]
            ax5.scatter(neg[f1], neg[f2], alpha=0.7, label="No contagiado", color=COLOR_NEG, s=20)
            ax5.scatter(pos[f1], pos[f2], alpha=0.7, label="Contagiado", color=COLOR_POS, s=20)
            ax5.set_xlabel(f1); ax5.set_ylabel(f2)
            ax5.legend()
            st.pyplot(fig5)
        # caso: uno num y otro categ -> jitter categorical axis
        elif ptypes.is_numeric_dtype(df[f1]) and not ptypes.is_numeric_dtype(df[f2]):
            num = f1; cat = f2
            fig6, ax6 = plt.subplots(figsize=(8,5))
            # map categories to numbers
            cats = df[cat].astype(str).unique()
            cat_map = {c:i for i,c in enumerate(cats)}
            for lab, dfg in df.groupby("_contagiado_bin"):
                yvals = dfg[cat].astype(str).map(cat_map).values + np.random.normal(scale=0.08, size=len(dfg))
                xvals = dfg[num].values
                color = COLOR_POS if lab else COLOR_NEG
                label = "Contagiado" if lab else "No contagiado"
                ax6.scatter(xvals, yvals, alpha=0.7, label=label, color=color, s=18)
            ax6.set_yticks(list(cat_map.values()))
            ax6.set_yticklabels(list(cat_map.keys()))
            ax6.set_xlabel(num); ax6.set_ylabel(cat)
            ax6.legend()
            st.pyplot(fig6)
        elif not ptypes.is_numeric_dtype(df[f1]) and ptypes.is_numeric_dtype(df[f2]):
            # invertir roles
            num = f2; cat = f1
            fig7, ax7 = plt.subplots(figsize=(8,5))
            cats = df[cat].astype(str).unique()
            cat_map = {c:i for i,c in enumerate(cats)}
            for lab, dfg in df.groupby("_contagiado_bin"):
                yvals = dfg[cat].astype(str).map(cat_map).values + np.random.normal(scale=0.08, size=len(dfg))
                xvals = dfg[num].values
                color = COLOR_POS if lab else COLOR_NEG
                label = "Contagiado" if lab else "No contagiado"
                ax7.scatter(xvals, yvals, alpha=0.7, label=label, color=color, s=18)
            ax7.set_yticks(list(cat_map.values()))
            ax7.set_yticklabels(list(cat_map.keys()))
            ax7.set_xlabel(num); ax7.set_ylabel(cat)
            ax7.legend()
            st.pyplot(fig7)
        else:
            # ambos categóricos -> gráfico de mosaic-like (heatmap crosstab)
            cross2 = pd.crosstab(df[f1].astype(str), df[f2].astype(str))
            fig8, ax8 = plt.subplots(figsize=(8,6))
            # Mostramos conteos totales por combinación y coloreamos por proporción de contagiados
            # Construimos tabla con % contagiados por combinación
            combos = df.groupby([f1, f2])["_contagiado_bin"].mean().unstack(fill_value=0)
            im = ax8.imshow(combos.values, aspect="auto")
            ax8.set_xticks(np.arange(len(combos.columns)))
            ax8.set_yticks(np.arange(len(combos.index)))
            ax8.set_xticklabels(combos.columns, rotation=45, ha="right")
            ax8.set_yticklabels(combos.index)
            fig8.colorbar(im, ax=ax8, label="Proporción contagiados (0-1)")
            ax8.set_title("Proporción de contagiados por combinación")
            st.pyplot(fig8)

# Botón para descargar subset con la columna binaria (por si el usuario quiere)
st.markdown("---")
st.markdown("### ⤓ Descargar datos")
csv_url = make_download_link(df.drop(columns=[objective for objective in ["_contagiado_bin"] if False], errors='ignore'))
st.markdown(f"[⬇ Descargar dataset con columna binaria ' _contagiado_bin' (CSV)]({csv_url})", unsafe_allow_html=True)

st.success("Visualizaciones generadas. Elige variables para explorar más.")
