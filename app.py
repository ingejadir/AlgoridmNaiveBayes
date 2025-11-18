import streamlit as st
import pandas as pd
import arff
import io
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# --- Título principal ---
st.title("🔍 Predicción con Naive Bayes (Archivo .ARFF)")

# --- Subida del archivo ---
archivo = st.file_uploader("📂 Sube un archivo .arff", type=["arff"])

if archivo is not None:
    try:
        # --- Leer el archivo ARFF (decodificar bytes a texto) ---
        data = arff.load(io.StringIO(archivo.getvalue().decode('utf-8')))
        df = pd.DataFrame(data["data"], columns=[a[0] for a in data["attributes"]])

        # --- Mostrar vista previa ---
        st.subheader("📊 Vista previa de los datos")
        st.dataframe(df.head())

        # --- Seleccionar la columna objetivo ---
        columnas = list(df.columns)
        objetivo = st.selectbox("🎯 Selecciona la columna de clase (variable objetivo)", columnas)

        if objetivo:
            X = df.drop(columns=[objetivo])
            y = df[objetivo]

            # --- Convertir categorías a números si es necesario ---
            X = pd.get_dummies(X)
            y = pd.factorize(y)[0]

            # --- Dividir datos ---
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # --- Entrenar modelo Naive Bayes ---
            modelo = GaussianNB()
            modelo.fit(X_train, y_train)

            # --- Predicciones ---
            y_pred = modelo.predict(X_test)
            probabilidades = modelo.predict_proba(X_test)

            # --- Mostrar resultados ---
            st.subheader("✅ Resultados del modelo")
            st.write("**Exactitud (accuracy):**", round(accuracy_score(y_test, y_pred), 3))
            st.text("Reporte de clasificación:")
            st.text(classification_report(y_test, y_pred))

            # --- Mostrar probabilidades ---
            st.subheader("📈 Probabilidades por clase")
            clases = [f"Clase {c}" for c in range(probabilidades.shape[1])]
            prob_df = pd.DataFrame(probabilidades, columns=clases)
            st.dataframe(prob_df.head())

            # --- Combinar predicciones con las probabilidades ---
            resultado_final = pd.concat([
                pd.DataFrame(X_test.reset_index(drop=True)),
                pd.Series(y_pred, name="Predicción"),
                prob_df
            ], axis=1)

            st.subheader("🧾 Predicciones con probabilidades")
            st.dataframe(resultado_final.head())

    except Exception as e:
        st.error(f"⚠️ Error al leer el archivo: {e}")

else:
    st.info("👆 Sube un archivo .arff para comenzar.")
