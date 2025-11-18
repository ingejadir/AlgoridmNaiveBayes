# 📌 Predicción con Naive Bayes (Archivo .ARFF)

Esta aplicación permite **cargar un archivo .ARFF**, procesar los datos y realizar predicciones utilizando el algoritmo **Naive Bayes**. Está desarrollada con **Streamlit** para ofrecer una interfaz interactiva y sencilla.

***

## 🚀 Características principales

*   **Carga de archivos .ARFF**.
*   **Vista previa de los datos** en formato tabla.
*   **Selección de la columna objetivo** (variable de clase).
*   **Preprocesamiento automático**:
    *   Conversión de variables categóricas a numéricas.
*   **Entrenamiento del modelo Naive Bayes**.
*   **Evaluación del modelo**:
    *   Exactitud (accuracy).
    *   Reporte de clasificación.
*   **Visualización de probabilidades por clase**.
*   **Tabla combinada** con predicciones y probabilidades.

***

## 🛠️ Tecnologías utilizadas

*   Python 3
*   Streamlit
*   Pandas
*   scikit-learn
*   liac-arff

***

## 📂 Estructura del proyecto

    ├── app.py        # Código principal de la aplicación
    ├── requirements.txt  # Dependencias del proyecto

***

## ⚙️ Instalación y ejecución

1.  **Clona el repositorio**:
    ```bash
    git clone https://github.com/usuario/naive-bayes-arff.git
    cd naive-bayes-arff
    ```

2.  **Crea un entorno virtual (opcional)**:
    ```bash
    python -m venv venv
    source venv/bin/activate   # En Linux/Mac
    venv\\Scripts\\activate    # En Windows
    ```

3.  **Instala las dependencias**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ejecuta la aplicación**:
    ```bash
    streamlit run app.py
    ```

***

## 📥 Uso de la aplicación

1.  Sube un archivo **.ARFF** desde la interfaz.
2.  Visualiza la **vista previa de los datos**.
3.  Selecciona la **columna objetivo**.
4.  Observa:
    *   **Exactitud del modelo**.
    *   **Reporte de clasificación**.
    *   **Probabilidades por clase**.
    *   **Predicciones con probabilidades**.

***

## ✅ Ejemplo de archivo `.ARFF`

Puedes usar datasets de UCI Machine Learning Repository.

***

## 🔍 Notas

*   El modelo utiliza **Gaussian Naive Bayes**.
*   El tamaño del conjunto de prueba es **30** del total.
*   Se realiza **codificación automática** de variables categóricas.
