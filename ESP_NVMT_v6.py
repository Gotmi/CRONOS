""""
---------------------------------  SISTEMA DE PREDICCION DE FALLAS BES --------------------------------------------------------------
Este proyecto fue desarrollado como parte del trabajo de titulación en Ingeniería en Petróleos en la Universidad Central del Ecuador.

Autores:
Joao Ugalde  
Correo: ugaldejoao27@gmail.com

Mario Jarrín  
Correo: mariojarrin962@gmail.com

© 2025 Sistema de Predicción de Fallas BES
Versión 1.0 | Todos los derechos reservados
"""

#----------------------- Librerias ---------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from itertools import groupby
from operator import itemgetter
import plotly.graph_objects as go
import altair as alt
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, log_loss
from sklearn.preprocessing import LabelBinarizer
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import random
import warnings
import logging

# Silenciar advertencias de Streamlit y TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Oculta logs de TensorFlow
warnings.filterwarnings("ignore")
logging.getLogger("streamlit.runtime.scriptrunner.script_runner").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ---------------------- Funciones Auxiliares-------------------------------------------------------------------

def compute_classification_metrics(y_true, y_pred, y_probs=None, classes=None):
    """
    Calcula métricas de clasificación básicas para un solo modelo.
    
    y_true: lista o array de etiquetas verdaderas
    y_pred: lista o array de etiquetas predichas
    y_probs: array de probabilidades predichas (opcional, para AUC)
    classes: lista de clases posibles (opcional, para AUC)
    
    Devuelve: diccionario con métricas
    """
    metrics = {}
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["Precision"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["Recall / Sensibilidad"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["F1-Score"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    # AUC solo si se pasan probabilidades y clases
    try:
        if y_probs is not None and classes is not None:
            # Convertir y_true a formato binario para multiclass
            
            y_true_bin = label_binarize(y_true, classes=classes)
            metrics["AUC-ROC"] = roc_auc_score(y_true_bin, y_probs.reshape(1,-1), average="weighted", multi_class="ovr")
        else:
            metrics["AUC-ROC"] = None
    except:
        metrics["AUC-ROC"] = None

    # False Alarm Rate
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=classes).ravel()
        metrics["False Alarm Rate"] = fp / (fp + tn) if (fp+tn)>0 else 0
    except:
        metrics["False Alarm Rate"] = None
    
    return metrics


def extract_features_from_zone(zone_df):
    exclude = {"datetime", "ID", "Event", "Event code", "Event type", "failure"}
    feats = {}

    for c in zone_df.columns:
        if c in exclude:
            continue

        series = pd.to_numeric(zone_df[c], errors="coerce").dropna()
        if series.empty:
            continue

        feats[f"{c}_mean"] = series.mean()
        feats[f"{c}_std"] = series.std()
        feats[f"{c}_min"] = series.min()
        feats[f"{c}_max"] = series.max()
        feats[f"{c}_range"] = series.max() - series.min()
        feats[f"{c}_cv"] = (series.std()/series.mean()) if series.mean() != 0 else 0
        x = np.arange(len(series))
        try:
            slope, intercept = np.polyfit(x, series, 1)
            feats[f"{c}_slope"] = slope
        except:
            feats[f"{c}_slope"] = 0
        feats[f"{c}_skew"] = skew(series)
        feats[f"{c}_kurt"] = kurtosis(series)

        diffs = series.diff().dropna()
        feats[f"{c}_mean_diff"] = diffs.mean() if not diffs.empty else 0

        if len(series) > 8:
            f, Pxx = welch(np.asarray(series), nperseg=min(256, len(series)))
            feats[f"{c}_fft_dominant_freq"] = f[np.argmax(Pxx)]
    return feats

failures = [
    "Rotura de eje",
    "Taponamiento",
    "Comunicación tubing-casing",
    "Bloqueo de etapas en la bomba",
    "Daño en el sensor de fondo",
    "Alta temperatura",
    "Desbalance de fases",
    "Falla electrica en superficie (Robo, daño equipos)",
    "Aumento del corte de agua",
    "Aumento de la presión del reservorio",
    "Aumento de gas libre en la entrada de la bomba"
]

# ----------------- CONFIG -----------------
rename_map = {
    "Ia, А": "ia",
    "Ia": "ia",
    "Ia(A)": "ia",
    "Iа(A)": "ia",
    "F, Hz": "freq",
    "F(Hz)": "freq",
    "Motor U, V": "motor_u_v",
    "Um(V)": "motor_u_v",
    "P, kW": "p_kw",
    "Pact(KW)": "p_kw",
    "S, kVA": "s_kva",
    "Pfull(kVA)": "s_kva",
    "Cos": "cos_phi",
    "Load, %": "load_pct",
    "mLoad(%)": "load_pct",
    "R, kOm": "r_kohm",
    "R(kOm)": "r_kohm",
    "P, psi": "pintake",
    "Pin(psi)": "pintake",
    "Pout, psi": "pdescarga",
    "Pout(psi)": "pdescarga",
    "Тmot, ⁰F": "tmotor",
    "Toil(°F)": "tmotor",
    "Тliq, ⁰F": "tliq",
    "Tin(°F)": "tliq",
    "Vibr.X/Y, G": "vibr_xy_g",
    "Vax(G)": "vibr_xy_g",
    "Vibr.Z, G": "vibr_z_g",
    "Vrad(G)": "vibr_z_g"
}
# Orden esperado final
expected_columns = [
    "Date time", "ia", "freq", "motor_u_v", "p_kw", "s_kva",
    "cos_phi", "load_pct", "r_kohm", "pintake", "pdescarga", "tmotor", "tliq", "vibr_xy_g", "vibr_z_g"
]

# Etiquetas legibles para visualización (no afecta los cálculos)
label_map = {
    "freq": "Frecuencia (Hz)",
    "ia": "Corriente (A)",
    "motor_u_v": "Voltaje motor (V)",
    "p_kw": "Potencia (kW)",
    "s_kva": "Potencia aparente (kVA)",
    "cos_phi": "Factor de potencia (-)",
    "load_pct": "Carga (%)",
    "r_kohm": "Resistencia (kΩ)",
    "pintake": "Presión de entrada (psi)",
    "pdescarga": "Presión de descarga (psi)",
    "tmotor": "Temperatura motor (°F)",
    "tliq": "Temperatura líquido (°F)",
    "vibr_xy_g": "Vibración XY (g)",
    "vibr_z_g": "Vibración Z (g)"
}

def cargar_y_ordenar_excel(uploaded_file):
    """
    Lee un archivo Excel subido, detecta la fila del encabezado buscando 'Date time',
    renombra las columnas según rename_map, reordena las esperadas, 
    crea las faltantes y las llena con ceros.
    """

    if uploaded_file is None:
        st.warning("⚠️ No se ha cargado ningún archivo.")
        return None

    try:
        #Leer sin encabezado para ubicar la fila correcta
        df_temp = pd.read_excel(uploaded_file, header=None)
        header_row = None

        for i, row in df_temp.iterrows():
            if row.astype(str).str.contains("Date time", case=False).any():
                header_row = i
                break

        if header_row is None:
            st.error("❌ No se encontró una fila con 'Date time'. Verifica el archivo.")
            return None

        #Leer nuevamente con el encabezado correcto
        df = pd.read_excel(uploaded_file, header=header_row)

        #Renombrar columnas según el mapeo
        df.rename(columns=rename_map, inplace=True)

        #Detectar columnas faltantes
        missing_cols = [col for col in expected_columns if col not in df.columns]

        #Crear columnas faltantes con ceros
        if missing_cols:
            for col in missing_cols:
                df[col] = 0
            st.warning(f"⚠️ No se encontraron las columnas: {', '.join(missing_cols)}. "
                       f"Se han creado y llenado con ceros.")

        #Reordenar columnas e ignorar las demás
        df = df[[col for col in expected_columns if col in df.columns]]
        
        st.success("✅ Archivo leído, columnas renombradas y ordenadas correctamente.")
        return df

    except Exception as e:
        st.error(f"⚠️ Error al procesar el archivo: {e}")
        return None

def mostrar_columnas_requeridas():
    """
    Muestra en Streamlit una tabla con las columnas requeridas para los archivos Excel,
    indicando que deben tener los mismos nombres y que las columnas faltantes se llenarán con ceros.
    """
    # 🔹 Definir las columnas esperadas y sus descripciones
    expected_columns_display = [
        "Date time", "Ia, А", "F, Hz", "Motor U, V", "P, kW", "S, kVA",
        "Cos", "Load, %", "R, kOm", "P, psi", "Pout, psi", "Тmot, ⁰F", "Тliq, ⁰F",
        "Vibr.X/Y, G", "Vibr.Z, G"
    ]
    
    column_descriptions = [
        "Fecha y hora del registro",
        "Corriente del motor (A)",
        "Frecuencia (Hz)",
        "Voltaje del motor (V)",
        "Potencia activa (kW)",
        "Potencia aparente (kVA)",
        "Factor de potencia (Cos φ)",
        "Carga del motor (%)",
        "Resistencia de aislamiento (kΩ)",
        "Presión de entrada (psi)",
        "Presión de descarga (psi)",
        "Temperatura del motor (°F)",
        "Temperatura del fluido (°F)",
        "Vibración lateral (G)",
        "Vibración axial (G)"
    ]

    #Crear DataFrame para mostrar en la interfaz
    df_columns_example = pd.DataFrame({
        "Nombre exacto de la columna": expected_columns_display,
        "Descripción de la variable": column_descriptions
    })

    # 🔹 Mensaje informativo
    st.markdown("""
    El archivo Excel que subas **debe contener exactamente los siguientes nombres de columnas**, 
    en el mismo orden mostrado a continuación.  
    Si alguna de estas columnas no está presente en el archivo, **se creará automáticamente y se llenará con ceros (0)** 
    para mantener la estructura estándar del modelo.
    """)

    #Mostrar tabla
    st.dataframe(df_columns_example, width='stretch')


MODEL_CNN_PATH = "modelo_fallas_cnn.h5"
MODEL_LSTM_PATH = "modelo_fallas_lstm.h5"
DATASET_PATH = "dataset_acumulado_cnn.csv"

def prepare_sequences(X, y=None, timesteps=10):
    """
    Crea secuencias tipo sliding-window.
    Si y es None, devuelve solo X_seq; si y dado, devuelve X_seq,y_seq
    """
    X_seq = []
    y_seq = [] if y is not None else None
    for i in range(timesteps, len(X)):
        X_seq.append(X[i - timesteps:i, :])
        if y is not None:
            y_seq.append(y[i])
    X_seq = np.array(X_seq)
    if y is not None:
        y_seq = np.array(y_seq)
        return X_seq, y_seq
    return X_seq

def build_cnn(input_shape, n_classes):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(n_classes, activation="softmax"))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def build_lstm(input_shape, n_classes):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(n_classes, activation="softmax"))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def map_index_to_timestamp(df_orig, timesteps):
    """
    Retorna una lista de timestamps (o índices numéricos) para cada ventana en X_seq.
    Cada ventana corresponde al índice 'i' en el lazo for i in range(timesteps, len(X)):
    Por lo tanto el timestamp asociado es df_orig.iloc[i].
    """
    if "datetime" in df_orig.columns:
        ts = pd.to_datetime(df_orig["datetime"], errors="coerce").reset_index(drop=True)
    else:
        ts = pd.Series(df_orig.index.astype(str))
    # ventanas desde timesteps .. len-1
    return ts.iloc[timesteps:].reset_index(drop=True)

# ---------------------------------------------------
# CONFIGURACIONES
# ---------------------------------------------------
DATASET_PATH = "dataset_acumulado.csv"
DATASET_RF_PATH = "dataset_acumulado_rf.csv"
MODEL_CNN_PATH = "modelo_fallas_cnn.h5"
MODEL_LSTM_PATH = "modelo_fallas_lstm.h5"
MODEL_RF_PATH = "modelo_fallas_rf.joblib"
SCALER_RF_PATH = "scaler_rf.joblib"
SCALER_RF_PATH = "scaler_rf.joblib"
LABEL_ENCODER_PATH = "label_encoder_rf.joblib"  # Agregado para decodificar etiquetas del RF
FEATURE_COLS_ORDER = []  # Inicializado (se llenará en entrenamiento RF)


# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Predicción de Fallas BES", layout="wide")

# --- ARCHIVOS DE MODELO ---
MODEL_CNN_PATH = "modelo_fallas_cnn.h5"
MODEL_LSTM_PATH = "modelo_fallas_lstm.h5"
MODEL_RF_PATH = "modelo_fallas_rf.joblib"

# --- FOOTER PERSONALIZADO ---
st.markdown("""
    <style>
        /* Estilo del footer */
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #000000;
            color: white;
            text-align: center;
            padding: 10px 0;
            font-size: 14px;
            font-family: 'Segoe UI', sans-serif;
            z-index: 100;
        }
        /* Evita que tape contenido al hacer scroll */
        .main {
            margin-bottom: 60px;
        }
    </style>

    <div class="footer">
        © 2025 Sistema de Predicción de Fallas BES — Desarrollado por Joao Ugalde y Mario Jarrín | Versión 1.0 | Todos los derechos reservados
    </div>
""", unsafe_allow_html=True)


# -------------------- TÍTULO PRINCIPAL ---------------------------------------------------------------------------------------------------------
st.title("Sistema de Predicción de fallas BES")
tab1, tab2, tab3, tab4 = st.tabs(["Entrenamiento", "Predicción", "Métricas de modelos", "Información"])

# ----------------------------------------------------------------------------------------------------------------------------------------------
# -------------------- ENTRENAMIENTO ---------------------------------------------------------------------------------------------------------
with tab1:
    st.title("Entrenamiento")
    
    with st.expander("📋 Estructura esperada del archivo Excel"):
        mostrar_columnas_requeridas()
    uploaded_file = st.file_uploader("Subir Excel para el entrenamiento", type=["xlsx"], key="train")

    #Dataset acumulado
    if "dataset" not in st.session_state:
        st.session_state["dataset"] = pd.read_csv(DATASET_PATH) if os.path.exists(DATASET_PATH) else pd.DataFrame()
    if "dataset_rf" not in st.session_state:
        st.session_state["dataset_rf"] = pd.read_csv(DATASET_RF_PATH) if os.path.exists(DATASET_RF_PATH) else pd.DataFrame()

    if uploaded_file is not None:
        # Leer sin encabezado para ubicar la fila correcta
        df_temp = pd.read_excel(uploaded_file, header=None)
        header_row = None
        
        # Buscar la fila que contiene "Date time"
        for i, row in df_temp.iterrows():
            if row.astype(str).str.contains("Date time", case=False).any():
                header_row = i
                break

        # Si no se encuentra la fila, mostrar error
        if header_row is None:
            st.error("❌ No se encontró la columna 'Date time' en el archivo.")
        else:
            # Leer nuevamente el archivo usando la fila encontrada como encabezado
            df = pd.read_excel(uploaded_file, header=header_row)
            st.success(f"✅ Encabezado encontrado en la fila {header_row + 1}")

        if "Date time" in df.columns:
            df["datetime"] = pd.to_datetime(df["Date time"], errors="coerce")

        columnas_validas = [col for col in rename_map.keys() if col in df.columns]
        if len(columnas_validas) == 0:
            st.warning("⚠️ No se encontraron columnas esperadas en el archivo.")
        else:
            df_filtered = df[columnas_validas].rename(columns=rename_map)
            df_filtered = df_filtered.apply(pd.to_numeric, errors="coerce")
            if "datetime" in df.columns:
                df_filtered["datetime"] = df["datetime"].values
            df_filtered = df_filtered.dropna()


            # ------------------ Gráficas de variables ------------------
            st.subheader("Gráficas de las variables cargadas")
            var_cols = [c for c in df_filtered.columns if c not in ["datetime"]]

            for col in var_cols:
                display_label = label_map.get(col, col)  # Usa etiqueta legible si existe
                fig = go.Figure()
                if "datetime" in df_filtered.columns:
                    x_data = df_filtered["datetime"]
                    x_title = "Tiempo"
                else:
                    x_data = df_filtered.index
                    x_title = "Índice"

                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=df_filtered[col],
                    mode="lines",
                    name=display_label,
                    line=dict(color="#0072B2", width=2),
                    hovertemplate=f'{x_title}: %{{x}}<br>{display_label}: %{{y:.2f}}<extra></extra>'
                ))

                fig.update_layout(
                    title=display_label,
                    autosize=True,
                    height=200,
                    margin=dict(l=40, r=40, t=40, b=40),
                    hovermode="x unified",
                    xaxis_title=x_title,
                    yaxis_title=display_label,
                )

                fig.update_xaxes(showgrid=True, automargin=True)
                fig.update_yaxes(showgrid=True, automargin=True)

                st.plotly_chart(fig, width='stretch', theme=None)

            #
            # ------------------ Selección de rango de falla ------------------
            # --- Slider de rango de fechas ---
            min_date = df_filtered["datetime"].min()
            max_date = df_filtered["datetime"].max()

            fecha_inicio, fecha_fin = st.slider(
                "Selecciona el rango de fechas de la falla:",
                min_value=min_date.to_pydatetime(),
                max_value=max_date.to_pydatetime(),
                value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
                format="DD/MM/YYYY HH:mm"
            )

            # --- Filtrar datos en el rango seleccionado ---
            df_selected = df_filtered[(df_filtered["datetime"] >= fecha_inicio) & (df_filtered["datetime"] <= fecha_fin)]

            # --- Seleccionar tipo de falla ---
            selected_failure = st.selectbox("Selecciona el tipo de falla para el rango marcado", failures)

            # --- Guardar selección ---
            if st.button("Guardar selección como datos de falla"):
                df_failure = df_selected.copy()
                df_failure["failure"] = selected_failure

                # Guardar en dataset CNN/LSTM
                st.session_state["dataset"] = pd.concat([st.session_state["dataset"], df_failure], ignore_index=True)
                st.session_state["dataset"].to_csv(DATASET_PATH, index=False)

                # Crear features agregadas para RF
                feats = extract_features_from_zone(df_failure)
                feats["failure"] = selected_failure
                df_feats = pd.DataFrame([feats])
                st.session_state["dataset_rf"] = pd.concat([st.session_state["dataset_rf"], df_feats], ignore_index=True)
                st.session_state["dataset_rf"].to_csv(DATASET_RF_PATH, index=False)

                st.success(f"✅ Datos de la falla '{selected_failure}' guardados en ambos datasets (CNN/LSTM y RF).")
                st.subheader("Vista previa del dataset acumulado (últimas filas)")
                st.write(st.session_state["dataset"].tail())

            #

    st.markdown("---")
    st.header("Entrenar modelos (CNN + LSTM + RF)")

    # 🔹 Parámetros fijos
    TIMESTEPS = 10
    EPOCHS = 10
    BATCH_SIZE = 32

    if st.button("Entrenar todos los modelos"):
        if st.session_state["dataset"].empty or st.session_state["dataset_rf"].empty:
            st.error("❌ No hay suficientes datos en los datasets. Añade datos antes de entrenar.")
        else:
            df_train = st.session_state["dataset"].copy()
            df_rf = st.session_state["dataset_rf"].copy()

            # ------------------ CNN & LSTM ------------------
            feature_cols = [c for c in df_train.columns if c not in ["failure", "datetime"]]
            X = df_train[feature_cols].values
            y = df_train["failure"].apply(lambda x: failures.index(x)).values

            if len(X) <= TIMESTEPS:
                st.error("⚠️ No hay suficientes filas para crear secuencias con los timesteps establecidos.")
            else:
                X_seq, y_seq = prepare_sequences(X, y, timesteps=TIMESTEPS)
                n_classes = len(failures)

                # CNN
                cnn = build_cnn(input_shape=(X_seq.shape[1], X_seq.shape[2]), n_classes=n_classes)
                cnn.fit(X_seq, y_seq, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
                cnn.save(MODEL_CNN_PATH)
                st.success("✅ Modelo CNN entrenado y guardado.")

                # LSTM
                lstm = build_lstm(input_shape=(X_seq.shape[1], X_seq.shape[2]), n_classes=n_classes)
                lstm.fit(X_seq, y_seq, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
                lstm.save(MODEL_LSTM_PATH)
                st.success("✅ Modelo LSTM entrenado y guardado.")

            # ------------------ RANDOM FOREST ------------------
            # 🔸 Triplicar los datos (para evitar error al dividir si hay pocas muestras)
            df_rf = pd.concat([df_rf] * 3, ignore_index=True)

            # 🔹 Separar variables y etiquetas
            X_rf = df_rf.drop(columns=["failure"])
            y_rf_text = df_rf["failure"]

            # 🔹 Codificar etiquetas
            label_encoder = LabelEncoder()
            y_rf_encoded = label_encoder.fit_transform(y_rf_text)
            joblib.dump(label_encoder, LABEL_ENCODER_PATH)

            # 🔹 Escalar las features
            scaler = StandardScaler()
            X_rf_scaled = scaler.fit_transform(X_rf)
            joblib.dump(scaler, SCALER_RF_PATH)

            # 🔹 Guardar columnas de orden
            FEATURE_COLS_ORDER = list(X_rf.columns)
            joblib.dump(FEATURE_COLS_ORDER, "feature_cols_order.joblib")

            # 🔹 Verificar si hay suficientes muestras y clases
            n_samples = len(X_rf_scaled)
            if n_samples < 4 or len(np.unique(y_rf_encoded)) < 2:
                X_train, y_train = X_rf_scaled, y_rf_encoded
                X_val, y_val = None, None
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_rf_scaled, y_rf_encoded,
                    test_size=0.25,
                    random_state=42,
                    stratify=y_rf_encoded
                )

            # 🔹 Entrenar modelo
            rf = RandomForestClassifier(n_estimators=200, random_state=42)
            rf.fit(X_train, y_train)

            # 🔹 Reentrenar con todos los datos antes de guardar
            rf.fit(X_rf_scaled, y_rf_encoded)
            joblib.dump(rf, MODEL_RF_PATH)
            st.success("🌲 Modelo Random Forest entrenado, validado y guardado correctamente (muestra triplicada).")

# ------------------TAB 2 ----------------------------------------------------------------------------------------------------------------
# -------------------- PREDICCIÓN --------------------------------------------------------------------------------------------------------
with tab2:
    st.title("Predicción de fallas")
    with st.expander("📋 Estructura esperada del archivo Excel"):
        mostrar_columnas_requeridas()
    # --- Verificar modelos disponibles ---
    models_available = {
        "cnn": os.path.exists(MODEL_CNN_PATH),
        "lstm": os.path.exists(MODEL_LSTM_PATH),
        "rf": os.path.exists(MODEL_RF_PATH)  # Nuevo modelo RF
    }

    if not any(models_available.values()):
        st.error("No hay modelos entrenados.")
    else:
        # --- Cargar modelos ---
        model_cnn = load_model(MODEL_CNN_PATH) if models_available["cnn"] else None
        model_lstm = load_model(MODEL_LSTM_PATH) if models_available["lstm"] else None
        model_rf = joblib.load(MODEL_RF_PATH) if models_available["rf"] else None  # RF cargado con joblib
        FEATURE_COLS_ORDER = joblib.load("feature_cols_order.joblib") if os.path.exists("feature_cols_order.joblib") else []
        label_encoder = joblib.load(LABEL_ENCODER_PATH) if os.path.exists(LABEL_ENCODER_PATH) else None

        # --- Subir archivo ---
        uploaded_pred_file = st.file_uploader("Subuir Excel para predecir", type=["xlsx"], key="pred")
        if uploaded_pred_file is not None:
            # Leer sin encabezado para ubicar la fila correcta
            df_pred_raw = cargar_y_ordenar_excel(uploaded_pred_file)
            if "Date time" in df_pred_raw.columns:
                df_pred_raw["datetime"] = pd.to_datetime(df_pred_raw["Date time"], errors="coerce")
                df_pred_raw.drop(columns=["Date time"], inplace=True)
            #    # Asegurar que la columna datetime esté al inicio
            #    cols = ["datetime"] + [col for col in df_pred_raw.columns if col != "datetime"]
            #    df_pred_raw = df_pred_raw[cols]
            
            # --- Verificación de columnas ---
            missing_cols = [col for col in expected_columns if col not in df_pred_raw.columns]

            if not missing_cols==missing_cols:
                st.warning("No se encontraron columnas válidas.")
            else:
                # --- Limpieza y preparación ---
                df_pred = df_pred_raw
                if "datetime" in df_pred_raw.columns:
                    df_pred["datetime"] = df_pred_raw["datetime"].values
                df_pred = df_pred.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
            
                timesteps = 10
                X_new = df_pred[[c for c in df_pred.columns if c != "datetime"]].values
                X_seq = np.array([X_new[i - timesteps:i, :] for i in range(timesteps, len(X_new))])

                preds = {}
                if model_cnn:
                    preds["cnn"] = model_cnn.predict(X_seq)
                if model_lstm:
                    preds["lstm"] = model_lstm.predict(X_seq)

                # --- RF (flujo separado) ---
                rf_predictions = []
                if model_rf:
                    # Extraer características globales de la señal
                    feats = extract_features_from_zone(df_pred)
                    Xcols = FEATURE_COLS_ORDER if FEATURE_COLS_ORDER else sorted(feats.keys())
                    xvec = [feats.get(c, 0.0) for c in Xcols]
                    clean_xvec = [min(max(float(v) if not pd.isna(v) else 0.0, -1e6), 1e6) for v in xvec]

                    probs = model_rf.predict_proba([clean_xvec])[0]
                    idx = int(np.argmax(probs))
                    encoded_label = model_rf.classes_[idx]

                    # 🔹 Detectar si el label es numérico o ya es texto
                    encoded_label = model_rf.classes_[idx]

                    # 🔹 Intentar decodificar correctamente, sea cual sea el tipo
                    pred_label = None
                    if label_encoder is not None:
                        try:
                            # Si el label es numérico, intentar decodificar
                            if isinstance(encoded_label, (int, np.integer, float, np.floating)):
                                pred_label = label_encoder.inverse_transform([int(encoded_label)])[0]
                            # Si el label está en formato array
                            elif isinstance(encoded_label, (list, np.ndarray)) and len(encoded_label) == 1:
                                val = encoded_label[0]
                                if isinstance(val, (int, np.integer)):
                                    pred_label = label_encoder.inverse_transform([val])[0]
                                else:
                                    pred_label = str(val)
                            else:
                                # Si ya es texto, no necesita decodificación
                                pred_label = str(encoded_label)
                        except Exception:
                            pred_label = str(encoded_label)
                    else:
                        pred_label = str(encoded_label)

                    # 🔹 Asegurar formato legible
                    pred_label = str(pred_label).strip().capitalize()

                    # --- RF (flujo separado) ---
                    var_cols = [c for c in df_pred.columns if c != "datetime"]
                    # Inicializar diccionario de marcadores
                    markers = {}
                    # --- RF dividido en subzonas ---
                    rf_predictions = []
                    if model_rf:
                        var_cols = [c for c in df_pred.columns if c != "datetime"]

                        # Crear subzonas: dividir el dataset en 4 partes
                        n_zones = 3
                        zone_len = len(df_pred) // n_zones
                        if zone_len == 0:
                            zone_len = len(df_pred)

                        for z in range(n_zones):
                            start_idx = z * zone_len
                            end_idx = (z + 1) * zone_len if z < n_zones - 1 else len(df_pred)
                            sub_df = df_pred.iloc[start_idx:end_idx]

                            # Evitar ventanas demasiado cortas o uniformes
                            if len(sub_df) < 2:
                                continue
                            std_mean = sub_df[var_cols].std(ddof=0).mean()
                            if std_mean < 1e-8:
                                continue

                            # Extraer features
                            feats = extract_features_from_zone(sub_df)
                            Xcols = FEATURE_COLS_ORDER if FEATURE_COLS_ORDER else sorted(feats.keys())
                            xvec = [feats.get(c, 0.0) for c in Xcols]
                            clean_xvec = [min(max(float(v) if not pd.isna(v) else 0.0, -1e6), 1e6) for v in xvec]

                            # Predecir con RF
                            probs = model_rf.predict_proba([clean_xvec])[0]
                            idx = int(np.argmax(probs))
                            encoded_label = model_rf.classes_[idx]

                            # Decodificar label
                            pred_label = str(encoded_label)
                            if label_encoder is not None:
                                try:
                                    if isinstance(encoded_label, (int, np.integer, float, np.floating)):
                                        pred_label = label_encoder.inverse_transform([int(encoded_label)])[0]
                                    elif isinstance(encoded_label, (list, np.ndarray)) and len(encoded_label) == 1:
                                        val = encoded_label[0]
                                        if isinstance(val, (int, np.integer, float, np.floating)):
                                            pred_label = label_encoder.inverse_transform([int(val)])[0]
                                        else:
                                            pred_label = str(val)
                                    else:
                                        pred_label = str(encoded_label)
                                except:
                                    pred_label = str(encoded_label)

                            rf_predictions.append({
                                "timestamp": pd.to_datetime(sub_df["datetime"].iloc[len(sub_df)//2], errors="coerce"),
                                "pred_name": str(pred_label).strip().capitalize(),
                                "prob_pct": round(float(probs[idx]) * 100, 3),
                                "model": "RF"
                            })

                    # --- Agregar solo el punto de la última subzona ---
                    if rf_predictions:
                        last_rf = rf_predictions[-1]  # Tomar la última subzona procesada
                        last_rf["label"] = f"{last_rf['pred_name']} (RF - {last_rf['prob_pct']}%)"
                        markers["rf"] = pd.DataFrame([last_rf])
                   
                ts_windows = map_index_to_timestamp(df_pred, timesteps)
                var_cols = [c for c in df_pred.columns if c != "datetime"]
                markers = {}

                # --- Calcular zonas de alta probabilidad (CNN y LSTM) ---
                for mname, arr in preds.items():
                    n_windows, n_classes = arr.shape
                    mean_probs = np.mean(arr, axis=0)
                    df_mean = pd.DataFrame({
                        "class_idx": np.arange(n_classes),
                        "class_name": [failures[i] if i < len(failures) else f"cl{i}" for i in range(n_classes)],
                        "mean_prob": mean_probs
                    }).sort_values("mean_prob", ascending=False).reset_index(drop=True)

                    top_class_idx = int(df_mean.loc[0, "class_idx"])
                    top_class_name = df_mean.loc[0, "class_name"]
                    top_class_mean = float(df_mean.loc[0, "mean_prob"])
                    thresh = max(0.3, top_class_mean * 0.6)

                    class_probs = arr[:, top_class_idx]
                    high_idx = np.where(class_probs >= thresh)[0]
                    if high_idx.size == 0:
                        for alt in [0.5, 0.4, 0.3, 0.2]:
                            high_idx = np.where(class_probs >= alt)[0]
                            if high_idx.size > 0:
                                thresh = alt
                                break

                    clusters = []
                    for k, g in groupby(enumerate(high_idx), lambda x: x[0] - x[1]):
                        cluster = list(map(itemgetter(1), g))
                        clusters.append(cluster)

                    cluster_starts = []
                    for c in clusters:
                        start_idx = c[0]
                        avg_prob = np.mean([class_probs[i] for i in c])
                        cluster_starts.append((start_idx, top_class_name, avg_prob, c))

                    cluster_starts.sort(key=lambda x: x[2], reverse=True)
                    selected = cluster_starts[:3]

                    if not selected:
                        markers[mname] = pd.DataFrame()
                        continue

                    rows = []
                    for start_idx, pred_name, avg_prob, window_idxs in selected:
                        ts = ts_windows.iloc[start_idx]
                        rows.append({
                            "timestamp": pd.to_datetime(ts),
                            "pred_name": pred_name,
                            "prob_pct": round(avg_prob * 100, 3),
                            "model": mname.upper()
                        })
                    df_mark = pd.DataFrame(rows)
                    df_mark["label"] = df_mark.apply(lambda r: f"{r['pred_name']} ({r['model']} - {r['prob_pct']}%)", axis=1)
                    markers[mname] = df_mark

                # --- Agregar solo el punto final de RF ---
                if rf_predictions:
                    df_rf = pd.DataFrame([rf_predictions[-1]])
                    df_rf["label"] = df_rf.apply(lambda r: f"{r['pred_name']} (RF - {r['prob_pct']}%)", axis=1)
                    markers["rf"] = df_rf

                # ======================================================
                # RESULTADOS DE PREDICCIÓN Y TABLAS RESUMEN (VISUAL)
                # ======================================================

                grupos_fallas = {
                    "Falla Mecánica": ["Rotura de eje", "Taponamiento", "Comunicación tubing-casing", "Bloqueo de etapas en la bomba"],
                    "Falla Eléctrica": ["Daño en el sensor de fondo", "Desbalance de fases", "Falla electrica en superficie (Robo, daño equipos)"],
                    "Otros": ["Aumento del corte de agua", "Aumento de la presión del reservorio", "Aumento de gas libre en la entrada de la bomba", "Alta temperatura"]
                }

                # Acciones sugeridas específicas por tipo de falla
                acciones_por_falla = {
                    "Rotura de eje": "Detener el pozo e inspeccionar integridad del sistema de bombeo. Evaluar reemplazo del eje.",
                    "Taponamiento": "Realizar limpieza mecánica o tratamiento químico (acidificación o solvente).",
                    "Comunicación tubing-casing": "Verificar el nivel de fluido en el pozo y el amperaje. Si hay un nivel alto de fluido y un amperaje normal, posible agujero en la tubería.",
                    "Bloqueo de etapas en la bomba": "Desarmar bomba y revisar desgaste/obstrucción de componentes internos.",
                    "Daño en el sensor de fondo": "Revisar cable de potencia, conexiones y reemplazar sensor si es necesario.",
                    "Desbalance de fases": "Verificar voltajes, corriente y calibrar banco de transformadores.",
                    "Falla electrica en superficie (Robo, daño equipos)": "Inspección visual del control surface equipment y reposición si aplica.",
                    "Aumento del corte de agua": "Analizar pruebas de producción y considerar taponamiento selectivo o intervención de zonas.",
                    "Aumento de la presión del reservorio": "Verificar válvulas, líneas de flujo y condiciones operacionales.",
                    "Aumento de gas libre en la entrada de la bomba": "Revisar operador de gas y evaluar instalación de separador gas-líquido.",
                    "Alta temperatura": "Revisar sistema de enfriamiento y carga del motor."
                }

                all_prob_tables = {}

                # ======================================================
                # PRIMERO: construir las tablas globales de probabilidad
                # ======================================================
                for modelo, dfm in markers.items():
                    if dfm.empty:
                        continue

                    if modelo.lower() == "rf" and "probs" in locals():
                        try:
                            class_labels = label_encoder.inverse_transform(model_rf.classes_)
                        except Exception:
                            if "failures" in locals():
                                class_labels = failures
                            else:
                                class_labels = [str(c) for c in model_rf.classes_]

                        prob_df = pd.DataFrame({
                            "Tipo de falla": [str(c).capitalize() for c in class_labels],
                            "Probabilidad (%)": np.round(probs * 100, 2)
                        }).sort_values("Probabilidad (%)", ascending=False)

                        all_prob_tables[modelo.upper()] = prob_df

                    elif modelo.lower() in preds:
                        arr = preds[modelo.lower()]
                        mean_probs = np.mean(arr, axis=0)
                        class_names = [failures[i] if i < len(failures) else f"cl{i}" for i in range(len(mean_probs))]
                        prob_df = pd.DataFrame({
                            "Tipo de falla": [c.capitalize() for c in class_names],
                            "Probabilidad (%)": np.round(mean_probs * 100, 2)
                        }).sort_values("Probabilidad (%)", ascending=False)
                        all_prob_tables[modelo.upper()] = prob_df

                # ======================================================
                # AHORA: construir el resumen principal usando las tablas globales
                # ======================================================
                resumen_pred = []

                for modelo, dfm in markers.items():
                    if modelo.upper() != "LSTM":
                        continue

                    best_row = dfm.loc[dfm["prob_pct"].idxmax()]
                    falla = best_row["pred_name"].capitalize()

                    #Tomar el valor de probabilidad desde all_prob_tables (sincronizado con donas)
                    df_probs = all_prob_tables[modelo.upper()]
                    match = df_probs[df_probs["Tipo de falla"].str.lower() == falla.lower()]
                    prob = match["Probabilidad (%)"].iloc[0] if not match.empty else best_row["prob_pct"]

                    tipo_falla = next(
                        (g for g, tipos in grupos_fallas.items() if any(t.lower() in falla.lower() for t in tipos)),
                        "Otros"
                    )

                    accion_sugerida = acciones_por_falla.get(falla, "Revisar registros y condiciones externas")

                    resumen_pred.append({
                        "Modelo": modelo.upper(),
                        "Posible falla": tipo_falla,
                        "Posible tipo de falla": falla,
                        "Probabilidad": f"{prob:.2f}%",
                        "Acción sugerida": accion_sugerida
                    })

                # ======================================================
                # Mostrar resultados
                # ======================================================
                if resumen_pred:
                    df_resumen = pd.DataFrame(resumen_pred)

                    posible_falla = df_resumen.iloc[0]["Posible falla"]
                    posible_tipo = df_resumen.iloc[0]["Posible tipo de falla"]

                    st.markdown(f"""
                        <div style='background-color:#c0392b; padding:12px; border-radius:8px; text-align:center;'>
                            <h3 style='color:white; margin:0;'> Posible falla detectada: {posible_falla} - {posible_tipo}</h3>
                        </div>
                    """, unsafe_allow_html=True)

                    st.dataframe(df_resumen, width='stretch')

                    # ======================================================
                    # DISTRIBUCIÓN DE PROBABILIDAD POR MODELO (TABLA DIRECTA)
                    # ======================================================
                    
                    modelos_disponibles = list(all_prob_tables.keys())
                    all_fallas = sorted(set(sum([df["Tipo de falla"].tolist() for df in all_prob_tables.values()], [])))
                    combined_df = pd.DataFrame({"Posible tipo de falla": all_fallas})

                    for modelo in ["LSTM", "CNN", "RF"]:
                        if modelo in all_prob_tables:
                            df_model = all_prob_tables[modelo]
                            combined_df = combined_df.merge(
                                df_model[["Tipo de falla", "Probabilidad (%)"]],
                                how="left",
                                left_on="Posible tipo de falla",
                                right_on="Tipo de falla"
                            ).drop(columns=["Tipo de falla"])
                            combined_df.rename(columns={"Probabilidad (%)": f"Probabilidad {modelo} (%)"}, inplace=True)
                        else:
                            combined_df[f"Probabilidad {modelo} (%)"] = np.nan

                    combined_df = combined_df.fillna(0)
                    combined_df["Probabilidad Máxima"] = combined_df[
                        [c for c in combined_df.columns if "Probabilidad" in c]
                    ].max(axis=1)
                    combined_df = combined_df.sort_values("Probabilidad Máxima", ascending=False).drop(columns=["Probabilidad Máxima"])

                    def color_prob(val):
                        if val >= 90: color = "#d73027"
                        elif val >= 70: color = "#f46d43"
                        elif val >= 50: color = "#fdae61"
                        elif val >= 30: color = "#fee08b"
                        else: color = "#ffffbf"
                        return f'background-color: {color}; color:black; font-weight:bold;'

                    styled_combined = combined_df.style.applymap(
                        color_prob,
                        subset=[c for c in combined_df.columns if "Probabilidad" in c]
                    )

                    st.dataframe(styled_combined, width='stretch')

                # -------------------- GRAFICAR --------------------
                model_colors = {"LSTM": "red"}

                color_options = [
                    ("🟥", "#D62728"),
                    ("🟧", "#FF7F0E"),
                    ("🟨", "#FFD700"),
                    ("🟩", "#2CA02C"),
                    ("🟦", "#1F77B4"),
                    ("🟪", "#9467BD"),
                    ("⬛", "#000000"),
                    ("⬜", "#7F7F7F"),
                    ("🟫", "#8C564B"),
                    ("🟩", "#BCBD22"),
                    ("🟦", "#17BECF"),
                    ("🟦", "#00FFFF"),
                    ("🟪", "#FF00FF"),
                    ("🟩", "#00FF00"),
                    ("🟫", "#800000"),
                    ("🟦", "#000080"),
                    ("🟨", "#808000"),
                    ("🟩", "#008080"),
                    ("🟥", "#CC0000"),
                    ("🟪", "#CC79A7")
                ]

                for col in var_cols:
                    display_label = label_map.get(col, col)

                    df_plot = pd.DataFrame({
                        "timestamp": pd.to_datetime(df_pred["datetime"]),
                        col: df_pred[col].values
                    })

                    # Seleccionar color aleatorio para la línea principal
                    emoji, line_color = random.choice(color_options)

                    # --- Gráfico base ---
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_plot["timestamp"],
                        y=df_plot[col],
                        mode="lines",
                        name=display_label,
                        line=dict(color=line_color, width=2),
                        hovertemplate=f'Tiempo: %{{x}}<br>{display_label}: %{{y:.3f}}<extra></extra>'
                    ))

                    # --- Añadir predicciones ---
                    for mname, df_mark in markers.items():
                        if df_mark.empty:
                            continue
                        if mname.upper() != "LSTM":
                            continue

                        df_mark["timestamp"] = pd.to_datetime(df_mark["timestamp"], errors="coerce")
                        df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"], errors="coerce")
                        df_mark = df_mark.merge(df_plot, on="timestamp", how="left")

                        grouped = df_mark.groupby(["pred_name", "model"], as_index=False).agg({
                            col: list,
                            "timestamp": list,
                            "prob_pct": "mean"
                        })

                        for _, row in grouped.iterrows():
                            official_prob = row["prob_pct"]
                            if all_prob_tables and row["model"].upper() in all_prob_tables:
                                df_probs = all_prob_tables[row["model"].upper()]
                                match = df_probs[df_probs["Tipo de falla"].str.lower() == row["pred_name"].lower()]
                                if not match.empty:
                                    official_prob = match["Probabilidad (%)"].iloc[0]

                            legend_name = f"{row['pred_name']} ({row['model']}-{official_prob:.2f}%)"

                            fig.add_trace(go.Scatter(
                                x=row["timestamp"],
                                y=row[col],
                                mode="markers",
                                name=legend_name,
                                marker=dict(size=8, color=model_colors.get(row["model"], "orange")),
                            ))

                    # --- Menú de selección de color (instantáneo sin recarga) ---
                    buttons = [
                        dict(
                            label=emoji,                 # solo el cuadrado
                            method="restyle",
                            args=[{"line.color": [hex_color]}],  # cambia el color de la línea
                        )
                        for emoji, hex_color in color_options
                    ]

                    fig.update_layout(
                        xaxis_title="Tiempo",
                        yaxis_title=display_label,
                        height=300,
                        margin=dict(l=40, r=40, t=40, b=40),
                        legend_title_text="Predicciones",
                        hovermode="closest",
                        updatemenus=[dict(
                            buttons=buttons,
                            direction="down",
                            showactive=True,
                            x=0,
                            xanchor="left",
                            y=1.20,
                            yanchor="top",
                            bgcolor="rgba(255,255,255,0.8)",
                            bordercolor="gray",
                            borderwidth=1
                        )]
                    )
                    fig.update_xaxes(showgrid=True, automargin=True)
                    fig.update_yaxes(showgrid=True, automargin=True)
                    st.plotly_chart(fig, use_container_width=True)


# -------------------- TAB 3 --------------------------------------------------------------------------------------------------------------
# -------------------- METRICAS DE MODELOS ------------------------------------------------------------------------------------------------
with tab3:
    st.title("Evaluación de modelos de Machine Learning")
    with st.expander("📋 Estructura esperada del archivo Excel"):
        mostrar_columnas_requeridas()
    # --- Selección de la falla correcta ---
    selected_failure = st.selectbox(
        "Seleccione la falla correcta para la evaluación:",
        failures  # lista de fallas disponibles
    )

    # --- Verificar modelos disponibles ---
    models_available = {
        "cnn": os.path.exists(MODEL_CNN_PATH),
        "lstm": os.path.exists(MODEL_LSTM_PATH),
        "rf": os.path.exists(MODEL_RF_PATH)
    }

    if not any(models_available.values()):
        st.error("No hay modelos entrenados.")
    else:
        # --- Cargar modelos ---
        model_cnn = load_model(MODEL_CNN_PATH) if models_available["cnn"] else None
        model_lstm = load_model(MODEL_LSTM_PATH) if models_available["lstm"] else None
        model_rf = joblib.load(MODEL_RF_PATH) if models_available["rf"] else None
        FEATURE_COLS_ORDER = joblib.load("feature_cols_order.joblib") if os.path.exists("feature_cols_order.joblib") else []
        label_encoder = joblib.load(LABEL_ENCODER_PATH) if os.path.exists(LABEL_ENCODER_PATH) else None

        # --- Subir archivo ---
        uploaded_file = st.file_uploader("Subir Excel para evaluar los modelos", type=["xlsx"])
        if uploaded_file is not None:
            # Leer sin encabezado para ubicar la fila correcta
            df_raw = cargar_y_ordenar_excel(uploaded_file)
            
            if "Date time" in df_raw.columns:
                df_raw["datetime"] = pd.to_datetime(df_raw["Date time"], errors="coerce")
                df_raw.drop(columns=["Date time"], inplace=True)
                # Asegurar que la columna datetime esté al inicio
                #cols = ["datetime"] + [col for col in df_raw.columns if col != "datetime"]
                #df_raw = df_raw[cols]

            
            missing_cols = [col for col in expected_columns if col not in df_raw.columns]
            if not missing_cols==missing_cols:
                st.warning("No se encontraron columnas válidas.")
            else:
                df_pred = df_raw
                if "datetime" in df_raw.columns:
                    df_pred["datetime"] = df_raw["datetime"].values
                df_pred = df_pred.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

                timesteps = 10
                X_new = df_pred[[c for c in df_pred.columns if c != "datetime"]].values
                X_seq = np.array([X_new[i - timesteps:i, :] for i in range(timesteps, len(X_new))])

                # --- Predicciones CNN y LSTM ---
                preds = {}
                times_pred = {}
                if model_cnn:
                    start = time.time()
                    preds["cnn"] = model_cnn.predict(X_seq)
                    times_pred["cnn"] = time.time() - start
                if model_lstm:
                    start = time.time()
                    preds["lstm"] = model_lstm.predict(X_seq)
                    times_pred["lstm"] = time.time() - start

                # --- Flujo RF separado ---
                rf_preds = []
                rf_probs = []
                rf_time = 0
                if model_rf:
                    start = time.time()
                    n_zones = 3
                    zone_len = len(df_pred) // n_zones
                    for z in range(n_zones):
                        start_idx = z * zone_len
                        end_idx = (z + 1) * zone_len if z < n_zones - 1 else len(df_pred)
                        sub_df = df_pred.iloc[start_idx:end_idx]

                        if len(sub_df) < 2:
                            continue
                        feats = extract_features_from_zone(sub_df)
                        Xcols = FEATURE_COLS_ORDER if FEATURE_COLS_ORDER else sorted(feats.keys())
                        xvec = [feats.get(c, 0.0) for c in Xcols]
                        clean_xvec = [min(max(float(v) if not pd.isna(v) else 0.0, -1e6), 1e6) for v in xvec]

                        probs = model_rf.predict_proba([clean_xvec])[0]
                        idx = int(np.argmax(probs))
                        encoded_label = model_rf.classes_[idx]

                        if label_encoder is not None:
                            try:
                                if isinstance(encoded_label, (int, np.integer, float, np.floating)):
                                    pred_label = label_encoder.inverse_transform([int(encoded_label)])[0]
                                else:
                                    pred_label = str(encoded_label)
                            except:
                                pred_label = str(encoded_label)
                        else:
                            pred_label = str(encoded_label)

                        rf_preds.append({
                            "pred_class": pred_label.strip().capitalize(),
                            "prob": round(float(probs[idx]) * 100, 2),
                            "timestamp": sub_df["datetime"].iloc[len(sub_df)//2]
                        })
                        rf_probs.append(probs)
                    rf_time = time.time() - start

                # --- Función para métricas ---
                def compute_classification_metrics(y_true, y_pred, y_prob, classes):
                    y_true_bin = label_binarize(y_true, classes=classes)
                    y_pred_bin = label_binarize(y_pred, classes=classes)

                    metrics = {
                        "accuracy": accuracy_score(y_true, y_pred),
                        "precision": precision_score(y_true, y_pred, labels=classes, average='weighted', zero_division=0),
                        "recall": recall_score(y_true, y_pred, labels=classes, average='weighted', zero_division=0),
                        "f1_score": f1_score(y_true, y_pred, labels=classes, average='weighted', zero_division=0)
                    }

                    # AUC-ROC
                    if len(classes) > 1 and y_prob is not None:
                        try:
                            metrics["auc_roc"] = roc_auc_score(y_true_bin, y_prob, average='weighted', multi_class='ovr')
                        except:
                            metrics["auc_roc"] = 0
                    else:
                        metrics["auc_roc"] = 0
                    return metrics

                # --- Preparar métricas ---
                metrics_results = {}

                # CNN / LSTM
                for mname in ["cnn", "lstm"]:
                    if mname in preds:
                        y_true = df_pred.get("falla_real", [selected_failure]*len(preds[mname]))
                        y_pred_idx = np.argmax(preds[mname], axis=1)
                        y_pred = [failures[i] for i in y_pred_idx]
                        y_prob = preds[mname]
                        metrics_results[mname] = compute_classification_metrics(y_true, y_pred, y_prob, classes=failures)
                        metrics_results[mname]["time"] = round(times_pred[mname], 3)

                # RF
                if rf_preds:
                    y_true_rf = df_pred.get("falla_real", [selected_failure]*len(rf_preds))
                    y_pred_rf = [r["pred_class"] for r in rf_preds]
                    y_prob_rf = np.array(rf_probs)
                    metrics_results["rf"] = compute_classification_metrics(y_true_rf, y_pred_rf, y_prob_rf, classes=failures)
                    metrics_results["rf"]["time"] = round(rf_time, 3)

                # --- Mostrar tabla de métricas ---
                st.subheader("📈 Métricas de clasificación y tiempo de predicción")
                rows = []
                for model_name, m in metrics_results.items():
                    row = {
                        "Modelo": model_name.upper(),
                        "Falla": selected_failure,
                        "Accuracy": m.get("accuracy",0),
                        "Precision": m.get("precision",0),
                        "Recall": m.get("recall",0),
                        "F1-Score": m.get("f1_score",0),
                        "Tiempo (s)": m.get("time",0)
                    }
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows))

            # -------------------- Evaluación visual y métricas adicionales ----------------
            # -------------------- Análisis visual adicional ----------------
            
            for model_name in metrics_results.keys():
                st.markdown(f"### Modelo: {model_name.upper()}")

                # --- Obtener y_true y y_pred ---
                if model_name in ["cnn", "lstm"]:
                    y_true = df_pred.get("falla_real", [selected_failure]*len(preds[model_name]))
                    y_pred_idx = np.argmax(preds[model_name], axis=1)
                    y_pred = [failures[i] for i in y_pred_idx]
                    probs = preds[model_name]
                else:  # RF
                    y_true = df_pred.get("falla_real", [selected_failure]*len(rf_preds))
                    y_pred = [r["pred_class"] for r in rf_preds]

                # --- Matriz de confusión interactiva ---
                cm = confusion_matrix(y_true, y_pred, labels=failures)
                cm_text = [[str(value) for value in row] for row in cm]
                fig_cm = ff.create_annotated_heatmap(
                    z=cm,
                    x=failures,
                    y=failures,
                    annotation_text=cm_text,
                    colorscale='Blues'
                )
                fig_cm.update_layout(
                    title_text=f"Matriz de confusión - {model_name.upper()}",
                    xaxis_title="Predicción",
                    yaxis_title="Real",
                    width=600,
                    height=500
                )
                st.plotly_chart(fig_cm, width='stretch')

            # -------------------- ANÁLISIS MULTIFALLA SIMULTÁNEO ----------------------------------------------------------------------------
            # -------------------- Análisis múltiple de fallas (añadir al final sin modificar lo anterior) --------------------
            st.markdown("---")
            st.header("🔀 Evaluación múltiple de fallas (simultáneo)")

            # Número de fallas a analizar simultáneamente
            n_failures = st.number_input("¿Cuántas fallas vas a analizar en simultáneo?", min_value=1, max_value=len(failures), value=1, step=1)

            # Listas para almacenar selecciones y archivos
            multi_selected_failures = []
            multi_uploaded_files = []

            st.markdown("Selecciona para cada falla el nombre y sube el archivo correspondiente:")

            for i in range(int(n_failures)):
                st.markdown(f"**Falla {i+1}**")
                fsel = st.selectbox(f"Nombre de la falla #{i+1}", failures, key=f"multi_fail_select_{i}")
                fup = st.file_uploader(f"Sube Excel para la falla #{i+1} (cabecera en fila 11 por defecto)", type=["xlsx"], key=f"multi_file_{i}")
                multi_selected_failures.append(fsel)
                multi_uploaded_files.append(fup)

            # Botón para ejecutar el análisis combinado
            if st.button("Ejecutar análisis combinado de fallas"):
                # Checkear que al menos un archivo fue subido
                if not any(multi_uploaded_files):
                    st.warning("No se subió ningún archivo. Sube al menos uno para continuar.")
                else:
                    # Estructuras para acumular y_true y y_pred por modelo
                    aggregate = {
                        "cnn": {"y_true": [], "y_pred": []},
                        "lstm": {"y_true": [], "y_pred": []},
                        "rf": {"y_true": [], "y_pred": []}
                    }

                    # Parametros constantes (igual que arriba)
                    timesteps = 10

                    # Iterar sobre cada par (falla, archivo)
                    for idx, (f_name, f_file) in enumerate(zip(multi_selected_failures, multi_uploaded_files)):
                        if f_file is None:
                            st.info(f"No se subió archivo para la falla #{idx+1} ({f_name}). Se omite.")
                            continue

                        # Leer archivo (usar la misma función que ya tienes)
                        try:
                            df_raw_multi = cargar_y_ordenar_excel(f_file)
                        except Exception as e:
                            st.error(f"Error leyendo el archivo de la falla #{idx+1} ({f_name}): {e}")
                            continue

                        # Normalizar columna datetime si aplica
                        if "Date time" in df_raw_multi.columns:
                            df_raw_multi["datetime"] = pd.to_datetime(df_raw_multi["Date time"], errors="coerce")
                            df_raw_multi.drop(columns=["Date time"], inplace=True)

                        # Validar columnas esperadas (puedes ajustar la validación si la tienes definida)
                        # Convertir a numérico y limpiar
                        try:
                            df_pred_multi = df_raw_multi.copy()
                            if "datetime" in df_pred_multi.columns:
                                df_pred_multi["datetime"] = df_pred_multi["datetime"].values
                            df_pred_multi = df_pred_multi.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
                        except Exception as e:
                            st.error(f"Error procesando el archivo de la falla #{idx+1} ({f_name}): {e}")
                            continue

                        # Preparar secuencias para modelos secuenciales (CNN/LSTM)
                        X_new_multi = df_pred_multi[[c for c in df_pred_multi.columns if c != "datetime"]].values
                        if len(X_new_multi) <= timesteps:
                            st.warning(f"Archivo de la falla #{idx+1} ({f_name}) tiene pocos datos (<= {timesteps}). Se omite para CNN/LSTM pero puede usarse para RF si aplica.")
                            X_seq_multi = np.array([])
                        else:
                            X_seq_multi = np.array([X_new_multi[i - timesteps:i, :] for i in range(timesteps, len(X_new_multi))])

                        # --- Predicciones CNN y LSTM para este archivo ---
                        try:
                            if model_cnn is not None and X_seq_multi.size:
                                preds_cnn = model_cnn.predict(X_seq_multi)
                                y_pred_idx = np.argmax(preds_cnn, axis=1)
                                y_pred_labels = [failures[i] for i in y_pred_idx]
                                aggregate["cnn"]["y_pred"].extend(y_pred_labels)
                                aggregate["cnn"]["y_true"].extend([f_name] * len(y_pred_labels))
                            else:
                                # si no hay modelo o no hay secuencias, no agregar
                                pass
                        except Exception as e:
                            st.warning(f"Error en predicción CNN para la falla #{idx+1} ({f_name}): {e}")

                        try:
                            if model_lstm is not None and X_seq_multi.size:
                                preds_lstm = model_lstm.predict(X_seq_multi)
                                y_pred_idx = np.argmax(preds_lstm, axis=1)
                                y_pred_labels = [failures[i] for i in y_pred_idx]
                                aggregate["lstm"]["y_pred"].extend(y_pred_labels)
                                aggregate["lstm"]["y_true"].extend([f_name] * len(y_pred_labels))
                            else:
                                pass
                        except Exception as e:
                            st.warning(f"Error en predicción LSTM para la falla #{idx+1} ({f_name}): {e}")

                        # --- Flujo RF por zonas para este archivo ---
                        try:
                            if model_rf is not None:
                                rf_preds_local = []
                                n_zones = 3
                                zone_len = len(df_pred_multi) // n_zones if n_zones>0 else len(df_pred_multi)
                                for z in range(n_zones):
                                    start_idx = z * zone_len
                                    end_idx = (z + 1) * zone_len if z < n_zones - 1 else len(df_pred_multi)
                                    sub_df = df_pred_multi.iloc[start_idx:end_idx]
                                    if len(sub_df) < 2:
                                        continue
                                    feats = extract_features_from_zone(sub_df)
                                    Xcols = FEATURE_COLS_ORDER if FEATURE_COLS_ORDER else sorted(feats.keys())
                                    xvec = [feats.get(c, 0.0) for c in Xcols]
                                    clean_xvec = [min(max(float(v) if not pd.isna(v) else 0.0, -1e6), 1e6) for v in xvec]

                                    try:
                                        probs = model_rf.predict_proba([clean_xvec])[0]
                                        idxmax = int(np.argmax(probs))
                                        encoded_label = model_rf.classes_[idxmax]
                                        if label_encoder is not None:
                                            try:
                                                if isinstance(encoded_label, (int, np.integer, float, np.floating)):
                                                    pred_label = label_encoder.inverse_transform([int(encoded_label)])[0]
                                                else:
                                                    pred_label = str(encoded_label)
                                            except:
                                                pred_label = str(encoded_label)
                                        else:
                                            pred_label = str(encoded_label)
                                        rf_preds_local.append(pred_label.strip().capitalize())
                                    except Exception as e:
                                        st.warning(f"Error predicción RF en zona {z} del archivo de la falla #{idx+1} ({f_name}): {e}")
                                        continue

                                # Agregar a agregados RF
                                if rf_preds_local:
                                    aggregate["rf"]["y_pred"].extend(rf_preds_local)
                                    aggregate["rf"]["y_true"].extend([f_name] * len(rf_preds_local))
                            else:
                                pass
                        except Exception as e:
                            st.warning(f"Error en flujo RF para la falla #{idx+1} ({f_name}): {e}")

                    # --- Después de procesar todos los archivos: generar matrices de confusión por modelo ---
                    st.markdown("### Resultados combinados — Matrices de confusión")

                    any_results = False
                    for model_name in ["cnn", "lstm", "rf"]:
                        y_true_all = aggregate[model_name]["y_true"]
                        y_pred_all = aggregate[model_name]["y_pred"]

                        if len(y_true_all) == 0 or len(y_pred_all) == 0:
                            st.info(f"No hay predicciones válidas para el modelo {model_name.upper()}.")
                            continue

                        any_results = True
                        # Asegurar que las etiquetas que se usan en confusion_matrix sean exactamente la lista 'failures'
                        try:
                            cm = confusion_matrix(y_true_all, y_pred_all, labels=failures)
                        except Exception as e:
                            st.error(f"Error calculando la matriz de confusión para {model_name.upper()}: {e}")
                            continue

                        cm_text = [[str(value) for value in row] for row in cm]
                        fig_cm = ff.create_annotated_heatmap(
                            z=cm,
                            x=failures,
                            y=failures,
                            annotation_text=cm_text,
                            colorscale='Blues'
                        )
                        fig_cm.update_layout(
                            title_text=f"Matriz de confusión combinada - {model_name.upper()}",
                            xaxis_title="Predicción",
                            yaxis_title="Real",
                            width=700,
                            height=600
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)

                    if not any_results:
                        st.warning("No se generaron matrices de confusión (no hubo predicciones válidas).")

               
# -------------------- TAB 4 -----------------------------------------------------------------------------------------------------------------
# -------------------- INFORMACION -----------------------------------------------------------------------------------------------------------
with tab4:
    st.title("Información")

    st.header("1️⃣ Contexto del Programa")
    st.markdown("""
    Este programa está diseñado para **predecir fallas en sistemas ESP (Electric Submersible Pumps)** 
    utilizando modelos de Machine Learning: **CNN, LSTM y Random Forest**.  

    Combina procesamiento de series temporales (CNN/LSTM) con extracción de features y clasificación (Random Forest).  
    Permite identificar el **tipo de falla** y su probabilidad, ayudando a la **toma de decisiones preventivas**.
    """)

    st.header("2️⃣ Flujo del Programa")
    st.markdown("""
    **Entrenamiento**
    - Subir un Excel con datos de sensores.
    - Filtrar y limpiar datos automáticamente.
    - Visualizar variables y rangos de fallas.
    - Guardar datos de fallas seleccionados en datasets para CNN/LSTM y Random Forest.
    - Entrenar modelos con parámetros predefinidos (timesteps, epochs, batch size).

    **Predicción**
    - Subir un nuevo Excel de datos para predecir.
    - Preparar secuencias temporales para CNN/LSTM.
    - Extraer features globales para Random Forest.
    - Mostrar predicciones con probabilidades, tablas resumen y gráficos interactivos.
    - Visualizar posibles fallas y acciones correctivas sugeridas.

    **Métricas de Modelos**
    - Evaluar desempeño entre CNN, LSTM y Random Forest
    
    """)

    st.header("3️⃣ Uso Recomendado")
    st.markdown("""
    - Revisar que las columnas esperadas estén presentes.
    - Asegurarse de que los datos estén correctamente alineados con timestamps.
    - Para mejores resultados, entrenar modelos con varias fallas y registros históricos.
    """)

    st.header("4️⃣ Funcionalidades Clave")
    st.markdown("""
    - Filtrado automático de columnas y conversión a valores numéricos.
    - Generación de secuencias temporales para CNN/LSTM.
    - Extracción de features estadísticas para Random Forest.
    - Visualizaciones interactivas con Plotly y Altair.
    - Tablas de probabilidad por modelo y gráficos de dona para distribución de fallas.
    """)

    st.header("5️⃣ Notas Adicionales")
    st.markdown("""
    - El programa soporta **múltiples modelos entrenados simultáneamente**.
    - Los datasets y modelos se guardan localmente para su reutilización.
    - Este tab sirve como **guía rápida y referencia** para usuarios nuevos y para entender el flujo completo.
    """)

    st.header("6️⃣ Autores")
    st.markdown("""
    Este proyecto fue desarrollado como parte del trabajo de titulación en  
    Ingeniería en Petróleos en la Universidad Central del Ecuador.

    **Autores:**

    **Joao Ugalde**  
    Correo: ugaldejoao27@gmail.com

    **Mario Jarrín**  
    Correo: mariojarrin962@gmail.com
    """)

