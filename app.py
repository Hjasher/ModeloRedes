import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
import matplotlib.pyplot as plt

NOMBRE = "Hjasher Medina"
CUENTA = "20231900178"

st.set_page_config(page_title="Clasificador de conexiones", layout="centered")

COLORES = ["#0891b2", "#f97316"]
VARIABLES = ["trafico_mbps", "latencia_ms"]


@st.cache_resource
def cargar():
    
    base = Path(__file__).parent
    modelo = joblib.load(base / "svm_rbf_conexiones.pkl")
    meta = json.loads((base / "model_metadata.json").read_text(encoding="utf-8"))
    datos = pd.read_csv(base / "conexiones_red.csv")
    return modelo, meta, datos


modelo, meta, datos = cargar()

st.title("Clasificador de conexiones de red")
st.markdown(f"**Nombre completo** {NOMBRE}")
st.markdown(f"**Número de cuenta** {CUENTA}")
st.caption(
    f"{meta['nombre_modelo']}. Datos sintéticos y educativos, no es un detector real de ataques. "
    f"Accuracy en prueba {meta['accuracy_prueba']:.2f}."
)
if sklearn.__version__ != meta["sklearn_version"]:
    st.warning(
        f"El modelo se guardó con scikit-learn {meta['sklearn_version']} y aquí hay {sklearn.__version__}. "
        "Fija esa versión en requirements.txt."
    )

with st.form("formulario"):
    col1, col2 = st.columns(2)
    trafico_in = col1.number_input("Tráfico (Mbps)", min_value=0.0, value=None, step=0.5, format="%.2f",
                                   placeholder="Escriba el tráfico")
    latencia_in = col2.number_input("Latencia (ms)", min_value=0.0, value=None, step=0.5, format="%.2f",
                                    placeholder="Escriba la latencia")
    enviado = st.form_submit_button("Clasificar conexión")

if enviado:
    if trafico_in is None or latencia_in is None:
        st.error("Escriba los dos valores antes de clasificar.")
    else:
        st.session_state["consulta"] = (float(trafico_in), float(latencia_in))

if "consulta" not in st.session_state:
    st.info("Escriba el tráfico y la latencia, y pulse Clasificar conexión.")
    st.stop()

trafico, latencia = st.session_state["consulta"]
st.write(f"Conexión consultada con {trafico:.2f} Mbps de tráfico y {latencia:.2f} ms de latencia.")

punto = pd.DataFrame({"trafico_mbps": [trafico], "latencia_ms": [latencia]})
clase = int(modelo.predict(punto)[0])
puntaje = float(modelo.decision_function(punto)[0])

if clase == 1:
    st.error(f"Conexión SOSPECHOSA (clase 1, {meta['clases']['1']})")
else:
    st.success(f"Conexión NORMAL (clase 0, {meta['clases']['0']})")
st.write(f"Puntaje de decisión del SVM: {puntaje:.2f}. Positivo indica clase 1 y negativo indica clase 0.")

t_min, t_max = meta["rango_trafico_mbps"]
l_min, l_max = meta["rango_latencia_ms"]
if not (t_min <= trafico <= t_max and l_min <= latencia <= l_max):
    st.warning(
        "Este punto está fuera del rango de los datos de entrenamiento "
        f"(tráfico {t_min:.0f} a {t_max:.0f} Mbps, latencia {l_min:.0f} a {l_max:.0f} ms). "
        "El modelo está extrapolando, así que la respuesta es poco confiable."
    )

# Frontera de decisión con el punto consultado
x_min, x_max = min(t_min, trafico) - 5, max(t_max, trafico) + 5
y_min, y_max = min(l_min, latencia) - 5, max(l_max, latencia) + 5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 220), np.linspace(y_min, y_max, 220))
rejilla = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=VARIABLES)
z = modelo.decision_function(rejilla).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.contourf(xx, yy, (z > 0).astype(int), levels=[-0.5, 0.5, 1.5], colors=COLORES, alpha=0.15)
ax.contour(xx, yy, z, levels=[0], colors=["#102a43"], linewidths=2)
for c, nombre in [(0, "Normal"), (1, "Sospechosa")]:
    sub = datos[datos["clase"] == c]
    ax.scatter(sub.trafico_mbps, sub.latencia_ms, c=COLORES[c], s=18, label=nombre, edgecolors="white", linewidths=0.3)
ax.scatter([trafico], [latencia], marker="*", s=300, c="#facc15", edgecolors="black", linewidths=1.2, label="Conexión consultada", zorder=5)
ax.set_xlabel("Tráfico (Mbps)")
ax.set_ylabel("Latencia (ms)")
ax.legend(fontsize=8, loc="best")
fig.tight_layout()
st.pyplot(fig)
