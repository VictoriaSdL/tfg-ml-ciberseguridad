# tfg-ml-ciberseguridad
# TFG – Análisis de modelos de Machine Learning para la detección y prevención de amenazas en ciberseguridad de redes

Trabajo Fin de Grado – Grado en Ingeniería en Tecnologías de Telecomunicación (GITT)  
Universidad Pontificia Comillas (ICAI)  
Autora: Victoria Sánchez de León González  
Director: Alfonso Vázquez Requejo

## Descripción

Este repositorio contiene la implementación de los modelos de Machine Learning desarrollados en el TFG. Se evalúan seis modelos aplicados a la detección de intrusiones en redes sobre el dataset CICIDS2017: KNN, CART, Random Forest, MLP, SVM y Llama 3.2-1B (zero-shot).

## Estructura del repositorio

- `KNN.ipynb` – Clasificador K-Nearest Neighbors
- `CART_SMOTE.ipynb` – Árbol de decisión CART con SMOTE
- `RF_SMOTE.ipynb` – Random Forest con SMOTE
- `RN_SMOTE.ipynb` – Red neuronal MLP con SMOTE
- `SVM_SMOTE.ipynb` – Support Vector Machine con SMOTE
- `LLM.py` – Experimento con Llama 3.2-1B en modo zero-shot

## Dataset

El dataset utilizado es el CICIDS2017, disponible en:  
https://www.unb.ca/cic/datasets/ids-2017.html

De todos los archivos disponibles, este proyecto utiliza los siguientes seis ficheros CSV:

- Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
- Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
- Friday-WorkingHours-Morning.pcap_ISCX.csv
- Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
- Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
- Tuesday-WorkingHours.pcap_ISCX.csv

Una vez descargados, colócalos en una carpeta llamada `dataset` en el mismo directorio que los scripts.

## Requisitos

- Python 3.9+
- pandas, numpy, scikit-learn, imbalanced-learn, transformers, torch

## Configuración del token de Hugging Face (solo para LLM.py)

El script LLM.py requiere un token de acceso a Hugging Face para descargar el modelo Llama 3.2-1B. Crea tu propio token en https://huggingface.co/settings/tokens y configúralo como variable de entorno:

```bash
export HF_TOKEN="tu_token_aqui"
```
