"""
Evaluación de LLM (Llama 3.2-1B) para detección de ataques en red
Dataset: CICIDS-2017 — TFG Ciberseguridad
"""

import os, glob, time, warnings
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings("ignore")

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN")
DATASETS_PATH    = os.path.join("dataset", "*.csv")
MODEL_NAME       = "meta-llama/Llama-3.2-1B"
SAMPLE_PER_CLASS = 20          # 20 x 10 clases = 200 registros (~30 min en CPU)
RANDOM_STATE     = 42
OUTPUT_CSV       = "resultados_llm_cicids.csv"
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Cargar CSV ─────────────────────────────────────────────────────────────
print("Cargando dataset...")
paths = glob.glob(DATASETS_PATH)
if not paths:
    raise FileNotFoundError(f"No se encontraron CSV en: {DATASETS_PATH}")

df_full = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

# Quitar espacios de TODOS los nombres de columna
df_full.columns = [c.strip() for c in df_full.columns]

# Limpiar infinitos y NaN
df_full.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
df_full.dropna(inplace=True)

print(f"  Registros tras limpieza: {len(df_full):,}")
print(f"  Columna de etiqueta existe: {'Label' in df_full.columns}")
print(f"  Clases:\n{df_full['Label'].value_counts().to_string()}\n")

# ── 2. Muestra estratificada ──────────────────────────────────────────────────
trozos = []
for clase, grupo in df_full.groupby("Label"):
    n = min(len(grupo), SAMPLE_PER_CLASS)
    trozos.append(grupo.sample(n, random_state=RANDOM_STATE))
df_sample = pd.concat(trozos, ignore_index=True)

# Aseguramos que la columna Label sigue existiendo tras el reset
assert "Label" in df_sample.columns, "ERROR: columna Label no encontrada en df_sample"

clases = sorted(df_full["Label"].unique().tolist())
print(f"Muestra: {len(df_sample)} registros | Clases: {len(clases)}\n")

# ── 3. Cargar modelo ──────────────────────────────────────────────────────────
print("Cargando modelo Llama...")
login(token=HF_TOKEN)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()
print("Modelo cargado.\n")

# ── 4. Features para el prompt ────────────────────────────────────────────────
FEATURES_PROMPT = [f for f in [
    "Destination Port", "Flow Duration",
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Flow Bytes/s", "Flow Packets/s",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count",
    "Average Packet Size", "Packet Length Mean", "Packet Length Std",
    "Flow IAT Mean", "Flow IAT Std",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
] if f in df_sample.columns]

# ── 5. Prompt ─────────────────────────────────────────────────────────────────
def construir_prompt(row):
    lineas = "\n".join(
        f"  - {f}: {row[f]:.4f}" if isinstance(row[f], float) else f"  - {f}: {row[f]}"
        for f in FEATURES_PROMPT
    )
    return (
        "You are a network security expert. "
        "Analyze the following network flow and classify it.\n\n"
        f"Features:\n{lineas}\n\n"
        f"Possible categories: {', '.join(clases)}\n\n"
        "Reply with ONLY the category name, nothing else.\n\n"
        "Category:"
    )

# ── 6. Inferencia ─────────────────────────────────────────────────────────────
def predecir(row):
    inputs = tokenizer(construir_prompt(row), return_tensors="pt",
                       truncation=True, max_length=600)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=15,
                             do_sample=False, repetition_penalty=1.2)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def parsear(resp):
    r = resp.lower()
    for c in clases:
        if c.lower() == r: return c
    for c in clases:
        if c.lower() in r: return c
    for kw, c in [("benign","BENIGN"),("normal","BENIGN"),("ddos","DDoS"),
                  ("portscan","PortScan"),("port scan","PortScan"),("bot","Bot"),
                  ("infiltration","Infiltration"),("brute","Web Attack \ufffd Brute Force"),
                  ("xss","Web Attack \ufffd XSS"),("sql","Web Attack \ufffd Sql Injection"),
                  ("ftp","FTP-Patator"),("ssh","SSH-Patator")]:
        if kw in r: return c
    return "UNKNOWN"

# ── 7. Bucle ──────────────────────────────────────────────────────────────────
predicciones, etiquetas_real, respuestas_raw = [], [], []
print(f"Evaluando {len(df_sample)} registros...\n")
t0 = time.time()

for i, (_, row) in enumerate(df_sample.iterrows(), start=1):
    resp = predecir(row)
    pred = parsear(resp)
    predicciones.append(pred)
    etiquetas_real.append(row["Label"])
    respuestas_raw.append(resp)
    if i % 10 == 0 or i == len(df_sample):
        el = time.time() - t0
        print(f"  [{i:>4}/{len(df_sample)}]  {el:.0f}s transcurridos  |  "
              f"estimado restante: {el/i*(len(df_sample)-i):.0f}s")

t_total = time.time() - t0

# ── 8. Métricas ───────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("RESULTADOS — Llama 3.2-1B sobre CICIDS-2017")
print("="*65)
acc = accuracy_score(etiquetas_real, predicciones)
print(f"\nAccuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"Tiempo total: {t_total:.1f}s  ({t_total/len(df_sample):.1f}s/registro)\n")
print(classification_report(etiquetas_real, predicciones, zero_division=0))
n_unk = predicciones.count("UNKNOWN")
print(f"Respuestas no parseables: {n_unk}/{len(predicciones)} ({100*n_unk/len(predicciones):.1f}%)")

# ── 9. Guardar ────────────────────────────────────────────────────────────────
df_out = df_sample.copy()
df_out["pred_llm"]     = predicciones
df_out["resp_raw_llm"] = respuestas_raw
df_out["correcto"]     = df_out["Label"] == df_out["pred_llm"]
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\nGuardado en: {OUTPUT_CSV}")