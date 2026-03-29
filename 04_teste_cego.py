# -*- coding: utf-8 -*-
"""
04_teste_cego.py
─────────────────────────────────────────────────────────────────────────────
Retreino e avaliação final com dados de teste cego.

ATENÇÃO: este script usa dados/teste_cego.csv SOMENTE aqui.
Usá-lo em qualquer etapa anterior configura data leakage.

Etapas (conforme enunciado):
  5a. Retreina CART com toda a base treino/validação (melhor config de 02)
  5b. Retreina MLP  com toda a base treino/validação (melhor config de 03)
  6.  Avalia ambos no teste cego (1000 vítimas)
  7.  Análise: generalização, matriz de confusão, classes problemáticas
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
import joblib

# ── Configurações ─────────────────────────────────────────────────────────
SEED     = 42
FEATURES = ['idade', 'fc', 'fr', 'pas', 'spo2', 'temp', 'pr', 'sg', 'fx', 'queim']
TARGET   = 'tri'
CLASSES  = ['verde(0)', 'amarelo(1)', 'vermelho(2)', 'preto(3)']
CORES_NOMES = {0: "verde", 1: "amarelo", 2: "vermelho", 3: "preto"}

os.makedirs("modelos",              exist_ok=True)
os.makedirs("resultados",           exist_ok=True)
os.makedirs("resultados/figuras",   exist_ok=True)

# ── Carregar datasets ─────────────────────────────────────────────────────
print("=" * 65)
print("RETREINO + TESTE CEGO")
print("=" * 65)

df_tv   = pd.read_csv("dados/treino_validacao.csv")
df_test = pd.read_csv("dados/teste_cego.csv")

X_tv    = df_tv[FEATURES].values
y_tv    = df_tv[TARGET].values
X_test  = df_test[FEATURES].values
y_test  = df_test[TARGET].values

print(f"Treino/validação : {X_tv.shape[0]} amostras")
print(f"Teste cego       : {X_test.shape[0]} amostras")

# ── Carregar melhores hiperparâmetros ─────────────────────────────────────
with open("resultados/cart_summary.json", encoding="utf-8") as f:
    cart_summary = json.load(f)
with open("resultados/mlp_summary.json",  encoding="utf-8") as f:
    mlp_summary  = json.load(f)

cart_params = cart_summary["best_params"]
mlp_params  = mlp_summary["best_params"]

# JSON salva None como "None" (string) — converte de volta
if cart_params.get("max_depth") == "None":
    cart_params["max_depth"] = None
else:
    cart_params["max_depth"] = int(cart_params["max_depth"])

cart_params["min_samples_leaf"] = int(cart_params["min_samples_leaf"])

print(f"\nMelhor CART : {cart_params}")
print(f"Melhor MLP  : {mlp_params}")

# ─────────────────────────────────────────────────────────────────────────
# ETAPA 5a — Retreino do CART com toda a base treino/validação
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("ETAPA 5a — Retreinando CART")
print("─" * 65)

cart_final = DecisionTreeClassifier(
    criterion        = cart_params["criterion"],
    max_depth        = cart_params["max_depth"],
    min_samples_leaf = cart_params["min_samples_leaf"],
    random_state     = SEED,
)
cart_final.fit(X_tv, y_tv)
joblib.dump(cart_final, "modelos/cart_final.pkl")
print("→ modelos/cart_final.pkl salvo")

# ─────────────────────────────────────────────────────────────────────────
# ETAPA 5b — Retreino da MLP com toda a base treino/validação
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("ETAPA 5b — Retreinando MLP")
print("─" * 65)

# Extrai os valores do pipeline
hidden = eval(mlp_params["mlp__hidden_layer_sizes"])   # string → tuple
activ  = mlp_params["mlp__activation"]
lr     = float(mlp_params["mlp__learning_rate_init"])

mlp_final = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp",    MLPClassifier(
        hidden_layer_sizes = hidden,
        activation         = activ,
        learning_rate_init = lr,
        max_iter           = 500,
        random_state       = SEED,
        early_stopping     = False,  # retreino usa todos os dados
    )),
])
mlp_final.fit(X_tv, y_tv)
joblib.dump(mlp_final, "modelos/mlp_final.pkl")
print("→ modelos/mlp_final.pkl salvo")

# ─────────────────────────────────────────────────────────────────────────
# ETAPA 6 — Teste Cego
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("ETAPA 6 — TESTE CEGO")
print("=" * 65)

y_pred_cart = cart_final.predict(X_test)
y_pred_mlp  = mlp_final.predict(X_test)


def metricas(y_true, y_pred, nome):
    """Calcula e imprime precision, recall e f1 por classe e média."""
    print(f"\n── {nome} ──")
    print(classification_report(
        y_true, y_pred,
        target_names=["verde(0)", "amarelo(1)", "vermelho(2)", "preto(3)"],
        digits=4,
    ))
    f1_w   = f1_score(y_true, y_pred, average="weighted")
    f1_mac = f1_score(y_true, y_pred, average="macro")
    return f1_w, f1_mac


f1w_cart, f1m_cart = metricas(y_test, y_pred_cart, "CART — Teste Cego")
f1w_mlp,  f1m_mlp  = metricas(y_test, y_pred_mlp,  "MLP  — Teste Cego")

# ── Tabela comparativa final ───────────────────────────────────────────────
print("\n── TABELA COMPARATIVA — TESTE CEGO ──")
print(f"{'Métrica':<30} {'CART':>10} {'MLP':>10}")
print("-" * 52)

for avg in ["weighted", "macro"]:
    p_cart = precision_score(y_test, y_pred_cart, average=avg, zero_division=0)
    r_cart = recall_score   (y_test, y_pred_cart, average=avg, zero_division=0)
    f_cart = f1_score       (y_test, y_pred_cart, average=avg, zero_division=0)
    p_mlp  = precision_score(y_test, y_pred_mlp,  average=avg, zero_division=0)
    r_mlp  = recall_score   (y_test, y_pred_mlp,  average=avg, zero_division=0)
    f_mlp  = f1_score       (y_test, y_pred_mlp,  average=avg, zero_division=0)
    print(f"  Precisão  ({avg}){'':<8} {p_cart:>10.4f} {p_mlp:>10.4f}")
    print(f"  Recall    ({avg}){'':<8} {r_cart:>10.4f} {r_mlp:>10.4f}")
    print(f"  F1-score  ({avg}){'':<8} {f_cart:>10.4f} {f_mlp:>10.4f}")
    print()

# ── Comparação treino/val × teste (overfitting / underfitting) ─────────────
print("── COMPARAÇÃO VALIDAÇÃO × TESTE CEGO ──")
print(f"{'Modelo':<10} {'f1_val_médio':>14} {'f1_teste':>10} {'diferença':>12}")
print("-" * 48)
print(f"{'CART':<10} {cart_summary['mean_val_f1']:>14.4f} {f1w_cart:>10.4f} "
      f"{abs(cart_summary['mean_val_f1'] - f1w_cart):>12.4f}")
print(f"{'MLP':<10} {mlp_summary['mean_val_f1']:>14.4f} {f1w_mlp:>10.4f} "
      f"{abs(mlp_summary['mean_val_f1'] - f1w_mlp):>12.4f}")

# ─────────────────────────────────────────────────────────────────────────
# ETAPA 7 — Figuras: matrizes de confusão
# ─────────────────────────────────────────────────────────────────────────
nomes_curtos = ["verde", "amarelo", "vermelho", "preto"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, y_pred, titulo in zip(
    axes,
    [y_pred_cart, y_pred_mlp],
    ["CART — Teste Cego", "MLP — Teste Cego"],
):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels = nomes_curtos,
        cmap           = "Blues",
        colorbar       = False,
        ax             = ax,
    )
    ax.set_title(titulo, fontsize=11)

plt.suptitle("Matrizes de Confusão — Teste Cego", fontsize=13)
plt.tight_layout()
plt.savefig("resultados/figuras/matrizes_confusao_teste_cego.png", dpi=150)
plt.close()
print("\n→ resultados/figuras/matrizes_confusao_teste_cego.png")

# ── Figura: barras comparativas por classe ────────────────────────────────
from sklearn.metrics import f1_score as f1_per_class

f1_cart_cls = f1_score(y_test, y_pred_cart, average=None, labels=[0,1,2,3], zero_division=0)
f1_mlp_cls  = f1_score(y_test, y_pred_mlp,  average=None, labels=[0,1,2,3], zero_division=0)

x    = np.arange(4)
w    = 0.35
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.bar(x - w/2, f1_cart_cls, w, label="CART", color="steelblue")
ax2.bar(x + w/2, f1_mlp_cls,  w, label="MLP",  color="tomato")
ax2.set_xticks(x)
ax2.set_xticklabels(nomes_curtos)
ax2.set_ylabel("F1-score")
ax2.set_title("F1-score por Classe — Teste Cego (CART × MLP)")
ax2.legend()
ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("resultados/figuras/f1_por_classe_teste_cego.png", dpi=150)
plt.close()
print("→ resultados/figuras/f1_por_classe_teste_cego.png")

# ── Salvar resumo final ────────────────────────────────────────────────────
final = {
    "cart": {
        "f1_weighted_teste" : float(f1w_cart),
        "f1_macro_teste"    : float(f1m_cart),
        "f1_val_medio"      : cart_summary["mean_val_f1"],
        "f1_por_classe"     : f1_cart_cls.tolist(),
    },
    "mlp": {
        "f1_weighted_teste" : float(f1w_mlp),
        "f1_macro_teste"    : float(f1m_mlp),
        "f1_val_medio"      : mlp_summary["mean_val_f1"],
        "f1_por_classe"     : f1_mlp_cls.tolist(),
    },
}
with open("resultados/teste_cego_summary.json", "w", encoding="utf-8") as f:
    json.dump(final, f, indent=2, ensure_ascii=False)

print("→ resultados/teste_cego_summary.json")
print("\n✔  04_teste_cego.py concluído.")
