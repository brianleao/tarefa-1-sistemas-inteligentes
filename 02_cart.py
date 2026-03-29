# -*- coding: utf-8 -*-
"""
02_cart.py
─────────────────────────────────────────────────────────────────────────────
Treinamento e validação cruzada do classificador CART (Árvore de Decisão).

Entradas (features 1–10):
    idade, fc, fr, pas, spo2, temp, pr, sg, fx, queim

Saída (feature 13):
    tri  →  0=verde | 1=amarelo | 2=vermelho | 3=preto

Proibido usar como entrada: gcs (11), avpu (12), tri (13), sobr (14).

Etapas:
  1. Carrega dados de treino/validação
  2. GridSearchCV com StratifiedKFold — varia criterion, max_depth e
     min_samples_leaf
  3. Exibe top-10 configurações e a melhor hiperparametrização
  4. Calcula viés e variância do f-score por fold
  5. Salva modelo, parâmetros e gráficos em resultados/
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import joblib

# ── Configurações ─────────────────────────────────────────────────────────
N_FOLDS  = 5
SEED     = 42

FEATURES = ['idade', 'fc', 'fr', 'pas', 'spo2', 'temp', 'pr', 'sg', 'fx', 'queim']
TARGET   = 'tri'
NOMES_CLASSES = ['verde(0)', 'amarelo(1)', 'vermelho(2)', 'preto(3)']

os.makedirs("modelos",              exist_ok=True)
os.makedirs("resultados",           exist_ok=True)
os.makedirs("resultados/figuras",   exist_ok=True)

# ── Carregar dados ────────────────────────────────────────────────────────
print("=" * 65)
print("CART — VALIDAÇÃO CRUZADA")
print("=" * 65)

df = pd.read_csv("dados/treino_validacao.csv")
X  = df[FEATURES].values
y  = df[TARGET].values

print(f"Dataset carregado: {X.shape[0]} amostras | {X.shape[1]} features")
print(f"\nDistribuição de classes (tri):")
contagem = pd.Series(y).value_counts().sort_index()
cores = {0: "verde", 1: "amarelo", 2: "vermelho", 3: "preto"}
for cls, cnt in contagem.items():
    print(f"  {cls} ({cores[cls]:>8}): {cnt:>5}  ({cnt/len(y)*100:.1f}%)")

# ── Grade de hiperparâmetros ───────────────────────────────────────────────
# 2 × 5 × 4 = 40 combinações  ×  5 folds  =  200 fits
param_grid = {
    'criterion':        ['gini', 'entropy'],
    'max_depth':        [3, 5, 10, 15, None],
    'min_samples_leaf': [4, 8, 16, 32],
}

cv    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
model = DecisionTreeClassifier(random_state=SEED)

clf = GridSearchCV(
    model,
    param_grid,
    cv                = cv,
    scoring           = 'f1_weighted',
    return_train_score= True,
    n_jobs            = -1,
    verbose           = 1,
)

n_combinacoes = 2 * 5 * 4
print(f"\nIniciando GridSearchCV:")
print(f"  {n_combinacoes} combinações × {N_FOLDS} folds = {n_combinacoes * N_FOLDS} fits")
print(f"  Scoring: f1_weighted  |  CV: StratifiedKFold(n={N_FOLDS})\n")
clf.fit(X, y)

# ── Salvar resultados completos ────────────────────────────────────────────
results = pd.DataFrame(clf.cv_results_)
results.to_csv("resultados/cart_cv_results.csv", index=False)

# ── Top-10 configurações ──────────────────────────────────────────────────
print("\n── TOP 10 CONFIGURAÇÕES (por f1_weighted médio de validação) ──")
cols_exibir = [
    'param_criterion', 'param_max_depth', 'param_min_samples_leaf',
    'mean_train_score', 'mean_test_score', 'std_test_score',
]
top10 = (results[cols_exibir]
         .sort_values('mean_test_score', ascending=False)
         .head(10)
         .rename(columns={
             'param_criterion':        'criterion',
             'param_max_depth':        'max_depth',
             'param_min_samples_leaf': 'min_samples_leaf',
             'mean_train_score':       'f1_treino',
             'mean_test_score':        'f1_val',
             'std_test_score':         'f1_val_std',
         }))
print(top10.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ── Melhor modelo ─────────────────────────────────────────────────────────
best_params = clf.best_params_
best_idx    = clf.best_index_

print(f"\n── MELHOR HIPERPARAMETRIZAÇÃO ──")
for k, v in best_params.items():
    print(f"  {k:20}: {v}")

# ── Scores por fold da melhor configuração ────────────────────────────────
fold_train = np.array([results.at[best_idx, f'split{i}_train_score'] for i in range(N_FOLDS)])
fold_val   = np.array([results.at[best_idx, f'split{i}_test_score']  for i in range(N_FOLDS)])

mean_train = fold_train.mean()
mean_val   = fold_val.mean()
var_train  = np.var(fold_train)   # 1/k * Σ(fs_i − fs̄)²  (conforme enunciado)
var_val    = np.var(fold_val)

# ── Tabela de Viés ─────────────────────────────────────────────────────────
print("\n── TABELA DE VIÉS (f-score médio) ──")
print(f"{'Métrica':<30} {'CART':>10}")
print(f"{'fs̄_t  (treino)':30} {mean_train:>10.4f}")
print(f"{'fs̄_v  (validação)':30} {mean_val:>10.4f}")
print(f"{'|fs̄_t − fs̄_v|':30} {abs(mean_train - mean_val):>10.4f}")

# ── Tabela de Variância ────────────────────────────────────────────────────
print("\n── TABELA DE VARIÂNCIA (f-score) ──")
print(f"{'Métrica':<30} {'CART':>10}")
print(f"{'Var_t  (treino)':30} {var_train:>10.6f}")
print(f"{'Var_v  (validação)':30} {var_val:>10.6f}")
print(f"{'|Var_t − Var_v|':30} {abs(var_train - var_val):>10.6f}")

# ── Detalhamento por fold ──────────────────────────────────────────────────
print("\n── F-SCORE POR FOLD (melhor configuração) ──")
print(f"{'Fold':>6} {'Treino':>10} {'Validação':>12}")
print("-" * 30)
for i in range(N_FOLDS):
    print(f"{i+1:>6} {fold_train[i]:>10.4f} {fold_val[i]:>12.4f}")
print("-" * 30)
print(f"{'Média':>6} {mean_train:>10.4f} {mean_val:>12.4f}")
print(f"{'Var':>6} {var_train:>10.6f} {var_val:>12.6f}")

# ── Persistência ──────────────────────────────────────────────────────────
joblib.dump(clf.best_estimator_, "modelos/cart_melhor.pkl")

summary = {
    "best_params"   : {k: (v if v is not None else "None") for k, v in best_params.items()},
    "n_folds"       : N_FOLDS,
    "scoring"       : "f1_weighted",
    "mean_train_f1" : float(mean_train),
    "mean_val_f1"   : float(mean_val),
    "vies"          : float(abs(mean_train - mean_val)),
    "var_train_f1"  : float(var_train),
    "var_val_f1"    : float(var_val),
    "fold_train"    : fold_train.tolist(),
    "fold_val"      : fold_val.tolist(),
}
with open("resultados/cart_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n→ modelos/cart_melhor.pkl")
print("→ resultados/cart_summary.json")
print("→ resultados/cart_cv_results.csv")

# ── Figura 1: f-score por fold ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(1, N_FOLDS + 1)
ax.plot(x, fold_train, "o-", color="steelblue", label="Treino")
ax.plot(x, fold_val,   "s-", color="tomato",    label="Validação")
ax.axhline(mean_train, linestyle="--", color="steelblue", alpha=0.5,
           label=f"Média treino = {mean_train:.3f}")
ax.axhline(mean_val,   linestyle="--", color="tomato",    alpha=0.5,
           label=f"Média val   = {mean_val:.3f}")
ax.set_xlabel("Fold")
ax.set_ylabel("F1-score (weighted)")
ax.set_title(f"CART — F1-score por Fold\n{best_params}")
ax.set_xticks(x)
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("resultados/figuras/cart_fscore_por_fold.png", dpi=150)
plt.close()
print("→ resultados/figuras/cart_fscore_por_fold.png")

# ── Figura 2: Árvore de decisão (visualização limitada a depth=4) ─────────
fig2, ax2 = plt.subplots(figsize=(24, 10))
plot_tree(
    clf.best_estimator_,
    feature_names = FEATURES,
    class_names   = ["verde", "amarelo", "vermelho", "preto"],
    filled        = True,
    rounded       = True,
    fontsize      = 6,
    max_depth     = 4,   # limita profundidade para legibilidade
    ax            = ax2,
)
ax2.set_title(f"CART — Árvore Aprendida (exibindo até depth=4)\n{best_params}", fontsize=10)
plt.tight_layout()
plt.savefig("resultados/figuras/cart_arvore.png", dpi=120)
plt.close()
print("→ resultados/figuras/cart_arvore.png")

# ── Figura 3: Importância das features ────────────────────────────────────
importances = clf.best_estimator_.feature_importances_
ordem = np.argsort(importances)[::-1]
fig3, ax3 = plt.subplots(figsize=(9, 4))
ax3.bar(range(len(FEATURES)), importances[ordem], color="steelblue")
ax3.set_xticks(range(len(FEATURES)))
ax3.set_xticklabels([FEATURES[i] for i in ordem], rotation=30, ha="right")
ax3.set_ylabel("Importância (Gini/Entropy)")
ax3.set_title("CART — Importância das Features")
ax3.grid(True, axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("resultados/figuras/cart_feature_importance.png", dpi=150)
plt.close()
print("→ resultados/figuras/cart_feature_importance.png")

print("\n✔  02_cart.py concluído.")
