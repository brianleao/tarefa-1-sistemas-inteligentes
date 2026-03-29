# -*- coding: utf-8 -*-
"""
03_mlp.py
─────────────────────────────────────────────────────────────────────────────
Treinamento e validação cruzada da Rede Neural MLP (MLPClassifier).

Entradas (features 1–10):
    idade, fc, fr, pas, spo2, temp, pr, sg, fx, queim

Saída (feature 13):
    tri  →  0=verde | 1=amarelo | 2=vermelho | 3=preto

Etapas:
  1. Carrega e normaliza os dados de treino/validação
  2. GridSearchCV com StratifiedKFold — varia hidden_layers, activation e
     learning_rate_init
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

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# ── Configurações ─────────────────────────────────────────────────────────
N_FOLDS  = 5
SEED     = 42

FEATURES = ['idade', 'fc', 'fr', 'pas', 'spo2', 'temp', 'pr', 'sg', 'fx', 'queim']
TARGET   = 'tri'

os.makedirs("modelos",              exist_ok=True)
os.makedirs("resultados",           exist_ok=True)
os.makedirs("resultados/figuras",   exist_ok=True)

# ── Carregar dados ────────────────────────────────────────────────────────
print("=" * 65)
print("MLP — VALIDAÇÃO CRUZADA")
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

# ── Pipeline: StandardScaler + MLPClassifier ──────────────────────────────
# MLP é sensível à escala das features — normalização é essencial
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp',    MLPClassifier(
        max_iter     = 500,
        random_state = SEED,
        early_stopping = True,  # evita overfitting no treinamento
        n_iter_no_change = 20,
    )),
])

# ── Grade de hiperparâmetros ───────────────────────────────────────────────
# Variações:
#   hidden_layer_sizes : arquitetura da rede (camadas × neurônios)
#   activation         : função de ativação
#   learning_rate_init : taxa de aprendizado inicial
#
# Nomenclatura: (64,) = 1 camada oculta de 64 neurônios
#               (128, 64) = 2 camadas ocultas de 128 e 64 neurônios
param_grid = {
    'mlp__hidden_layer_sizes': [
        (64,),
        (128,),
        (64, 32),
        (128, 64),
        (128, 64, 32),
    ],
    'mlp__activation':          ['relu', 'tanh'],
    'mlp__learning_rate_init':  [0.001, 0.01],
}
# 5 × 2 × 2 = 20 combinações × 5 folds = 100 fits

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

clf = GridSearchCV(
    pipe,
    param_grid,
    cv                 = cv,
    scoring            = 'f1_weighted',
    return_train_score = True,
    n_jobs             = -1,
    verbose            = 1,
)

n_combinacoes = 5 * 2 * 2
print(f"\nIniciando GridSearchCV:")
print(f"  {n_combinacoes} combinações × {N_FOLDS} folds = {n_combinacoes * N_FOLDS} fits")
print(f"  Scoring: f1_weighted  |  CV: StratifiedKFold(n={N_FOLDS})")
print(f"  Pipeline: StandardScaler → MLPClassifier(max_iter=500)\n")
clf.fit(X, y)

# ── Salvar resultados completos ────────────────────────────────────────────
results = pd.DataFrame(clf.cv_results_)
results.to_csv("resultados/mlp_cv_results.csv", index=False)

# ── Top-10 configurações ──────────────────────────────────────────────────
print("\n── TOP 10 CONFIGURAÇÕES (por f1_weighted médio de validação) ──")
cols_exibir = [
    'param_mlp__hidden_layer_sizes',
    'param_mlp__activation',
    'param_mlp__learning_rate_init',
    'mean_train_score',
    'mean_test_score',
    'std_test_score',
]
top10 = (results[cols_exibir]
         .sort_values('mean_test_score', ascending=False)
         .head(10)
         .rename(columns={
             'param_mlp__hidden_layer_sizes': 'camadas',
             'param_mlp__activation':         'ativação',
             'param_mlp__learning_rate_init': 'lr',
             'mean_train_score':              'f1_treino',
             'mean_test_score':               'f1_val',
             'std_test_score':                'f1_val_std',
         }))
print(top10.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ── Melhor modelo ─────────────────────────────────────────────────────────
best_params = clf.best_params_
best_idx    = clf.best_index_

print(f"\n── MELHOR HIPERPARAMETRIZAÇÃO ──")
for k, v in best_params.items():
    print(f"  {k:40}: {v}")

# ── Scores por fold da melhor configuração ────────────────────────────────
fold_train = np.array([results.at[best_idx, f'split{i}_train_score'] for i in range(N_FOLDS)])
fold_val   = np.array([results.at[best_idx, f'split{i}_test_score']  for i in range(N_FOLDS)])

mean_train = fold_train.mean()
mean_val   = fold_val.mean()
var_train  = np.var(fold_train)
var_val    = np.var(fold_val)

# ── Tabela de Viés ─────────────────────────────────────────────────────────
print("\n── TABELA DE VIÉS (f-score médio) ──")
print(f"{'Métrica':<30} {'MLP':>10}")
print(f"{'fs̄_t  (treino)':30} {mean_train:>10.4f}")
print(f"{'fs̄_v  (validação)':30} {mean_val:>10.4f}")
print(f"{'|fs̄_t − fs̄_v|':30} {abs(mean_train - mean_val):>10.4f}")

# ── Tabela de Variância ────────────────────────────────────────────────────
print("\n── TABELA DE VARIÂNCIA (f-score) ──")
print(f"{'Métrica':<30} {'MLP':>10}")
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
joblib.dump(clf.best_estimator_, "modelos/mlp_melhor.pkl")

summary = {
    "best_params"   : {k: str(v) for k, v in best_params.items()},
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
with open("resultados/mlp_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n→ modelos/mlp_melhor.pkl")
print("→ resultados/mlp_summary.json")
print("→ resultados/mlp_cv_results.csv")

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
ax.set_title(f"MLP — F1-score por Fold\n{best_params}")
ax.set_xticks(x)
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("resultados/figuras/mlp_fscore_por_fold.png", dpi=150)
plt.close()
print("→ resultados/figuras/mlp_fscore_por_fold.png")

# ── Figura 2: comparação CART × MLP (f-score por fold) ───────────────────
try:
    with open("resultados/cart_summary.json", encoding="utf-8") as f:
        cart = json.load(f)

    cart_train = np.array(cart["fold_train"])
    cart_val   = np.array(cart["fold_val"])

    fig2, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, treino, val, titulo in zip(
        axes,
        [cart_train, fold_train],
        [cart_val,   fold_val],
        ["CART", "MLP"],
    ):
        ax.plot(x, treino, "o-", color="steelblue", label="Treino")
        ax.plot(x, val,    "s-", color="tomato",    label="Validação")
        ax.axhline(treino.mean(), linestyle="--", color="steelblue", alpha=0.5)
        ax.axhline(val.mean(),    linestyle="--", color="tomato",    alpha=0.5)
        ax.set_title(f"{titulo}  (treino={treino.mean():.3f} | val={val.mean():.3f})")
        ax.set_xlabel("Fold")
        ax.set_ylabel("F1-score (weighted)")
        ax.set_xticks(x)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

    fig2.suptitle("Comparação CART × MLP — F1-score por Fold", fontsize=12)
    plt.tight_layout()
    plt.savefig("resultados/figuras/comparacao_cart_mlp_folds.png", dpi=150)
    plt.close()
    print("→ resultados/figuras/comparacao_cart_mlp_folds.png")

except FileNotFoundError:
    print("(cart_summary.json não encontrado — gráfico comparativo não gerado)")

print("\n✔  03_mlp.py concluído.")
