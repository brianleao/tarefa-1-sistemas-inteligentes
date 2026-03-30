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

N_FOLDS = 5
SEED = 42

FEATURES = ['idade', 'fc', 'fr', 'pas', 'spo2', 'temp', 'pr', 'sg', 'fx', 'queim']
TARGET = 'tri'

os.makedirs("modelos", exist_ok=True)
os.makedirs("resultados", exist_ok=True)
os.makedirs("resultados/figuras", exist_ok=True)

# carrega dados
print("=" * 65)
print("MLP - VALIDACAO CRUZADA")
print("=" * 65)

df = pd.read_csv("dados/treino_validacao.csv")
X = df[FEATURES].values
y = df[TARGET].values

print(f"Dataset carregado: {X.shape[0]} amostras | {X.shape[1]} features")
print(f"\nDistribuicao de classes (tri):")
contagem = pd.Series(y).value_counts().sort_index()
cores = {0: "verde", 1: "amarelo", 2: "vermelho", 3: "preto"}
for cls, cnt in contagem.items():
    print(f"  {cls} ({cores[cls]:>8}): {cnt:>5}  ({cnt/len(y)*100:.1f}%)")

# pipeline com normalizacao
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(
        max_iter=500,
        random_state=SEED,
        early_stopping=True,
        n_iter_no_change=20,
    )),
])

# hiperparametros
param_grid = {
    'mlp__hidden_layer_sizes': [
        (64,),
        (128,),
        (64, 32),
        (128, 64),
        (128, 64, 32),
    ],
    'mlp__activation': ['relu', 'tanh'],
    'mlp__learning_rate_init': [0.001, 0.01],
}

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

clf = GridSearchCV(
    pipe,
    param_grid,
    cv=cv,
    scoring='f1_weighted',
    return_train_score=True,
    n_jobs=-1,
    verbose=1,
)

n_combinacoes = 5 * 2 * 2
print(f"\nIniciando GridSearchCV:")
print(f"  {n_combinacoes} combinacoes x {N_FOLDS} folds = {n_combinacoes * N_FOLDS} fits")
print(f"  Scoring: f1_weighted  |  CV: StratifiedKFold(n={N_FOLDS})\n")
clf.fit(X, y)

# salva resultados
results = pd.DataFrame(clf.cv_results_)
results.to_csv("resultados/mlp_cv_results.csv", index=False)

# top-10 configuracoes
print("\nTOP 10 CONFIGURACOES (por f1_weighted medio de validacao):")
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
             'param_mlp__activation': 'ativacao',
             'param_mlp__learning_rate_init': 'lr',
             'mean_train_score': 'f1_treino',
             'mean_test_score': 'f1_val',
             'std_test_score': 'f1_val_std',
         }))
print(top10.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# melhor configuracao
best_params = clf.best_params_
best_idx = clf.best_index_

print(f"\nMELHOR HIPERPARAMETRIZACAO:")
for k, v in best_params.items():
    print(f"  {k:40}: {v}")

# scores por fold
fold_train = np.array([results.at[best_idx, f'split{i}_train_score'] for i in range(N_FOLDS)])
fold_val = np.array([results.at[best_idx, f'split{i}_test_score'] for i in range(N_FOLDS)])

mean_train = fold_train.mean()
mean_val = fold_val.mean()
var_train = np.var(fold_train)
var_val = np.var(fold_val)

# tabela de vies
print("\nTABELA DE VIES (f-score medio):")
print(f"{'Metrica':<30} {'MLP':>10}")
print(f"{'fs_t  (treino)':30} {mean_train:>10.4f}")
print(f"{'fs_v  (validacao)':30} {mean_val:>10.4f}")
print(f"{'|fs_t - fs_v|':30} {abs(mean_train - mean_val):>10.4f}")

# tabela de variancia
print("\nTABELA DE VARIANCIA (f-score):")
print(f"{'Metrica':<30} {'MLP':>10}")
print(f"{'Var_t  (treino)':30} {var_train:>10.6f}")
print(f"{'Var_v  (validacao)':30} {var_val:>10.6f}")
print(f"{'|Var_t - Var_v|':30} {abs(var_train - var_val):>10.6f}")

# f-score por fold
print("\nF-SCORE POR FOLD (melhor configuracao):")
print(f"{'Fold':>6} {'Treino':>10} {'Validacao':>12}")
print("-" * 30)
for i in range(N_FOLDS):
    print(f"{i+1:>6} {fold_train[i]:>10.4f} {fold_val[i]:>12.4f}")
print("-" * 30)
print(f"{'Media':>6} {mean_train:>10.4f} {mean_val:>12.4f}")
print(f"{'Var':>6} {var_train:>10.6f} {var_val:>12.6f}")

# salva modelo e resumo
joblib.dump(clf.best_estimator_, "modelos/mlp_melhor.pkl")

summary = {
    "best_params": {k: str(v) for k, v in best_params.items()},
    "n_folds": N_FOLDS,
    "scoring": "f1_weighted",
    "mean_train_f1": float(mean_train),
    "mean_val_f1": float(mean_val),
    "vies": float(abs(mean_train - mean_val)),
    "var_train_f1": float(var_train),
    "var_val_f1": float(var_val),
    "fold_train": fold_train.tolist(),
    "fold_val": fold_val.tolist(),
}
with open("resultados/mlp_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\nSalvo: modelos/mlp_melhor.pkl")
print("Salvo: resultados/mlp_summary.json")
print("Salvo: resultados/mlp_cv_results.csv")

# figura 1: f-score por fold
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(1, N_FOLDS + 1)
ax.plot(x, fold_train, "o-", color="steelblue", label="Treino")
ax.plot(x, fold_val, "s-", color="tomato", label="Validacao")
ax.axhline(mean_train, linestyle="--", color="steelblue", alpha=0.5,
           label=f"Media treino = {mean_train:.3f}")
ax.axhline(mean_val, linestyle="--", color="tomato", alpha=0.5,
           label=f"Media val   = {mean_val:.3f}")
ax.set_xlabel("Fold")
ax.set_ylabel("F1-score (weighted)")
ax.set_title(f"MLP - F1-score por Fold\n{best_params}")
ax.set_xticks(x)
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("resultados/figuras/mlp_fscore_por_fold.png", dpi=150)
plt.close()
print("Salvo: resultados/figuras/mlp_fscore_por_fold.png")

# figura 2: comparacao CART x MLP
try:
    with open("resultados/cart_summary.json", encoding="utf-8") as f:
        cart = json.load(f)

    cart_train = np.array(cart["fold_train"])
    cart_val = np.array(cart["fold_val"])

    fig2, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, treino, val, titulo in zip(
        axes,
        [cart_train, fold_train],
        [cart_val, fold_val],
        ["CART", "MLP"],
    ):
        ax.plot(x, treino, "o-", color="steelblue", label="Treino")
        ax.plot(x, val, "s-", color="tomato", label="Validacao")
        ax.axhline(treino.mean(), linestyle="--", color="steelblue", alpha=0.5)
        ax.axhline(val.mean(), linestyle="--", color="tomato", alpha=0.5)
        ax.set_title(f"{titulo}  (treino={treino.mean():.3f} | val={val.mean():.3f})")
        ax.set_xlabel("Fold")
        ax.set_ylabel("F1-score (weighted)")
        ax.set_xticks(x)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

    fig2.suptitle("Comparacao CART x MLP - F1-score por Fold", fontsize=12)
    plt.tight_layout()
    plt.savefig("resultados/figuras/comparacao_cart_mlp_folds.png", dpi=150)
    plt.close()
    print("Salvo: resultados/figuras/comparacao_cart_mlp_folds.png")

except FileNotFoundError:
    print("cart_summary.json nao encontrado - grafico comparativo nao gerado")
