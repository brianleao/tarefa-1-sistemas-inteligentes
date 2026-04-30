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

N_FOLDS = 5
SEED = 42

FEATURES = ['idade', 'fc', 'fr', 'pas', 'spo2', 'temp', 'pr', 'sg', 'fx', 'queim']
TARGET = 'tri'

os.makedirs("modelos", exist_ok=True)
os.makedirs("resultados", exist_ok=True)
os.makedirs("resultados/figuras", exist_ok=True)

# carrega dados
print("=" * 65)
print("CART - VALIDACAO CRUZADA")
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

# hiperparametros
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 10, 15, None],
    'min_samples_leaf': [4, 8, 16, 32],
}

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
model = DecisionTreeClassifier(random_state=SEED)

clf = GridSearchCV(
    model,
    param_grid,
    cv=cv,
    scoring='f1_macro',
    return_train_score=True,
    n_jobs=-1,
    verbose=1,
)

n_combinacoes = 2 * 5 * 4
print(f"\nIniciando GridSearchCV:")
print(f"  {n_combinacoes} combinacoes x {N_FOLDS} folds = {n_combinacoes * N_FOLDS} fits")
print(f"  Scoring: f1_macro  |  CV: StratifiedKFold(n={N_FOLDS})\n")
clf.fit(X, y)

# salva resultados
results = pd.DataFrame(clf.cv_results_)
results.to_csv("resultados/cart_cv_results.csv", index=False)

# top-10 configuracoes
print("\nTOP 10 CONFIGURACOES (por f1_macro medio de validacao):")
cols_exibir = [
    'param_criterion', 'param_max_depth', 'param_min_samples_leaf',
    'mean_train_score', 'mean_test_score', 'std_test_score',
]
top10 = (results[cols_exibir]
         .sort_values('mean_test_score', ascending=False)
         .head(10)
         .rename(columns={
             'param_criterion': 'criterion',
             'param_max_depth': 'max_depth',
             'param_min_samples_leaf': 'min_samples_leaf',
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
    print(f"  {k:20}: {v}")

# scores por fold
fold_train = np.array([results.at[best_idx, f'split{i}_train_score'] for i in range(N_FOLDS)])
fold_val = np.array([results.at[best_idx, f'split{i}_test_score'] for i in range(N_FOLDS)])

mean_train = fold_train.mean()
mean_val = fold_val.mean()
var_train = np.var(fold_train)
var_val = np.var(fold_val)

# tabela de vies
print("\nTABELA DE VIES (f-score medio):")
print(f"{'Metrica':<30} {'CART':>10}")
print(f"{'fs_t  (treino)':30} {mean_train:>10.4f}")
print(f"{'fs_v  (validacao)':30} {mean_val:>10.4f}")
print(f"{'|fs_t - fs_v|':30} {abs(mean_train - mean_val):>10.4f}")

# tabela de variancia
print("\nTABELA DE VARIANCIA (f-score):")
print(f"{'Metrica':<30} {'CART':>10}")
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
joblib.dump(clf.best_estimator_, "modelos/cart_melhor.pkl")

summary = {
    "best_params": {k: (v if v is not None else "None") for k, v in best_params.items()},
    "n_folds": N_FOLDS,
    "scoring": "f1_macro",
    "mean_train_f1": float(mean_train),
    "mean_val_f1": float(mean_val),
    "vies": float(abs(mean_train - mean_val)),
    "var_train_f1": float(var_train),
    "var_val_f1": float(var_val),
    "fold_train": fold_train.tolist(),
    "fold_val": fold_val.tolist(),
}
with open("resultados/cart_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\nSalvo: modelos/cart_melhor.pkl")
print("Salvo: resultados/cart_summary.json")
print("Salvo: resultados/cart_cv_results.csv")

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
ax.set_title(f"CART - F1-score por Fold\n{best_params}")
ax.set_xticks(x)
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("resultados/figuras/cart_fscore_por_fold.png", dpi=150)
plt.close()
print("Salvo: resultados/figuras/cart_fscore_por_fold.png")

# figura 2: arvore de decisao
fig2, ax2 = plt.subplots(figsize=(24, 10))
plot_tree(
    clf.best_estimator_,
    feature_names=FEATURES,
    class_names=["verde", "amarelo", "vermelho", "preto"],
    filled=True,
    rounded=True,
    fontsize=6,
    max_depth=4,
    ax=ax2,
)
ax2.set_title(f"CART - Arvore Aprendida (ate depth=4)\n{best_params}", fontsize=10)
plt.tight_layout()
plt.savefig("resultados/figuras/cart_arvore.png", dpi=120)
plt.close()
print("Salvo: resultados/figuras/cart_arvore.png")

# figura 3: importancia das features
importances = clf.best_estimator_.feature_importances_
ordem = np.argsort(importances)[::-1]
fig3, ax3 = plt.subplots(figsize=(9, 4))
ax3.bar(range(len(FEATURES)), importances[ordem], color="steelblue")
ax3.set_xticks(range(len(FEATURES)))
ax3.set_xticklabels([FEATURES[i] for i in ordem], rotation=30, ha="right")
ax3.set_ylabel("Importancia (Gini/Entropy)")
ax3.set_title("CART - Importancia das Features")
ax3.grid(True, axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("resultados/figuras/cart_feature_importance.png", dpi=150)
plt.close()
print("Salvo: resultados/figuras/cart_feature_importance.png")
