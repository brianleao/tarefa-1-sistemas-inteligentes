import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gerar_dados_vitimas as gdv

# parametros
N_TREINO = 5000
N_TESTE = 1000
SEED_TREINO = 42
SEED_TESTE = 123
MEDIA_IDADE = 35
DESVIO_IDADE = 15
TIPO_ACIDENTE = "rodoviario"
NIVEL_RUIDO = 0.05

os.makedirs("dados", exist_ok=True)

# gera dataset de treino/validacao
print("=" * 60)
print("GERANDO DATASET DE TREINO/VALIDACAO")
print("=" * 60)
gdv.OUTPUT_CSV = Path("dados/treino_validacao.csv")
df_treino = gdv.gerar_dataset_vitimas(
    n_vitimas=N_TREINO,
    media_idade=MEDIA_IDADE,
    desvio_idade=DESVIO_IDADE,
    tipo_acidente=TIPO_ACIDENTE,
    nivel_ruido=NIVEL_RUIDO,
    seed=SEED_TREINO,
)
print(f"\nSalvo em dados/treino_validacao.csv  ({len(df_treino)} vitimas)")

# gera dataset de teste cego
print("\n" + "=" * 60)
print("GERANDO DATASET DE TESTE CEGO")
print("=" * 60)
gdv.OUTPUT_CSV = Path("dados/teste_cego.csv")
df_teste = gdv.gerar_dataset_vitimas(
    n_vitimas=N_TESTE,
    media_idade=MEDIA_IDADE,
    desvio_idade=DESVIO_IDADE,
    tipo_acidente=TIPO_ACIDENTE,
    nivel_ruido=NIVEL_RUIDO,
    seed=SEED_TESTE,
)
print(f"\nSalvo em dados/teste_cego.csv  ({len(df_teste)} vitimas)")

print("\n" + "=" * 60)
print("PARAMETROS DO DATASET")
print("=" * 60)
print(f"  Vitimas treino/validacao : {N_TREINO}")
print(f"  Vitimas teste cego       : {N_TESTE}")
print(f"  Media de idade           : {MEDIA_IDADE} anos")
print(f"  Desvio padrao de idade   : {DESVIO_IDADE} anos")
print(f"  Tipo de acidente         : {TIPO_ACIDENTE}")
print(f"  Nivel de ruido           : {NIVEL_RUIDO}")
print(f"  Semente treino           : {SEED_TREINO}")
print(f"  Semente teste            : {SEED_TESTE}")
