# -*- coding: utf-8 -*-
"""
01_gerar_dados.py
─────────────────────────────────────────────────────────────────────────────
Gera os dois datasets necessários para o trabalho:
  • dados/treino_validacao.csv  → usado nas etapas 2 e 3 (CART e MLP)
  • dados/teste_cego.csv        → usado SOMENTE na etapa 6 (teste cego)

ATENÇÃO: o dataset de teste cego NÃO deve ser usado em nenhuma etapa de
treino ou validação para evitar data leakage.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # evita abrir janela; salva figuras em arquivo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gerar_dados_vitimas as gdv

# ── Parâmetros ────────────────────────────────────────────────────────────
N_TREINO      = 5000          # entre 2.000 e 10.000 conforme o enunciado
N_TESTE       = 1000          # dataset de teste cego (fixo pelo enunciado)
SEED_TREINO   = 42
SEED_TESTE    = 123
MEDIA_IDADE   = 35            # anos
DESVIO_IDADE  = 15            # desvio padrão
TIPO_ACIDENTE = "rodoviario"  # distribuição de classes usada
NIVEL_RUIDO   = 0.05          # 5% de ruído
# ─────────────────────────────────────────────────────────────────────────

os.makedirs("dados", exist_ok=True)

# ── Treino / Validação ────────────────────────────────────────────────────
print("=" * 60)
print("GERANDO DATASET DE TREINO/VALIDAÇÃO")
print("=" * 60)
gdv.OUTPUT_CSV = Path("dados/treino_validacao.csv")
df_treino = gdv.gerar_dataset_vitimas(
    n_vitimas     = N_TREINO,
    media_idade   = MEDIA_IDADE,
    desvio_idade  = DESVIO_IDADE,
    tipo_acidente = TIPO_ACIDENTE,
    nivel_ruido   = NIVEL_RUIDO,
    seed          = SEED_TREINO,
)
print(f"\n→ Salvo em dados/treino_validacao.csv  ({len(df_treino)} vítimas)")

# ── Teste Cego ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("GERANDO DATASET DE TESTE CEGO")
print("=" * 60)
gdv.OUTPUT_CSV = Path("dados/teste_cego.csv")
df_teste = gdv.gerar_dataset_vitimas(
    n_vitimas     = N_TESTE,
    media_idade   = MEDIA_IDADE,
    desvio_idade  = DESVIO_IDADE,
    tipo_acidente = TIPO_ACIDENTE,
    nivel_ruido   = NIVEL_RUIDO,
    seed          = SEED_TESTE,
)
print(f"\n→ Salvo em dados/teste_cego.csv  ({len(df_teste)} vítimas)")

# ── Resumo dos parâmetros (para o relatório) ──────────────────────────────
print("\n" + "=" * 60)
print("PARÂMETROS DE CRIAÇÃO DO DATASET")
print("=" * 60)
print(f"  Vítimas — treino/validação : {N_TREINO}")
print(f"  Vítimas — teste cego       : {N_TESTE}")
print(f"  Média de idade             : {MEDIA_IDADE} anos")
print(f"  Desvio padrão de idade     : {DESVIO_IDADE} anos")
print(f"  Tipo de acidente           : {TIPO_ACIDENTE}")
print(f"  Nível de ruído             : {NIVEL_RUIDO}")
print(f"  Semente treino             : {SEED_TREINO}")
print(f"  Semente teste              : {SEED_TESTE}")
print("\n✔  01_gerar_dados.py concluído.")
