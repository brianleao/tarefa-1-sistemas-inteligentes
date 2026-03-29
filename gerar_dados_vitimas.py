import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from collections import Counter

# === Configurações fixas ===
from pathlib import Path
BASE_FOLDER = Path("../datasets/vict/10v")
OUTPUT_CSV = BASE_FOLDER / "data.csv"


def gerar_dataset_vitimas(n_vitimas=100, media_idade=35, desvio_idade=5,
                          tipo_acidente="aereo", nivel_ruido=0.02, seed=None):
    """
    Gera dataset sintético de vítimas com ruído configurável.
    nivel_ruido: float entre 0 e 1 que controla intensidade do ruído.
                 0 -> sem ruído, 1 -> ruído máximo permitido.
                 Também define probabilidade de erro em AVPU e TRI.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # porcentual de vitimas por tipo:
    # 0 (verde), 1 (amarelo), 2 (vermelho), 3 (preto)
    dist_tri_por_tipo = {
        'aereo': {0: 0.2, 1: 0.3, 2: 0.4, 3: 0.1},
        'rodoviario': {0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1},
        'ferroviario': {0: 0.3, 1: 0.3, 2: 0.3, 3: 0.1},
        'deslizamento': {0: 0.25, 1: 0.25, 2: 0.3, 3: 0.2},
        'uniforme': {0: 0.25, 1: 0.25, 2: 0.25, 3:0.25}
    }

    # faixas de probabilidade de sobrevivencia em funcao do tipo
    faixas_fuzzy = {0:(0.90,1.0),1:(0.75,0.95),2:(0.15,0.8),3:(0.0,0.25)}

    # faixas de valores ou valores para os atributos em funcao do estado
    # de triagem
    triagem_parametros = {
        0: {'fc': (60,100), 'fr':(12,20), 'pas':(110,130), 'spo2':(96,100),
            'gcs':(15,15), 'avpu':[0], 'temp':(36.5,37.4), 'pr':[1],
            'sg':[0], 'fx':[1], 'queim':[0]},
        1: {'fc':(100,120), 'fr':(20,30), 'pas':(90,110), 'spo2':(90,95),
            'gcs':(13,14), 'avpu':[0,1], 'temp':(37.0,38.5), 'pr':[1,0],
            'sg':[1,2], 'fx':[0,1], 'queim':[0]*85+[1]*10+[2]*5},
        2: {'fc':(121,160), 'fr':(31,45), 'pas':(60,89), 'spo2':(75,89),
            'gcs':(9,12), 'avpu':[2], 'temp':(34.0,35.0), 'pr':[0],
            'sg':[3], 'fx':[1], 'queim':[0]*40+[2]*30+[3]*30},
        3: {'fc':(0,0), 'fr':(0,0), 'pas':(0,0), 'spo2':(0,74),
            'gcs':(3,6), 'avpu':[3], 'temp':(25.0,34.0), 'pr':[0],
            'sg':[3], 'fx':[0,1], 'queim':[0]*50+[3]*50}
    }

    distrib = dist_tri_por_tipo[tipo_acidente]
    classificacoes = np.random.choice(list(distrib.keys()), size=n_vitimas, p=list(distrib.values()))
    idades = np.clip(np.random.normal(media_idade, desvio_idade, n_vitimas), 0, 90).astype(int)

    dados = []

    for i in range(n_vitimas):
        tri = classificacoes[i]
        idade = idades[i]
        params = triagem_parametros[tri]
        gcs_range = params['gcs']
        gcs_value = gcs_range[0] if gcs_range[0]==gcs_range[1] else np.random.randint(*gcs_range)

        def ruido_int(val, min_val, max_val):
            delta = int((max_val - min_val + 1) * nivel_ruido)
            ruido = np.random.randint(-delta, delta+1)
            return int(np.clip(val + ruido, min_val, max_val))

        def ruido_float(val, min_val, max_val):
            delta = (max_val - min_val) * nivel_ruido
            ruido = np.random.uniform(-delta, delta)
            return float(np.clip(val + ruido, min_val, max_val))

        # --- Numéricas ---
        registro = {
            'idade': ruido_int(idade, 1, 90),
            'fc': ruido_int(np.random.randint(*params['fc']) if params['fc']!=(0,0) else 0, 0, 200),
            'fr': ruido_int(np.random.randint(*params['fr']) if params['fr']!=(0,0) else 0, 0, 50),
            'pas': ruido_int(np.random.randint(*params['pas']) if params['pas']!=(0,0) else 0, 0, 200),
            'spo2': ruido_int(np.random.randint(*params['spo2']), 0, 100),
            'temp': ruido_float(np.random.uniform(*params['temp']), 25.0, 42.0)
        }

        # --- Categóricas ---
        # pr, sg, fx, queim
        for col in ['pr', 'sg', 'fx', 'queim']:
            registro[col] = random.choice(params[col])

        registro['gcs'] = ruido_int(gcs_value, 3, 15)

        # --- Ruído em AVPU ---
        avpu_val = random.choice(params['avpu'])
        if random.random() < nivel_ruido:  # com probabilidade nivel_ruido, troca para outro valor
            outros = [v for v in params['avpu'] if v != avpu_val]
            if outros:
                avpu_val = random.choice(outros)
        registro['avpu'] = avpu_val

        # --- Ruído em TRI ---
        tri_val = tri
        if random.random() < nivel_ruido:
            outros = [v for v in [0,1,2,3] if v != tri_val]
            tri_val = random.choice(outros)
        registro['tri'] = tri_val

        registro['sobr'] = round(ruido_float(np.random.uniform(*faixas_fuzzy[tri]),0.0,1.0),2)

        dados.append(registro)

    df = pd.DataFrame(dados)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDataset salvo como {OUTPUT_CSV}")

    cores = {0: 'verde', 1: 'amarelo', 2: 'vermelho', 3: 'preto'}
    contagem = Counter(df['tri'])
    print("\nNúmero de vítimas por classificação START:")
    for k in sorted(contagem.keys()):
        print(f"  {k} ({cores[k]}): {contagem[k]}")

    plt.figure(figsize=(8, 5))
    plt.hist(df['sobr'], bins=10, color='skyblue', edgecolor='black',
             weights=np.ones(n_vitimas)/n_vitimas)
    plt.title('Distribuição percentual da probabilidade de sobrevivência')
    plt.xlabel('Probabilidade de Sobrevivência')
    plt.ylabel('% de Vitimas')
    plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    return df


def main():
    # você pode alterar os parâmetros conforme necessário
    df = gerar_dataset_vitimas(
        n_vitimas=10,
        media_idade=25,
        desvio_idade=3,
        tipo_acidente="uniforme",
        nivel_ruido=0.02
    )
    print("\nPrimeiras linhas do dataset gerado:")
    print(df.head())


if __name__ == "__main__":
    main()
