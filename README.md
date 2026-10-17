# MNIST Classifier API

API REST para classificação de dígitos MNIST usando MLP (scikit-learn) e FastAPI.

## Requisitos

- Python 3.13+
- `pip install -r requirements.txt`

## Estrutura

```
.
├── app/
│   ├── __init__.py
│   ├── main.py            # Ponto de entrada + lifespan
│   ├── preprocessing.py   # Normalização compartilhada (notebook + API)
│   ├── routers.py         # Endpoints (APIRouter)
│   ├── schemas.py         # Modelos Pydantic
│   └── services.py        # Carregamento do modelo e inferência
├── data/
│   └── meus_digitos/      # Dígitos desenhados para teste manual
├── models/
│   ├── mnist_mlp_pipeline.joblib   # Pipeline serializado
│   └── metrics.json                # Métricas do modelo
├── mnist_api_fastapi.ipynb         # Notebook (Fases 1-3)
├── requirements.txt
└── README.md
```

## Como executar

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Subir o servidor (a partir da raiz do repositório)
uvicorn app.main:app --reload
```

Documentação interativa: http://127.0.0.1:8000/docs

## Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/health` | Status da API e do modelo |
| POST | `/predict` | Previsão do dígito |

## Modelo

- **Algoritmo:** MLPClassifier (scikit-learn)
- **Acurácia (teste):** 97.54%
- **F1-Score (teste):** 97.54%
- **ROC-AUC OVR (teste):** 99.94%
- **Artefato:** `models/mnist_mlp_pipeline.joblib`

## Notebook

O notebook `mnist_api_fastapi.ipynb` documenta as Fases 1–3:
1. Treino do Pipeline com serialização via joblib
2. Estrutura modular da API (schemas, services, routers, main)
3. Ciclo de vida com `lifespan` (modelo carregado uma única vez)

Execute com o **diretório de trabalho na raiz** do repositório.

## Arquitetura

```
schemas.py   services.py     (independentes, sem importar app.*)
      \         /
       routers.py            (importa os dois)
           |
        main.py              (importa routers)
```

- **services** não conhece HTTP — pode ser usado por CLI ou notebook
- **routers** traduz exceções de domínio em status codes
- **main** gerencia o ciclo de vida (lifespan) e inclui os routers
