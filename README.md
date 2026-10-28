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
│   ├── schemas.py         # Modelos Pydantic (Fase 4)
│   └── services.py        # Carregamento do modelo e inferência
├── data/
│   └── meus_digitos/      # Dígitos 28x28 do Desafio C (ver README.md da pasta)
├── models/
│   ├── mnist_mlp_pipeline.joblib   # Pipeline serializado
│   └── metrics.json                # Métricas do modelo
├── tests/
│   └── test_schemas.py             # Testes dos schemas (Fase 4)
├── mnist_api_fastapi.ipynb         # Notebook (Fases 1-4)
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

## Payload do `POST /predict`

```text
{ "pixels": [0, 0, 13, 3, 18, ..., 0] }   # 784 inteiros, de 0 a 255
```

Regras do contrato (Fase 4):

- exatamente **784 inteiros** (28 × 28 achatados, linha a linha), de 0 a 255;
- **sem ponto decimal**: o schema de entrada é estrito (`strict=True`), então
  `255.0`, `"255"` e `true` também geram **HTTP 422**, mesmo com valor válido —
  é a contrapartida de não aceitar conversão implícita;
- converta o array para inteiro antes de enviar:
  `{"pixels": imagem.astype("uint8").ravel().tolist()}`;
- campos extras no corpo também geram 422.

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

O notebook `mnist_api_fastapi.ipynb` documenta as Fases 1–4:
1. Treino do Pipeline com serialização via joblib
2. Estrutura modular da API (schemas, services, routers, main)
3. Ciclo de vida com `lifespan` (modelo carregado uma única vez)
4. Schemas Pydantic estritos (entrada/saída) e prova do HTTP 422

Execute com o **diretório de trabalho na raiz** do repositório.

## Testes

```bash
# Valida os schemas da Fase 4 (payloads válidos e inválidos, entrada e saída)
python tests/test_schemas.py
```

O mesmo arquivo roda com `pytest tests/test_schemas.py` caso o pytest seja instalado.

> **Pendência da Fase 5:** somar em `tests/` um teste do `POST /predict` real
> respondendo 422. Hoje a prova do 422 está no mini-app descartável da seção 9
> do notebook; o critério 7 pede 422 "de forma consistente" na rota de produção.

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
