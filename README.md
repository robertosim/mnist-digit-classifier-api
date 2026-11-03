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
│   ├── routers.py         # Endpoints e status codes (Fase 5)
│   ├── schemas.py         # Modelos Pydantic (Fase 4)
│   └── services.py        # Carregamento do modelo e inferência (Fase 5)
├── client/                # Cliente desktop (Tkinter) para desenhar e prever
│   ├── __init__.py
│   ├── api.py             # HTTP para /health e /predict (urllib, sem dependência)
│   ├── cursores.py        # Cursor em forma de caneta/borracha + ícones (Pillow)
│   ├── interface.py       # Janela principal, painel de resultado, fila assíncrona
│   ├── main.py            # Entry point do cliente (--servidor)
│   ├── preprocessamento.py # Quadro desenhado -> 784 pixels no formato MNIST
│   └── quadro.py          # Quadro negro de desenho (canvas + buffer PIL)
├── data/
│   └── meus_digitos/      # Dígitos 28x28 do Desafio C (ver README.md da pasta)
├── models/
│   ├── mnist_mlp_pipeline.joblib   # Pipeline serializado
│   └── metrics.json                # Métricas do modelo
├── tests/
│   ├── test_api.py              # Testes dos endpoints (Fase 5)
│   ├── test_cliente.py          # Testes do cliente desktop
│   └── test_schemas.py          # Testes dos schemas (Fase 4)
├── mnist_api_fastapi.ipynb      # Notebook (Fases 1-5)
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

## Cliente desktop

Aplicativo em **Python + Tkinter** (biblioteca padrão — sem dependência nova)
que conecta na API, desenha num quadro negro e devolve o dígito predito com a
confiança. Só precisa do Pillow, que já é dependência do projeto.

```bash
# com o servidor no ar (uvicorn app.main:app --reload), em outro terminal:
python -m client.main

# apontar para outro endereço:
python -m client.main --servidor http://127.0.0.1:8000
```

**Como usar:** escolha a **caneta** (cor e espessura livres) e desenhe um
dígito; a **borracha** troca o cursor e apaga (pinta de preto, a cor do fundo).
Clique em **Prever dígito** — a imagem é tratada no cliente (escala de cinza,
normalização de brilho, recorte, redimensionamento para 20×20 e centralização
pelo centro de massa, como o MNIST) e enviada para `POST /predict`.

| Atalho | Ação |
|--------|------|
| `Enter` | prever |
| `C` | caneta |
| `B` | borracha |
| `Del` | limpar quadro |

> **Dica de acurácia:** traço **grosso** (36–60 px) e figura bem distribuída
> no quadro melhoram muito a previsão — é assim que os dígitos do MNIST são
> (finos no papel, mas normalizados para ~20×20 de área cheia). Com traço fino
> demais, o modelo erra mais; o painel lateral mostra a imagem 28×28 que foi
> enviada, para você conferir o que o servidor viu.

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

Exemplo com `curl` (resposta **200 OK**):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"pixels": [0, 0, 13, 3, 18, ..., 0]}'   # 784 inteiros
```

```json
{ "digito": 1, "confianca": 0.9994,
  "probabilidades": [0.0, 0.999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] }
```

`GET /health` responde `200` sempre: `{"status": "ok", "model_loaded": true}`
(ou `{"status": "indisponivel", "model_loaded": false}` se a carga falhar).

## Endpoints

| Método | Rota | Sucesso | Erros mapeados |
|--------|------|---------|----------------|
| GET | `/health` | **200** sempre: `{"status": "ok", "model_loaded": true}` | — |
| POST | `/predict` | **200** com `PredictResponse` | **400**, **422**, **500** |

### Semântica dos códigos (Fase 5)

| Status | Quando acontece | De onde vem |
|--------|-----------------|-------------|
| **200 OK** | previsão devolvida no contrato de saída / health respondido | `routers.py` |
| **400 Bad Request** | imagem inválida detectada pela camada de serviço (ex.: nº de pixels errado) | `services.ImagemInvalidaError` → `HTTPException(400)` |
| **422 Unprocessable** | payload fora do contrato (tipo, tamanho, faixa, campo extra) — **antes** de chegar ao modelo | Pydantic/FastAPI (`schemas.py`) |
| **500 Internal Server Error** | modelo ausente da RAM **ou** erro inesperado, com `{"detail": ...}` | `HTTPException(500)` + handler global de `Exception` em `main.py` |
| **404 Not Found** | rota inexistente | FastAPI/Starlette |

A API **não cai**: falha no carregamento do modelo vira `model_loaded: false`
no `/health` e 500 no `/predict`; qualquer exceção não prevista é capturada
pelo handler global, que grava o stack trace nos logs e responde 500 padronizado.

## Modelo

- **Algoritmo:** MLPClassifier (scikit-learn)
- **Acurácia (teste):** 97.54%
- **F1-Score (teste):** 97.54%
- **ROC-AUC OVR (teste):** 99.94%
- **Artefato:** `models/mnist_mlp_pipeline.joblib`

## Notebook

O notebook `mnist_api_fastapi.ipynb` documenta as Fases 1–5:
1. Treino do Pipeline com serialização via joblib
2. Estrutura modular da API (schemas, services, routers, main)
3. Ciclo de vida com `lifespan` (modelo carregado uma única vez)
4. Schemas Pydantic estritos (entrada/saída) e prova do HTTP 422
5. Endpoints reais: `/health`, `/predict` e o mapeamento de 200/400/422/500

Execute com o **diretório de trabalho na raiz** do repositório.

## Testes

```bash
# Fase 5: endpoints e status codes (200, 400, 422, 404 e 500)
python tests/test_api.py

# Fase 4: schemas de entrada e saída (payloads válidos e inválidos)
python tests/test_schemas.py

# Cliente desktop: tratamento da imagem, cursores, API e fila assíncrona
python tests/test_cliente.py
```

Os três arquivos também rodam com `pytest tests/` caso o pytest esteja
instalado. Juntos são **38 casos**, incluindo o `POST /predict` real
respondendo 422 (prova do critério 7 da rubrica na rota de produção).

## Arquitetura

```
schemas.py   services.py     (independentes, sem importar app.*)
      \         /
       routers.py            (importa os dois)
           |
        main.py              (importa routers)
```

- **services** não conhece HTTP — pode ser usado por CLI ou notebook
- **routers** traduz exceções de domínio em status codes (400/422/500)
- **main** gerencia o ciclo de vida (lifespan), inclui os routers e registra
  o handler global que transforma qualquer exceção não prevista em 500 JSON
