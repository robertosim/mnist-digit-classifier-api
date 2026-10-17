"""Ponto de entrada da aplicação FastAPI.

Responsabilidade: criar a instância FastAPI, incluir os routers e
gerenciar o ciclo de vida (lifespan).

Como rodar (a partir da raiz do repositório):
    uvicorn app.main:app --reload

- "app.main" = módulo `main` dentro do pacote `app`
- "app"      = variável `app` (objeto FastAPI) dentro desse módulo
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import services
from app.routers import router

# Configura o logging ANTES de criar o logger, para que uvicorn mostre
# as mensagens de app.services e app.main no console.
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s - %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida: carrega o modelo na subida, libera na descida.

    Tudo ANTES do yield roda na subida do servidor.
    Tudo DEPOIS do yield roda no encerramento (SIGINT, SIGTERM).
    """
    # --- Subida ---
    try:
        app.state.modelo = services.carregar_modelo()
    except services.ModeloNaoCarregadoError as exc:
        logger.error("Falha ao carregar modelo: %s", exc)
        app.state.modelo = None

    yield

    # --- Descida ---
    app.state.modelo = None
    logger.info("Modelo liberado.")


app = FastAPI(
    title="MNIST Classifier API",
    version="0.1.0",
    description="API REST para classificação de dígitos MNIST com MLP.",
    lifespan=lifespan,
)

app.include_router(router)
