"""Ponto de entrada da aplicação FastAPI.

Responsabilidade: criar a instância FastAPI, incluir os routers e,
na Fase 3, gerenciar o ciclo de vida (lifespan).

Como rodar (a partir da raiz do repositório):
    uvicorn app.main:app --reload

- "app.main" = módulo `main` dentro do pacote `app`
- "app"      = variável `app` (objeto FastAPI) dentro desse módulo
"""
from fastapi import FastAPI

from app.routers import router

app = FastAPI(
    title="MNIST Classifier API",
    version="0.1.0",
    description="API REST para classificação de dígitos MNIST com MLP.",
)

app.include_router(router)
