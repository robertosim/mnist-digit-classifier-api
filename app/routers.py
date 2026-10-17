"""Rotas da API (APIRouter).

Responsabilidade: expor endpoints HTTP, traduzir exceções de domínio
(vindas do service) em status codes. NÃO deve conter regra de modelo —
nada de joblib, numpy ou preprocessamento aqui.

Rotas que virão na Fase 5:
    GET  /v1/health  → status da API e do modelo
    POST /v1/predict → previsão do dígito
"""
from fastapi import APIRouter

router = APIRouter()
