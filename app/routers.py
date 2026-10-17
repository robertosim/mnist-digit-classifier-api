"""Rotas da API (APIRouter).

Responsabilidade: expor endpoints HTTP, traduzir exceções de domínio
(vindas do service) em status codes. NÃO deve conter regra de modelo —
nada de joblib, numpy ou preprocessamento aqui.

Rotas que virão na Fase 5:
    GET  /health  → status da API e do modelo
    POST /predict → previsão do dígito
"""
from fastapi import APIRouter

router = APIRouter()
