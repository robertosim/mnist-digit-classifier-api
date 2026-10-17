"""Modelos Pydantic (schemas) da API.

Responsabilidade: definir a forma EXATA do JSON aceito e retornado.
Pydantic valida automaticamente e gera a documentação OpenAPI.
Nenhuma regra de negócio ou HTTP deve aparecer aqui.
"""
from pydantic import BaseModel


class PredictRequest(BaseModel):
    """Corpo da requisição de previsão (Fase 4)."""
    pass


class PredictResponse(BaseModel):
    """Corpo da resposta de previsão (Fase 4)."""
    pass


class HealthResponse(BaseModel):
    """Resposta do endpoint de saúde (Fase 4)."""
    pass


class ErrorResponse(BaseModel):
    """Resposta padrão de erro (Fase 4)."""
    pass
