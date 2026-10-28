"""Modelos Pydantic (schemas) da API.

Responsabilidade: definir a forma EXATA do JSON aceito e retornado.
Pydantic valida automaticamente e gera a documentação OpenAPI (/docs).
Nenhuma regra de negócio ou HTTP deve aparecer aqui.

Contrato de entrada: 784 pixels inteiros (imagem 28x28 achatada), cada um de 0 a 255.
Dados fora desse contrato são barrados pelo Pydantic (o FastAPI responde 422)
ANTES de chegar ao modelo.
"""
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

N_PIXELS = 784  # 28 x 28
N_CLASSES = 10  # dígitos 0 a 9

# strict=True: só aceita inteiro de verdade (rejeita "12", 12.5, 12.0 e True).
Pixel = Annotated[int, Field(strict=True, ge=0, le=255)]
Probabilidade = Annotated[float, Field(ge=0.0, le=1.0)]


def _exemplo_pixels() -> list[int]:
    """Imagem de exemplo para o Swagger: um traço vertical (parece um '1')."""
    pixels = [0] * N_PIXELS
    for linha in range(4, 24):
        pixels[linha * 28 + 14] = 255
    return pixels


class PredictRequest(BaseModel):
    """Corpo da requisição de previsão."""

    model_config = ConfigDict(
        extra="forbid",  # campo desconhecido no JSON vira erro 422
        json_schema_extra={"example": {"pixels": _exemplo_pixels()}},
    )

    pixels: list[Pixel] = Field(
        min_length=N_PIXELS,
        max_length=N_PIXELS,
        description="Imagem 28x28 achatada em 784 inteiros (0 = preto, 255 = branco).",
    )


class PredictResponse(BaseModel):
    """Corpo da resposta de previsão.

    Os campos de saída ficam SEM strict de propósito: a Fase 5 devolve
    escalares do NumPy (np.int64 do argmax, np.float64 do predict_proba)
    e, no modo laxo, o Pydantic os converte sozinho.
    """

    digito: int = Field(ge=0, le=9, description="Dígito previsto (0 a 9).")
    confianca: float = Field(
        ge=0.0, le=1.0, description="Probabilidade da classe prevista (0 a 1)."
    )
    probabilidades: list[Probabilidade] = Field(
        min_length=N_CLASSES,
        max_length=N_CLASSES,
        description="Probabilidade de cada dígito, do 0 ao 9 (nessa ordem).",
    )


class HealthResponse(BaseModel):
    """Resposta do endpoint de saúde."""

    status: str = Field(description="Estado do serviço.", examples=["ok"])
    model_loaded: bool = Field(description="True se o modelo está carregado na RAM.")


class ErrorResponse(BaseModel):
    """Resposta padrão de erro."""

    detail: str = Field(description="Mensagem de erro legível.", examples=["Modelo não carregado."])
