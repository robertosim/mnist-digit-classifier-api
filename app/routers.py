"""Rotas da API (APIRouter).

Responsabilidade: expor endpoints HTTP, traduzir exceções de domínio
(vindas do service) em status codes. NÃO deve conter regra de modelo —
nada de joblib, numpy ou preprocessamento aqui.

Semântica de status da Fase 5:
    GET  /health  → 200 sempre (o corpo diz se o modelo está na RAM)
    POST /predict → 200 OK       previsão devolvida no contrato de saída
                   → 400         imagem inválida (ImagemInvalidaError)
                   → 422         payload fora do contrato (Pydantic, antes
                                 de chegar ao modelo — ver schemas.py)
                   → 500         modelo ausente da RAM ou erro inesperado
                                 (handler global em main.py)
"""
from fastapi import APIRouter, HTTPException, Request, status

from app import services
from app.schemas import ErrorResponse, HealthResponse, PredictRequest, PredictResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Checagem de integridade do serviço",
    tags=["monitoramento"],
)
def saude(request: Request) -> HealthResponse:
    """Relata se o serviço está ativo e se o modelo reside na RAM.

    Sempre responde 200 OK: quem consulta saúde quer CONHECER o estado,
    não receber um erro. O campo `model_loaded` é que diz se o modelo
    carregou no lifespan.
    """
    modelo = getattr(request.app.state, "modelo", None)
    return HealthResponse(
        status="ok" if modelo is not None else "indisponivel",
        model_loaded=modelo is not None,
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Previsão do dígito",
    tags=["inferencia"],
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Imagem inválida detectada pela camada de serviço.",
        },
        500: {
            "model": ErrorResponse,
            "description": "Modelo ausente da RAM ou erro inesperado.",
        },
    },
)
def prever_digito(corpo: PredictRequest, request: Request) -> PredictResponse:
    """Valida o corpo (Pydantic), chama o service e devolve a previsão.

    O Pydantic já barrou dados inválidos com 422 antes desta função
    rodar; aqui só tratamos o que passou pelo contrato.
    """
    modelo = getattr(request.app.state, "modelo", None)
    if modelo is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Modelo não carregado na memória. "
                "Verifique models/mnist_mlp_pipeline.joblib."
            ),
        )

    try:
        resultado = services.prever(modelo, corpo.pixels)
    except services.ImagemInvalidaError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except services.ModeloNaoCarregadoError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc

    # Construir o schema de saída aqui garante que só sai o contrato:
    # se o service devolver algo fora dele, estoura 500 em vez de
    # entregar JSON quebrado ao cliente.
    return PredictResponse(**resultado)
