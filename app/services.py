"""Camada de serviço: carregamento do modelo e inferência.

Responsabilidade: regras de negócio puras — carregar o .joblib e executar
a previsão matemática. NÃO conhece HTTP (nada de Request, Response,
HTTPException ou status codes). Se amanhã trocar FastAPI por CLI ou
notebook, este módulo continua igual.
"""
import logging
import time
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "mnist_mlp_pipeline.joblib"


class ImagemInvalidaError(ValueError):
    """A imagem recebida não tem formato válido (esperado 784 pixels)."""


class ModeloNaoCarregadoError(RuntimeError):
    """O arquivo de modelo não existe ou não pôde ser carregado."""


def carregar_modelo():
    """Carrega o pipeline joblib do disco e retorna o modelo treinado.

    Valida que o arquivo existe, carrega com joblib, confirma que o
    objeto tem predict_proba e registra no log o caminho e o tempo.

    Returns:
        Objeto Pipeline treinado, pronto para predict_proba.

    Raises:
        ModeloNaoCarregadoError: se o arquivo não existir ou o objeto
            não tiver predict_proba.
    """
    if not MODEL_PATH.exists():
        raise ModeloNaoCarregadoError(
            f"Arquivo de modelo não encontrado: {MODEL_PATH}. "
            "Execute o notebook da Fase 1 para gerar o .joblib."
        )

    inicio = time.perf_counter()
    modelo = joblib.load(MODEL_PATH)
    duracao = time.perf_counter() - inicio

    if not hasattr(modelo, "predict_proba"):
        raise ModeloNaoCarregadoError(
            f"Objeto carregado de {MODEL_PATH} não tem predict_proba. "
            "Verifique se o .joblib contém um Pipeline sklearn válido."
        )

    logger.info(
        "Modelo carregado de %s em %.3fs (tipo: %s)",
        MODEL_PATH,
        duracao,
        type(modelo).__name__,
    )
    return modelo


def prever(modelo, imagem):
    """Executa a inferência e retorna dígito previsto + probabilidades.

    Args:
        modelo: objeto carregado por carregar_modelo().
        imagem: array-like com 784 pixels (0-255).

    Returns:
        dict com chaves "digito", "confianca" e "probabilidades".

    Raises:
        ImagemInvalidaError: se a imagem não tiver 784 pixels.
    """
    raise NotImplementedError("Fase 5: implementar inferência.")
