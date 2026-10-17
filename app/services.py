"""Camada de serviço: carregamento do modelo e inferência.

Responsabilidade: regras de negócio puras — carregar o .joblib e executar
a previsão matemática. NÃO conhece HTTP (nada de Request, Response,
HTTPException ou status codes). Se amanhã trocar FastAPI por CLI ou
notebook, este módulo continua igual.
"""
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "mnist_mlp_pipeline.joblib"


class ImagemInvalidaError(ValueError):
    """A imagem recebida não tem formato válido (esperado 784 pixels)."""


class ModeloNaoCarregadoError(RuntimeError):
    """O arquivo de modelo não existe ou não pôde ser carregado."""


def carregar_modelo():
    """Carrega o pipeline joblib do disco e retorna o modelo treinado.

    Raises:
        ModeloNaoCarregadoError: se MODEL_PATH não existir.
    """
    raise NotImplementedError("Fase 3: implementar carregamento do modelo.")


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
