"""Pré-processamento compartilhado entre treino (notebook) e inferência (API).

Este módulo existe em um arquivo importável (e não no notebook) por um motivo
crítico de serialização:

O `joblib.dump` gravará no `.joblib` o *caminho* da função
(`app.preprocessing.normalizar_pixels`), e não o código-fonte dela.
Ao carregar o pipeline em outro processo (a API), o Python precisa conseguir
importar essa função a partir de um módulo. Se ela estivesse definida apenas
dentro do notebook, o import falharia com `AttributeError`/`ModuleNotFoundError`.

Por isso:
  1) A função mora em `app/preprocessing.py`
  2) Criamos `app/__init__.py` para transformar `app/` em um pacote importável
  3) O notebook e a API importam do MESMO módulo, garantindo que a normalização
     aplicada em produção é idêntica à usada no treino.
"""
from typing import Any

import numpy as np


def normalizar_pixels(X: Any) -> np.ndarray:
    """Converte pixels de 0-255 para 0.0-1.0 (float64).

    Recebe um array-like (np.ndarray, lista de listas etc.) e retorna
    um ndarray float64 com valores escalados por 1/255.

    Essa transformação é **fixa** (não aprende parâmetros dos dados),
    então pode ser aplicada dentro do Pipeline sem risco de leakage.
    """
    X = np.asarray(X)
    return X.astype("float64") / 255.0
