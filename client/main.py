"""Ponto de entrada do cliente desktop.

Como rodar (a partir da raiz do repositório, com o servidor no ar):

    uvicorn app.main:app            # terminal 1
    python -m client.main           # terminal 2

Endereço do servidor:

    python -m client.main --servidor http://127.0.0.1:8000

O endereço também pode ser trocado na barra superior da janela.
"""
import argparse
import sys
from pathlib import Path

# Permite executar `python client/main.py` direto da raiz do repositório
# (mesmo artifício dos tests/): sem isto, `from client import ...` falharia
# quando o script roda fora de um pacote.
_RAIZ = Path(__file__).resolve().parents[1]
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

from client import api  # noqa: E402  (import depois do sys.path.insert)
from client.interface import Aplicacao  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m client.main",
        description="Cliente desktop da MNIST Classifier API (quadro de desenho + previsão).",
    )
    parser.add_argument(
        "--servidor",
        default=api.PADRAO,
        help=f"endereço base da API (padrão: {api.PADRAO})",
    )
    argumentos = parser.parse_args()

    aplicacao = Aplicacao(endereco=argumentos.servidor)
    aplicacao.mainloop()


if __name__ == "__main__":
    main()
