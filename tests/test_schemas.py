"""Testes da Fase 4: validação estrita dos schemas Pydantic.

Rodar (sem depender de pytest):

    python tests/test_schemas.py

Com pytest instalado, o mesmo arquivo roda com:

    pytest tests/test_schemas.py
"""
import sys
from pathlib import Path

# Permite executar o arquivo direto da raiz do repositório.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from pydantic import ValidationError

from app.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

PIXELS_VALIDOS = [0] * 784


def espera_erro(chamada):
    """Roda `chamada` e falha se ela NÃO levantar ValidationError."""
    try:
        chamada()
    except ValidationError as erro:
        return erro
    raise AssertionError("esperava ValidationError, mas a validação passou")


def primeiro_tipo(erro):
    """Retorna o tipo do primeiro erro de validação reportado."""
    return erro.errors()[0]["type"]


def test_payload_valido_e_aceito():
    requisicao = PredictRequest(pixels=PIXELS_VALIDOS)
    assert len(requisicao.pixels) == 784
    assert requisicao.pixels[0] == 0


def test_quantidade_errada_de_pixels():
    erro_curto = espera_erro(lambda: PredictRequest(pixels=[0] * 783))
    assert primeiro_tipo(erro_curto) == "too_short"
    erro_longo = espera_erro(lambda: PredictRequest(pixels=[0] * 785))
    assert primeiro_tipo(erro_longo) == "too_long"


def test_pixel_fora_da_faixa():
    erro_negativo = espera_erro(lambda: PredictRequest(pixels=[-1] + [0] * 783))
    assert primeiro_tipo(erro_negativo) == "greater_than_equal"
    erro_grande = espera_erro(lambda: PredictRequest(pixels=[256] + [0] * 783))
    assert primeiro_tipo(erro_grande) == "less_than_equal"


def test_texto_e_float_sao_rejeitados():
    # strict=True impede a conversão automática de str/float para int.
    erro_texto = espera_erro(lambda: PredictRequest(pixels=["abc"] + [0] * 783))
    assert primeiro_tipo(erro_texto) == "int_type"
    erro_texto_numerico = espera_erro(lambda: PredictRequest(pixels=["12"] + [0] * 783))
    assert primeiro_tipo(erro_texto_numerico) == "int_type"
    erro_float = espera_erro(lambda: PredictRequest(pixels=[12.5] + [0] * 783))
    assert primeiro_tipo(erro_float) == "int_type"
    # 255.0 também é rejeitado: é float, mesmo sendo valor válido.
    # Documentado no README — é a contrapartida de aceitar só int de verdade.
    erro_float_valido = espera_erro(lambda: PredictRequest(pixels=[255.0] + [0] * 783))
    assert primeiro_tipo(erro_float_valido) == "int_type"


def test_booleano_e_nulo_sao_rejeitados():
    # True não vale como 1 e None não vale como 0.
    erro_bool = espera_erro(lambda: PredictRequest(pixels=[True] + [0] * 783))
    assert primeiro_tipo(erro_bool) == "int_type"
    erro_nulo = espera_erro(lambda: PredictRequest(pixels=[None] + [0] * 783))
    assert primeiro_tipo(erro_nulo) == "int_type"


def test_estrutura_do_payload():
    # pixels precisa ser uma lista de 784 itens: string não serve e
    # campo ausente também não.
    erro_texto = espera_erro(lambda: PredictRequest(pixels="abc"))
    assert primeiro_tipo(erro_texto) == "list_type"
    erro_ausente = espera_erro(PredictRequest)
    assert primeiro_tipo(erro_ausente) == "missing"


def test_campo_extra_e_rejeitado():
    erro = espera_erro(lambda: PredictRequest(pixels=PIXELS_VALIDOS, extra=1))
    assert primeiro_tipo(erro) == "extra_forbidden"


def test_resposta_rejeita_digito_e_confianca_invalidos():
    # digito=10 ultrapassa o le=9; confianca=1.5 ultrapassa o le=1.
    erro_digito = espera_erro(
        lambda: PredictResponse(digito=10, confianca=0.9, probabilidades=[0.1] * 10)
    )
    assert primeiro_tipo(erro_digito) == "less_than_equal"
    assert erro_digito.errors()[0]["loc"] == ("digito",)
    erro_confianca = espera_erro(
        lambda: PredictResponse(digito=7, confianca=1.5, probabilidades=[0.1] * 10)
    )
    assert primeiro_tipo(erro_confianca) == "less_than_equal"
    assert erro_confianca.errors()[0]["loc"] == ("confianca",)


def test_resposta_rejeita_probabilidades_com_tamanho_errado():
    erro = espera_erro(
        lambda: PredictResponse(digito=7, confianca=0.5, probabilidades=[0.1] * 9)
    )
    assert primeiro_tipo(erro) == "too_short"


def test_resposta_aceita_escalares_numpy():
    # A Fase 5 devolve np.int64 (argmax) e np.float64 (predict_proba):
    # os schemas de saída ficaram sem strict de propósito para aceitá-los.
    resposta = PredictResponse(
        digito=np.int64(4),
        confianca=np.float64(0.97),
        probabilidades=np.full(10, 0.1),
    )
    assert resposta.digito == 4 and isinstance(resposta.digito, int)
    assert isinstance(resposta.confianca, float)


def test_resposta_e_saude_validas_sao_aceitas():
    resposta = PredictResponse(digito=7, confianca=0.5, probabilidades=[0.1] * 10)
    assert resposta.digito == 7
    saude = HealthResponse(status="ok", model_loaded=True)
    assert saude.status == "ok"
    erro = ErrorResponse(detail="imagem inválida")
    assert erro.detail == "imagem inválida"


def _main():
    testes = sorted(
        (nome, funcao)
        for nome, funcao in globals().items()
        if nome.startswith("test_") and callable(funcao)
    )
    falhas = 0
    for nome, funcao in testes:
        try:
            funcao()
        except Exception as exc:  # noqa: BLE001 - relata e continua
            falhas += 1
            print(f"FALHOU {nome}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok     {nome}")
    print(f"\n{len(testes) - falhas}/{len(testes)} testes passaram")
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(_main())
