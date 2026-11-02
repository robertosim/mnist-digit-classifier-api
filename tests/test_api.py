"""Testes da Fase 5: endpoints e tratamento de erros HTTP.

Rodar (sem depender de pytest):

    python tests/test_api.py

Com pytest instalado, o mesmo arquivo roda com:

    pytest tests/test_api.py

Cada teste monta o próprio `with TestClient(app)` porque o bloco `with`
é o que dispara o lifespan (carregamento do modelo na RAM).
"""
import sys
from pathlib import Path

# Permite executar o arquivo direto da raiz do repositório.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging

from fastapi.testclient import TestClient

from app import services
from app.main import app

# Deixa a saída dos testes limpa: sem as linhas INFO do lifespan,
# do httpx2 e sem o traceback do 500 proposital (a resposta JSON
# provada pelo próprio teste já mostra que o handler rodou).
logging.getLogger("app").setLevel(logging.CRITICAL)
logging.getLogger("httpx2").setLevel(logging.WARNING)

PIXELS_VALIDOS = [0] * 784


def pixels_exemplo():
    """Traço vertical (parece um '1') — mesmo exemplo do Swagger."""
    pixels = list(PIXELS_VALIDOS)
    for linha in range(4, 24):
        pixels[linha * 28 + 14] = 255
    return pixels


# ---------------------------------------------------------------- health


def test_health_retorna_200_com_modelo_na_ram():
    with TestClient(app) as cliente:  # `with` roda o lifespan
        resposta = cliente.get("/health")
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo == {"status": "ok", "model_loaded": True}


def test_health_continua_200_sem_modelo():
    # Saúde informa o estado, não pune: continua 200, mas avisa.
    with TestClient(app) as cliente:
        app.state.modelo = None
        resposta = cliente.get("/health")
    assert resposta.status_code == 200
    assert resposta.json() == {"status": "indisponivel", "model_loaded": False}


# --------------------------------------------------------------- predict


def test_predict_valido_retorna_200_com_contrato_de_saida():
    with TestClient(app) as cliente:
        resposta = cliente.post("/predict", json={"pixels": pixels_exemplo()})

    assert resposta.status_code == 200
    corpo = resposta.json()
    assert set(corpo) == {"digito", "confianca", "probabilidades"}
    assert 0 <= corpo["digito"] <= 9
    assert 0.0 <= corpo["confianca"] <= 1.0
    assert len(corpo["probabilidades"]) == 10
    # confiança = probabilidade da classe vencedora; soma das probs ~ 1.
    assert corpo["confianca"] == max(corpo["probabilidades"])
    assert corpo["digito"] == corpo["probabilidades"].index(max(corpo["probabilidades"]))
    assert abs(sum(corpo["probabilidades"]) - 1.0) < 1e-5


def test_predict_pixels_invalidos_gera_422_na_rota_real():
    # Pendência da Fase 4 cumprida: o 422 provado na rota de produção.
    with TestClient(app) as cliente:
        resposta = cliente.post("/predict", json={"pixels": [0] * 783})
    assert resposta.status_code == 422
    detalhe = resposta.json()["detail"][0]
    assert detalhe["loc"] == ["body", "pixels"]
    assert detalhe["type"] == "too_short"


def test_predict_tipos_errados_geram_422():
    casos = {
        "texto": {"pixels": ["abc"] + [0] * 783},
        "float": {"pixels": [12.5] + [0] * 783},
        "booleano": {"pixels": [True] + [0] * 783},
        "campo ausente": {},
        "campo extra": {"pixels": list(PIXELS_VALIDOS), "digito": 7},
    }
    with TestClient(app) as cliente:
        for nome, corpo in casos.items():
            resposta = cliente.post("/predict", json=corpo)
            assert resposta.status_code == 422, f"{nome}: esperado 422"


def test_predict_com_modelo_fora_da_ram_gera_500():
    with TestClient(app) as cliente:
        app.state.modelo = None  # simula falha de carregamento no lifespan
        resposta = cliente.post("/predict", json={"pixels": list(PIXELS_VALIDOS)})
    assert resposta.status_code == 500
    assert "Modelo não carregado" in resposta.json()["detail"]


def test_imagem_invalida_na_camada_de_servico_gera_400():
    # O Pydantic não deixa passar imagem errada pela rota, então provamos
    # o mapeamento 400 injectando a falha exatamente onde ela pode ocorrer:
    # dentro de services.prever (o mesmo caminho de um CLI/notebook).
    original = services.prever

    def _imagem_invalida(modelo, imagem):
        raise services.ImagemInvalidaError("Esperados 784 pixels; recebidos 783.")

    services.prever = _imagem_invalida
    try:
        with TestClient(app) as cliente:
            resposta = cliente.post("/predict", json={"pixels": list(PIXELS_VALIDOS)})
    finally:
        services.prever = original

    assert resposta.status_code == 400
    assert resposta.json()["detail"] == "Esperados 784 pixels; recebidos 783."


def test_erro_inesperado_gera_500_padronizado_sem_derrubar_a_api():
    # raise_server_exceptions=False: o ServerErrorMiddleware chama o
    # handler global de Exception registrado no main.py e devolve o JSON.
    original = services.prever

    def _estouro(modelo, imagem):
        raise RuntimeError("falha proposital")

    services.prever = _estouro
    try:
        with TestClient(app, raise_server_exceptions=False) as cliente:
            resposta = cliente.post("/predict", json={"pixels": list(PIXELS_VALIDOS)})
    finally:
        services.prever = original

    assert resposta.status_code == 500
    assert resposta.json() == {"detail": "Erro interno do servidor. Tente novamente."}


# ------------------------------------------------------------ serviço puro


def test_servico_rejeita_imagem_de_tamanho_errado():
    # services.prever não depende de HTTP: defende-se sozinho.
    try:
        services.prever(object(), [0] * 783)
    except services.ImagemInvalidaError:
        pass
    else:
        raise AssertionError("esperava ImagemInvalidaError para 783 pixels")


def test_servico_rejeita_modelo_none():
    try:
        services.prever(None, list(PIXELS_VALIDOS))
    except services.ModeloNaoCarregadoError:
        pass
    else:
        raise AssertionError("esperava ModeloNaoCarregadoError com modelo None")


def test_rota_inexistente_gera_404():
    with TestClient(app) as cliente:
        resposta = cliente.get("/rota-que-nao-existe")
    assert resposta.status_code == 404


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
