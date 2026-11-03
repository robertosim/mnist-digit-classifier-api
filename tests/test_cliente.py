"""Testes do cliente: pré-processamento, contrato da API e erros de rede.

Rodar (sem depender de pytest):

    python tests/test_cliente.py

Com pytest instalado, o mesmo arquivo roda com:

    pytest tests/test_cliente.py

Cobrem o caminho crítico do desktop: quadro -> 784 pixels -> corpo do
POST /predict. Importam só Pillow/Pydantic (nada de tkinter), então rodam
em ambiente sem display gráfico.
"""
import sys
from pathlib import Path

# Permite executar o arquivo direto da raiz do repositório.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw

from client import api, cursores, preprocessamento
from client.preprocessamento import QuadroVazioError, extrair_pixels, preparar_imagem

# O contrato da API é a fonte da verdade: o payload do cliente precisa
# ser aceito pelo MESMO schema que valida o POST /predict.
from app.schemas import PredictRequest


def quadro_vazio() -> Image.Image:
    return Image.new("RGB", (560, 560), "black")


def quadro_com_traco(cor: str = "#ffffff", caixa=(180, 120, 380, 440)) -> Image.Image:
    """Quadro com um retângulo preenchido (um 'bloco' de dígito)."""
    imagem = quadro_vazio()
    ImageDraw.Draw(imagem).rectangle(caixa, fill=cor, width=20)
    return imagem


def nao_zeros(pixels):
    """Coordenadas (linha, coluna) dos pixels com traço."""
    return [
        (indice // 28, indice % 28)
        for indice, valor in enumerate(pixels)
        if valor > 0
    ]


# ------------------------------------------------- pré-processamento


def test_quadro_vazio_gera_erro_claro():
    try:
        extrair_pixels(quadro_vazio())
    except QuadroVazioError:
        pass
    else:
        raise AssertionError("esperava QuadroVazioError para quadro preto")


def test_extrair_pixels_devolve_784_inteiros_0_a_255():
    pixels = extrair_pixels(quadro_com_traco())
    assert len(pixels) == 784
    assert all(isinstance(valor, int) and not isinstance(valor, bool) for valor in pixels)
    assert min(pixels) >= 0 and max(pixels) <= 255
    assert max(pixels) == 255  # traço branco normalizado para o topo


def test_tratamento_normaliza_qualquer_cor_para_branco_sobre_preto():
    # Vermelho puro tem luminância 76/255; sem a normalização o modelo
    # receberia um traço apagado em vez de um dígito branco.
    pixels = extrair_pixels(quadro_com_traco(cor="#ff0000"))
    assert max(pixels) == 255
    # O fundo continua preto (0) — contrato MNIST: só o bloco do traço
    # (13x20 no centro) fica claro, o resto da tela 28x28 é preto.
    assert pixels.count(0) >= 500


def test_tratamento_centraliza_desenho_feito_num_canto():
    # Traço jogado no canto superior esquerdo: depois do recorte ele
    # precisa ser centralizado na tela 28x28, com a margem do MNIST.
    pixels = extrair_pixels(quadro_com_traco(caixa=(10, 10, 120, 200)))
    linhas = [linha for linha, _ in nao_zeros(pixels)]
    colunas = [coluna for _, coluna in nao_zeros(pixels)]
    assert min(linhas) >= 3 and max(linhas) <= 25
    assert min(colunas) >= 3 and max(colunas) <= 25


def test_tratamento_preserva_a_proporcao_do_desenho():
    # Retângulo 5x1 (bem mais largo que alto) não pode virar quadrado.
    imagem = quadro_vazio()
    ImageDraw.Draw(imagem).rectangle((100, 240, 400, 300), fill="#ffffff")
    conteudo = nao_zeros(extrair_pixels(imagem))
    linhas = [linha for linha, _ in conteudo]
    colunas = [coluna for _, coluna in conteudo]
    altura, largura = max(linhas) - min(linhas) + 1, max(colunas) - min(colunas) + 1
    assert largura == 20  # é a dimensão maior que estoura na caixa 20x20
    assert 3 <= altura <= 7  # 60/300 da largura


def test_centralizacao_usa_o_centro_de_massa_e_nao_o_do_retangulo():
    # Triângulo: a massa fica longe do centro do retângulo que o contém.
    # É a regra do MNIST (com centróide o "7" fino previa errado; com o
    # centro do retângulo a mesma imagem saía como 2 com 61% de confiança).
    imagem = quadro_vazio()
    ImageDraw.Draw(imagem).polygon([(200, 100), (400, 100), (400, 400)], fill="#ffffff")

    miniatura = preparar_imagem(imagem)
    centro_x, centro_y = preprocessamento._centro_de_massa(miniatura)
    assert abs(centro_x - 14) <= 1.0
    assert abs(centro_y - 14) <= 1.0

    linhas = [linha for linha, _ in nao_zeros(list(miniatura.tobytes()))]
    colunas = [coluna for _, coluna in nao_zeros(list(miniatura.tobytes()))]
    centro_do_retangulo = (
        (min(colunas) + max(colunas)) / 2,
        (min(linhas) + max(linhas)) / 2,
    )
    # Prova que NÃO é o centro geométrico: os dois pontos divergem.
    assert abs(centro_do_retangulo[0] - centro_x) > 1.0


def test_massa_puxada_para_a_borda_nao_estoura_a_imagem():
    # Bloco pesado à direita + risco fino à esquerda: o deslocamento do
    # colo negativo precisa ser aparado pela imagem, sem erro e sem
    # perder a regra de "massa no centro" (dentro da tolerância).
    imagem = quadro_vazio()
    desenho = ImageDraw.Draw(imagem)
    desenho.rectangle((300, 150, 400, 250), fill="#ffffff")
    desenho.line([(100, 198), (300, 198)], fill="#ffffff", width=4)

    miniatura = preparar_imagem(imagem)
    assert miniatura.size == (28, 28)
    assert sum(1 for valor in miniatura.tobytes() if valor) > 0
    centro_x, centro_y = preprocessamento._centro_de_massa(miniatura)
    assert abs(centro_x - 14) <= 1.5


def test_miniatura_e_uma_imagem_28x28_em_cinza():
    miniatura = preparar_imagem(quadro_com_traco())
    assert miniatura.size == (28, 28)
    assert miniatura.mode == "L"


def test_payload_do_cliente_e_aceito_pelo_schema_da_api():
    # Prova de ponta a ponta do formato: o que o cliente monta passa
    # pelo MESMO Pydantic que barra o POST /predict com 422.
    pixels = extrair_pixels(quadro_com_traco())
    contrato = PredictRequest(pixels=pixels)
    assert contrato.pixels == pixels
    assert len(contrato.pixels) == 784


# ---------------------------------------------------------------- API


def test_api_recusa_payload_de_tamanho_errado_sem_rede():
    try:
        api.ClienteApi("http://127.0.0.1:1").prever([0] * 783)
    except ValueError as exc:
        assert "784" in str(exc)
    else:
        raise AssertionError("esperava ValueError para 783 pixels")


def test_api_sem_servidor_vira_erro_de_conexao():
    # Porta 1 em localhost: conexão recusada na hora, sem depender de rede.
    try:
        api.ClienteApi("http://127.0.0.1:1", timeout=5).saude()
    except api.ErroDeConexao as exc:
        assert "127.0.0.1" in str(exc)
    else:
        raise AssertionError("esperava ErroDeConexao com o servidor fora")


def test_erro_422_da_api_vira_frase_legivel():
    corpo_422 = (
        '{"detail": [{"type": "too_short", "loc": ["body", "pixels"],'
        ' "msg": "List should have at least 784 items"}]}'
    )
    texto = api._detalhe_legivel(corpo_422)
    assert "body.pixels" in texto
    assert "784" in texto
    # Mesmo formato, mas com a lista direto no lugar do "detail" (proxy).
    assert "body.pixels" in api._detalhe_legivel(corpo_422.replace('{"detail": ', "").rstrip("}"))


def test_erro_de_texto_cru_tambem_vira_frase():
    assert api._detalhe_legivel('{"detail": "Modelo não carregado."}') == "Modelo não carregado."


# ------------------------------------------------------------- cursores


def test_cursor_tem_64px_e_fundo_transparente():
    for imagem in (cursores.cursor_caneta("#ffffff"), cursores.cursor_borracha(20)):
        assert imagem.size == (64, 64)
        assert imagem.mode == "RGBA"
        assert imagem.getpixel((0, 0))[3] == 0  # canto vazio
        assert imagem.getchannel("A").getbbox()  # tem desenho


def test_ponta_da_caneta_fica_no_ponto_de_contato():
    # O centro (32, 32) é onde o quadro posiciona o cursor; a lâmina
    # precisa existir bem ao lado dele (a ponta afina até esse ponto).
    imagem = cursores.cursor_caneta("#ff3b30")
    vizinhanca = [
        imagem.getpixel((x, y))[3]
        for x in range(31, 36)
        for y in range(31, 36)
    ]
    assert max(vizinhanca) > 0


def test_icone_de_ferramenta_e_quadrado_no_tamanho_pedido():
    assert cursores.icone("caneta", tamanho=28).size == (28, 28)
    assert cursores.icone("borracha", tamanho=32).size == (32, 32)


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
