"""Cursores e ícones de caneta/borracha gerados com Pillow.

Nenhum arquivo de imagem externo: os pixels são desenhados em tempo de
execução com `ImageDraw`. Dois formatos são produzidos:

    cursor_*()  -> quadrado LADO x LADO (64 px) com o ponto de CONTATO
                   exatamente no centro. É esse centro que o quadro leva
                   para o ponteiro do mouse, então a ponta da caneta (ou
                   o centro da borracha) cai no pixel que será pintado.

    icone_*()   -> recorte reduzido do cursor, para os botões de ferramenta.

O fundo é transparente (RGBA), para que a ferramenta não esconda o que
já foi desenhado no quadro.
"""
from PIL import Image, ImageDraw

LADO = 64
CENTRO = LADO // 2

# Corpo das ferramentas inclinado a 45 graus, como uma caneta na mão.
_DESL = 0.70710678  # cos(45) = sen(45)
_D = (_DESL, -_DESL)   # sentido "para cima e à direita"
_P = (_DESL, _DESL)    # perpendicular (direção da largura)

BORDA = "#f5f5f5"
COR_NIB = "#e8e8ec"
COR_ARO = "#c9c9ce"
COR_CORPO_BORRACHA = "#e9e9ef"
COR_LATAO = "#ff6b8a"


def _pt(ao_largo: float, ao_lado: float) -> tuple[float, float]:
    """Ponto no sistema da ferramenta: `ao_largo` ao longo do corpo,
    `ao_lado` ao longo da largura. O (0, 0) é o ponto de contato."""
    return (
        CENTRO + _D[0] * ao_largo + _P[0] * ao_lado,
        CENTRO + _D[1] * ao_largo + _P[1] * ao_lado,
    )


def cursor_caneta(cor: str) -> Image.Image:
    """Cursor na forma de caneta, com a ponta no centro da imagem.

    Args:
        cor: cor do traço escolhida pelo usuário (ex.: "#ff3b30"). O
            corpo da caneta usa essa mesma cor, deixando visível o que
            será pintado.

    Returns:
        Image RGBA 64x64; o pixel (32, 32) é a ponta da caneta.
    """
    img = Image.new("RGBA", (LADO, LADO), (0, 0, 0, 0))
    desenho = ImageDraw.Draw(img)

    # Lâmina metálica (bico): triângulo que afina até o ponto de contato.
    bico = [_pt(0, 0), _pt(7, 3), _pt(7, -3)]
    # Corpo: retângulo inclinado, da lâmina até a ponta oposta.
    corpo = [_pt(7, 5), _pt(38, 5), _pt(38, -5), _pt(7, -5)]
    # Aro metálico entre o corpo e a lâmina.
    aro = [_pt(7, 5.6), _pt(14, 5.6), _pt(14, -5.6), _pt(7, -5.6)]

    desenho.polygon(corpo, fill=cor, outline=BORDA)
    desenho.polygon(aro, fill=COR_ARO, outline=BORDA)
    desenho.polygon(bico, fill=COR_NIB, outline=BORDA)
    return img


def cursor_borracha(espessura: int) -> Image.Image:
    """Cursor na forma de borracha, com o centro da área de apagagem
    no centro da imagem.

    O corpo tem largura mínima visual (para a ferramenta continuar
    reconhecível com traços finos), mas o círculo desenhado por cima
    mostra a área que REALMENTE será apagada — diâmetro igual à
    espessura escolhida.

    Args:
        espessura: espessura atual da ferramenta, em pixels do quadro.

    Returns:
        Image RGBA 64x64; o pixel (32, 32) é o centro da área de apagagem.
    """
    raio = max(espessura, 1) / 2
    meia_largura = max(raio, 8)   # largura visual mínima do corpo
    comprimento = 22

    img = Image.new("RGBA", (LADO, LADO), (0, 0, 0, 0))
    desenho = ImageDraw.Draw(img)

    corpo = [
        _pt(0, meia_largura),
        _pt(comprimento, meia_largura),
        _pt(comprimento, -meia_largura),
        _pt(0, -meia_largura),
    ]
    # Borracha: a faixa em contato com o quadro fica na base do corpo.
    latao = [
        _pt(0, meia_largura),
        _pt(8, meia_largura),
        _pt(8, -meia_largura),
        _pt(0, -meia_largura),
    ]

    desenho.polygon(corpo, fill=COR_CORPO_BORRACHA, outline=BORDA)
    desenho.polygon(latao, fill=COR_LATAO, outline=BORDA)

    # Área real de apagagem (por cima, para ficar visível).
    desenho.ellipse(
        [CENTRO - raio, CENTRO - raio, CENTRO + raio, CENTRO + raio],
        outline=BORDA,
        width=2,
    )
    return img


def icone(ferramenta: str, cor: str = "#ffffff", espessura: int = 20, tamanho: int = 28) -> Image.Image:
    """Ícone de barra de ferramentas: o cursor reduzido ao seu conteúdo.

    Args:
        ferramenta: "caneta" ou "borracha".
        cor: cor do corpo da caneta (ignorada na borracha).
        espessura: usada só para desenhar o corpo da borracha no ícone.
        tamanho: lado do quadrado final, em pixels.

    Returns:
        Image RGBA `tamanho` x `tamanho`.
    """
    if ferramenta == "borracha":
        cursor = cursor_borracha(espessura)
    else:
        cursor = cursor_caneta(cor)

    # bbox do canal alfa: só o que foi realmente desenhado.
    caixa = cursor.getchannel("A").getbbox() or (0, 0, LADO, LADO)
    recorte = cursor.crop(caixa)
    return recorte.resize((tamanho, tamanho), Image.LANCZOS)
