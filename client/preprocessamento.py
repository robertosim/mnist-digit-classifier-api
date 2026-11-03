"""Conversão do quadro desenhado no formato aceito pela API.

A API exige 784 inteiros (0-255) que representam um dígito MNIST:
fundo PRETO, dígito BRANCO, imagem de 28x28 com a figura centralizada
e ocupando cerca de 20x20 pixels (o resto é margem).

O usuário, porém, desenha o que quiser num quadro RGB maior, com
qualquer cor. Este módulo faz todo o tratamento necessário antes do
`POST /predict`:

    1. escala de cinza            (a cor do traço não muda a forma);
    2. normalização de brilho     (traço vermelho/escuro vira "branco");
    3. recorte ao redor do traço  (fundo preto do quadro é ignorado);
    4. redimensionamento          (cabe em 20x20 preservando proporção);
    5. centralização              (pelo CENTRO DE MASSA, regra do MNIST).

Por que o centro de massa (e não o centro do retângulo do traço): foi
assim que o MNIST foi normalizado no treino, e a diferença se mede na
API — com o centro do retângulo, um "7" fino saía errado (2, 61%); com
o centro de massa, o mesmo traço sai certo (7, 100%).

Sem tkinter de propósito: aqui só há Pillow, então os testes rodam em
qualquer ambiente (inclusive sem display gráfico).
"""
from PIL import Image

LADO = 28                # 28 x 28 = 784, o contrato da API
N_PIXELS = LADO * LADO
CAIXA_MNIST = 20         # dígitos MNIST cabem em 20x20 dentro do 28x28
CENTRO = LADO // 2       # (14, 14), o mesmo centro usado no treino
LIMIAR_CONTEUDO = 16     # abaixo disso é considerado vazio (0-255)


class QuadroVazioError(ValueError):
    """Não há traço desenhado (ou ele é fraco demais) no quadro."""


def _centro_de_massa(imagem: Image.Image) -> tuple[float, float]:
    """Centro ponderado pelo brilho de cada pixel, em coordenadas da imagem.

    É a "assinatura" de posição aprendida no treino: onde a massa do
    traço se concentra, não onde termina o retângulo que a contém.
    """
    largura, _ = imagem.size
    # No modo L, tobytes() devolve um byte por pixel (0-255) — API estável
    # em qualquer Pillow (getdata foi deprecado no 13 e some no Pillow 14).
    pixels = imagem.tobytes()
    soma = massa_x = massa_y = 0.0
    for indice, valor in enumerate(pixels):
        if valor:
            peso = float(valor)
            soma += peso
            massa_x += peso * (indice % largura)
            massa_y += peso * (indice // largura)
    if soma == 0:
        raise QuadroVazioError("O quadro está vazio. Desenhe um dígito antes de enviar.")
    return massa_x / soma, massa_y / soma


def preparar_imagem(imagem: Image.Image) -> Image.Image:
    """Aplica o tratamento completo e devolve a imagem MNIST 28x28.

    Args:
        imagem: quadro original (qualquer modo/size — RGB 560x560, etc).

    Returns:
        Image mode "L" de 28x28: fundo preto, dígito branco, centralizado
        pelo centro de massa.

    Raises:
        QuadroVazioError: se não houver conteúdo desenhado.
    """
    cinza = imagem.convert("L")

    brilho_maximo = cinza.getextrema()[1]
    if brilho_maximo <= LIMIAR_CONTEUDO:
        raise QuadroVazioError("O quadro está vazio. Desenhe um dígito antes de enviar.")

    # 2. O traço mais claro vira 255: desenhar de vermelho, azul ou
    #    cinza escuro produz a mesma "assinatura" de um dígito branco.
    if brilho_maximo < 255:
        fator = 255.0 / brilho_maximo
        cinza = cinza.point(lambda valor: min(255, int(valor * fator)))

    # 3. Recorta só onde há traço (mata as margens do quadro).
    caixa = cinza.point(lambda valor: 255 if valor > LIMIAR_CONTEUDO else 0).getbbox()
    if caixa is None:
        raise QuadroVazioError("O quadro está vazio. Desenhe um dígito antes de enviar.")
    recorte = cinza.crop(caixa)

    # 4. Cabe em 20x20 preservando a proporção (não distorce o dígito).
    largura, altura = recorte.size
    escala = min(CAIXA_MNIST / largura, CAIXA_MNIST / altura)
    novo_tamanho = (
        max(1, round(largura * escala)),
        max(1, round(altura * escala)),
    )
    recorte = recorte.resize(novo_tamanho, Image.LANCZOS)

    # 5. Centraliza pela massa do traço numa tela 28x28 preta — a folga
    #    restante é a mesma margem que existe no banco MNIST. Deslocamentos
    #    negativos (dígito "pesado" para um lado) são aparados pela
    #    própria imagem, como o normalizador original também fazia.
    centro_x, centro_y = _centro_de_massa(recorte)
    saida = Image.new("L", (LADO, LADO), 0)
    saida.paste(recorte, (round(CENTRO - centro_x), round(CENTRO - centro_y)))
    return saida


def extrair_pixels(imagem: Image.Image) -> list[int]:
    """Converte o quadro em 784 inteiros (0-255), linha a linha.

    É exatamente o corpo do `POST /predict`, já no formato do contrato
    (`schemas.PredictRequest`: 784 inteiros estritos, sem ponto decimal).

    Raises:
        QuadroVazioError: se não houver traço desenhado.
    """
    mnist = preparar_imagem(imagem)
    # tobytes() no modo L devolve um byte por pixel (0-255) e list()
    # transforma em inteiros de verdade, como o Pydantic exige (strict).
    pixels = list(mnist.tobytes())
    if len(pixels) != N_PIXELS:
        raise AssertionError(f"esperados {N_PIXELS} pixels, veio {len(pixels)}")
    return pixels
