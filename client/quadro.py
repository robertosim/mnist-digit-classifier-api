"""Quadro negro de desenho: caneta, borracha e cursor personalizado.

Como funciona

* O quadro é uma `tk.Canvas` preta de 560x560 que exibe uma imagem PIL
  (`self._imagem`, RGB). Esse único buffer é ao mesmo tempo o que o
  usuário VÊ e o que é ENVIADO para a API — assim o que sai da tela é
  sempre idêntico ao que o modelo recebe.
* O ponteiro do sistema some dentro do quadro (`cursor="none"`) e no
  lugar aparece a caneta (ou a borracha) gerada em `client.cursores`,
  com o ponto de contato colado no mouse. A ponta da caneta cai
  exatamente no pixel que está sendo pintado.
* Borracha não apaga "de verdade" o histórico: ela pinta de preto a
  área sob o cursor. Como o fundo do quadro É preto, o resultado visual
  é idêntico a apagar — e mantém um único buffer simples.
"""
import io
import tkinter as tk

from PIL import Image, ImageDraw, ImageTk

from client import cursores, preprocessamento

COR_APRAGAR = "#000000"
ESPESSURA_MIN = 4
ESPESSURA_MAX = 64
ESPESSURA_PADRAO = 24


class QuadroNegro(tk.Frame):
    """Área de desenho com ferramentas de caneta e borracha."""

    def __init__(self, mestre, largura: int = 560, altura: int = 560, **kwargs):
        super().__init__(mestre, **kwargs)

        # --- estado da ferramenta (a interface altera via definir_*) ---
        self.ferramenta = "caneta"
        self.cor = "#ffffff"
        self.espessura = ESPESSURA_PADRAO

        # --- buffer único: desenho e pré-processamento leem isto aqui ---
        self._imagem = Image.new("RGB", (largura, altura), "black")
        self._desenhando = False
        self._ultimo_ponto = None

        self._canvas = tk.Canvas(
            self,
            width=largura,
            height=altura,
            bg="black",
            bd=0,
            highlightthickness=1,
            highlightbackground="#4a4a4f",
            cursor="none",  # esconde o ponteiro; ver _posicionar_cursor
            takefocus=True,
        )
        self._canvas.pack(anchor="nw")

        self._foto_tabuleiro = ImageTk.PhotoImage(self._imagem)
        self._id_imagem = self._canvas.create_image(0, 0, anchor="nw", image=self._foto_tabuleiro)
        self._id_cursor = self._canvas.create_image(0, 0, anchor="center", state="hidden")
        self._foto_cursor = None  # mantém viva a referência do PhotoImage

        self._canvas.bind("<Enter>", self._ao_entrar)
        self._canvas.bind("<Leave>", self._ao_sair)
        self._canvas.bind("<Motion>", self._ao_mover)
        self._canvas.bind("<B1-Motion>", self._ao_mover)
        self._canvas.bind("<ButtonPress-1>", self._ao_pressionar)
        self._canvas.bind("<ButtonRelease-1>", self._ao_soltar)

        self.atualizar_cursor()

    # ------------------------------------------------------- ferramentas

    def definir_ferramenta(self, ferramenta: str) -> None:
        """Alterna entre "caneta" e "borracha" (muda o cursor na hora)."""
        if ferramenta not in ("caneta", "borracha"):
            raise ValueError(f"ferramenta desconhecida: {ferramenta}")
        self.ferramenta = ferramenta
        self.atualizar_cursor()

    def definir_cor(self, cor: str) -> None:
        """Define a cor do traço da caneta e regenera o cursor."""
        self.cor = cor
        if self.ferramenta == "caneta":
            self.atualizar_cursor()

    def definir_espessura(self, espessura: float) -> None:
        """Define a espessura (comum às duas ferramentas), em pixels."""
        self.espessura = max(ESPESSURA_MIN, min(ESPESSURA_MAX, int(espessura)))
        self.atualizar_cursor()

    def atualizar_cursor(self) -> None:
        """Regenera a imagem da caneta/borracha (PNG com transparência)."""
        if self.ferramenta == "borracha":
            imagem = cursores.cursor_borracha(self.espessura)
        else:
            imagem = cursores.cursor_caneta(self.cor)

        arquivo = io.BytesIO()
        imagem.save(arquivo, format="PNG")
        # tk.PhotoImage lê PNG com alfa (Tk 8.6) — sem ele, a ferramenta
        # taparia o desenho que está por baixo do cursor.
        self._foto_cursor = tk.PhotoImage(data=arquivo.getvalue())
        self._canvas.itemconfig(self._id_cursor, image=self._foto_cursor)

    # ----------------------------------------------------------- eventos

    def _ao_entrar(self, _evento) -> None:
        self._canvas.itemconfig(self._id_cursor, state="normal")

    def _ao_sair(self, _evento) -> None:
        # Saiu do quadro: esconde a ferramenta e corta o traço atual,
        # senão o retorno geraria uma linha reta ligando os dois cantos.
        self._canvas.itemconfig(self._id_cursor, state="hidden")
        self._encerrar_traco()

    def _ao_pressionar(self, evento) -> None:
        self._canvas.focus_set()
        self._desenhando = True
        self._ultimo_ponto = None
        self._pontilhar(evento.x, evento.y)

    def _ao_soltar(self, _evento) -> None:
        self._encerrar_traco()

    def _ao_mover(self, evento) -> None:
        self._posicionar_cursor(evento.x, evento.y)
        if self._desenhando:
            self._estender(evento.x, evento.y)

    def _encerrar_traco(self) -> None:
        self._desenhando = False
        self._ultimo_ponto = None

    def _posicionar_cursor(self, x: int, y: int) -> None:
        self._canvas.coords(self._id_cursor, x, y)

    # ----------------------------------------------------------- desenho

    def _cor_efetiva(self) -> str:
        return COR_APRAGAR if self.ferramenta == "borracha" else self.cor

    def _pontilhar(self, x: int, y: int) -> None:
        """Carimba um ponto (início do traço) e atualiza a tela."""
        self._carimbar(x, y, self._cor_efetiva())
        self._ultimo_ponto = (x, y)
        self._atualizar()

    def _estender(self, x: int, y: int) -> None:
        """Pinta o segmento entre o ponto anterior e o atual."""
        if self._ultimo_ponto is None:
            self._pontilhar(x, y)
            return

        x0, y0 = self._ultimo_ponto
        if (x0, y0) != (x, y):
            cor = self._cor_efetiva()
            desenho = ImageDraw.Draw(self._imagem)
            # joint="curve" dá articulação arredondada nas curvas.
            desenho.line([(x0, y0), (x, y)], fill=cor, width=self.espessura, joint="curve")
            # Pontas arredondadas: sem isto o traço terminaria "cortado".
            self._carimbar(x0, y0, cor)
            self._carimbar(x, y, cor)
            self._ultimo_ponto = (x, y)
            self._atualizar()

    def _carimbar(self, x: int, y: int, cor: str) -> None:
        """Círculo do tamanho da espessura, centrado em (x, y)."""
        raio = self.espessura / 2
        if raio < 1:
            ImageDraw.Draw(self._imagem).point((x, y), fill=cor)
            return
        ImageDraw.Draw(self._imagem).ellipse(
            [x - raio, y - raio, x + raio, y + raio],
            fill=cor,
        )

    def _atualizar(self) -> None:
        """Recompõe a imagem exibida a partir do buffer de desenho.

        Colagem no MESMO PhotoImage (5,2 ms) em vez de criar outro a cada
        evento de mouse (8,0 ms) — a diferença aparece em movimentos
        rápidos, onde cada `<Motion>` chega a cada ~8 ms.
        """
        self._foto_tabuleiro.paste(self._imagem)

    # ------------------------------------------------------ para a interface

    def limpar(self) -> None:
        """Apaga todo o quadro (volta ao preto)."""
        self._imagem.paste("black", (0, 0, self._imagem.width, self._imagem.height))
        self._encerrar_traco()
        self._atualizar()

    def extrair_pixels(self) -> list[int]:
        """Quadro -> 784 pixels no formato da API (ver preprocessamento).

        Raises:
            preprocessamento.QuadroVazioError: nada desenhado.
        """
        return preprocessamento.extrair_pixels(self._imagem)

    def miniatura(self) -> Image.Image:
        """Imagem 28x28 tratada, para mostrar ao usuário o que foi enviado."""
        return preprocessamento.preparar_imagem(self._imagem)
