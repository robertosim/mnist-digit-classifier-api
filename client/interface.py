"""Janela principal do cliente desktop.

Organização da tela

    ┌ barra do servidor: endereço + Conectar + estado da conexão ─────┐
    ├ ferramentas: [Caneta][Borracha]  Cor: [■■■■■]  [Limpar] ────────┤
    ├ espessura:   [────●────] 24 px  [ prévia ]   atalhos ───────────┤
    │                                                                │
    │   quadro negro 560x560                painel de resultado       │
    │   (caneta/borracha sob o mouse)        dígito · confiança       │
    │                                        imagem 28x28 enviada     │
    │                                        probabilidades 0..9      │
    └─────────────────────────────────────────────────────────────────┘

As chamadas à API rodam em uma thread separada com o resultado posto
numa `queue.Queue`, consumida por `after(50 ms)`. Assim a janela nunca
congela — nem mesmo quando o servidor está fora do ar e o `timeout`
demora a estourar. Nada de widget é tocado fora da thread do Tk.
"""
import queue
import threading
import tkinter as tk
from tkinter import colorchooser, ttk

from PIL import Image, ImageTk

from client import api, cursores, preprocessamento
from client.quadro import ESPESSURA_MAX, ESPESSURA_MIN, ESPESSURA_PADRAO, QuadroNegro

# ---------------------------------------------------------------- paleta
COR_FUNDO = "#1c1c1e"
COR_PAINEL = "#2c2c2e"
COR_CAMPO = "#3a3a3c"
COR_BORDA = "#48484a"
COR_TEXTO = "#f2f2f7"
COR_SECUNDARIO = "#8e8e93"
COR_DESTAQUE = "#0a84ff"
COR_ERRO = "#ff453a"
COR_OK = "#30d158"
COR_AVISO = "#ffd60a"

CORES_ATRITO = [
    "#ffffff",
    "#ff3b30",
    "#ff9500",
    "#ffcc00",
    "#34c759",
    "#00c7be",
    "#007aff",
    "#5856d6",
    "#af52de",
    "#ff2d55",
]

MINIATURA_PX = 60  # 28 x ~2 — mostrada só como prova do tratamento


class Aplicacao(tk.Tk):
    """Janela do cliente: desenho + envio para a API + previsão."""

    def __init__(self, endereco: str = api.PADRAO):
        super().__init__()
        self.title("MNIST · Cliente de Classificação de Dígitos")
        self.configure(bg=COR_FUNDO)
        self.geometry("910x800")
        self.minsize(880, 720)

        self.api = api.ClienteApi(endereco)
        self.var_endereco = tk.StringVar(value=self.api.endereco)
        self._fila: queue.Queue = queue.Queue()
        self._ocupado = False
        self._pendente: tuple | None = None  # próxima requisição, se houver
        self._fechando = False
        self._foto_miniatura = None
        self._icone_caneta = None
        self._botoes_cor: list[tuple[tk.Button, str]] = []
        self._rotulos_digito: list[tk.Label] = []

        self._estilo()
        self._barra_servidor(self)
        self._barra_ferramentas(self)
        self._corpo(self)

        self.bind("<KeyPress>", self._atalho)
        self.protocol("WM_DELETE_WINDOW", self._fechar)
        # Primeiro contato com o servidor assim que a janela abre.
        self.after(300, self._verificar_saude)

    # ------------------------------------------------------------ montagem

    def _estilo(self) -> None:
        """Tema ttk escuro, para os widgets nativos casarem com o quadro."""
        estilo = ttk.Style(self)
        try:
            estilo.theme_use("clam")
        except tk.TclError:
            pass
        estilo.configure(
            "TProgressbar",
            troughcolor=COR_CAMPO,
            background=COR_DESTAQUE,
            bordercolor=COR_CAMPO,
            lightcolor=COR_DESTAQUE,
            darkcolor=COR_DESTAQUE,
        )
        estilo.configure("TScale", background=COR_PAINEL, troughcolor=COR_CAMPO)
        estilo.configure(
            "TButton",
            background=COR_CAMPO,
            foreground=COR_TEXTO,
            focuscolor=COR_DESTAQUE,
            padding=(10, 6),
        )
        estilo.map(
            "TButton",
            background=[("active", COR_BORDA), ("disabled", COR_FUNDO)],
            foreground=[("disabled", COR_SECUNDARIO)],
        )

    @staticmethod
    def _rotulo(pai, texto: str, **kwargs) -> tk.Label:
        padrao = dict(bg=COR_PAINEL, fg=COR_TEXTO, font=("Segoe UI", 10))
        padrao.update(kwargs)
        return tk.Label(pai, text=texto, **padrao)

    @staticmethod
    def _separador(pai) -> None:
        ttk.Separator(pai, orient="vertical").pack(side="left", fill="y", padx=10, pady=4)

    def _barra_servidor(self, pai) -> None:
        barra = tk.Frame(pai, bg=COR_PAINEL, padx=10, pady=8)
        barra.pack(side="top", fill="x")

        self._rotulo(barra, "Servidor:", bg=COR_PAINEL).pack(side="left")
        tk.Entry(
            barra,
            textvariable=self.var_endereco,
            width=30,
            bg=COR_CAMPO,
            fg=COR_TEXTO,
            insertbackground=COR_TEXTO,
            relief="flat",
            bd=6,
            highlightthickness=1,
            highlightbackground=COR_BORDA,
        ).pack(side="left", padx=(8, 6))
        self.botao_conectar = ttk.Button(barra, text="Conectar", command=self._verificar_saude)
        self.botao_conectar.pack(side="left")

        self.rotulo_conexao = self._rotulo(
            barra, "● não verificado", bg=COR_PAINEL, fg=COR_SECUNDARIO, font=("Segoe UI", 10, "bold")
        )
        self.rotulo_conexao.pack(side="left", padx=12)

        self._rotulo(
            barra,
            "Enter prever  ·  C caneta  ·  B borracha  ·  Del limpar",
            bg=COR_PAINEL,
            fg=COR_SECUNDARIO,
            font=("Segoe UI", 9),
        ).pack(side="right")

    def _barra_ferramentas(self, pai) -> None:
        # --- linha A: ferramentas, cores e limpar ---
        linha_a = tk.Frame(pai, bg=COR_PAINEL, padx=10, pady=8)
        linha_a.pack(side="top", fill="x")

        self._rotulo(linha_a, "Ferramenta:", bg=COR_PAINEL).pack(side="left")

        self._icone_caneta = ImageTk.PhotoImage(cursores.icone("caneta", cor=self.cor_atual()))
        self._icone_borracha = ImageTk.PhotoImage(cursores.icone("borracha"))
        self.botao_caneta = self._botao_ferramenta(linha_a, "Caneta", self._icone_caneta, "caneta")
        self.botao_borracha = self._botao_ferramenta(linha_a, "Borracha", self._icone_borracha, "borracha")

        self._separador(linha_a)
        self._rotulo(linha_a, "Cor:", bg=COR_PAINEL).pack(side="left")
        for cor in CORES_ATRITO:
            botao = tk.Button(
                linha_a,
                bg=cor,
                activebackground=cor,
                activeforeground=cor,
                width=3,
                height=1,
                bd=0,
                relief="flat",
                highlightthickness=2,
                highlightbackground=COR_BORDA,
                takefocus=False,
                cursor="hand2",
                command=lambda c=cor: self._selecionar_cor(c),
            )
            botao.pack(side="left", padx=3)
            self._botoes_cor.append((botao, cor))

        tk.Button(
            linha_a,
            text="…",
            font=("Segoe UI", 11, "bold"),
            width=3,
            height=1,
            bg=COR_CAMPO,
            fg=COR_TEXTO,
            activebackground=COR_BORDA,
            activeforeground=COR_TEXTO,
            bd=0,
            highlightthickness=2,
            highlightbackground=COR_BORDA,
            takefocus=False,
            cursor="hand2",
            command=self._escolher_cor,
        ).pack(side="left", padx=(6, 0))

        self.botao_limpar = tk.Button(
            linha_a,
            text="Limpar quadro",
            font=("Segoe UI", 10),
            bg=COR_CAMPO,
            fg=COR_TEXTO,
            activebackground=COR_BORDA,
            activeforeground=COR_ERRO,
            relief="flat",
            bd=0,
            padx=10,
            pady=4,
            takefocus=False,
            cursor="hand2",
            command=self._limpar,
        )
        self.botao_limpar.pack(side="right")

        # --- linha B: espessura + prévia ---
        linha_b = tk.Frame(pai, bg=COR_PAINEL, padx=10, pady=8)
        linha_b.pack(side="top", fill="x")

        self._rotulo(linha_b, "Espessura:", bg=COR_PAINEL).pack(side="left")
        self.escala_espessura = ttk.Scale(
            linha_b,
            orient="horizontal",
            from_=ESPESSURA_MIN,
            to=ESPESSURA_MAX,
            length=180,
            command=self._ao_mudar_espessura,
        )
        self.escala_espessura.set(ESPESSURA_PADRAO)
        self.escala_espessura.pack(side="left", padx=(8, 6))

        self.rotulo_espessura = self._rotulo(
            linha_b, f"{ESPESSURA_PADRAO} px", bg=COR_PAINEL, fg=COR_SECUNDARIO, width=6, anchor="w"
        )
        self.rotulo_espessura.pack(side="left")

        self.canvas_preview = tk.Canvas(
            linha_b,
            width=76,
            height=44,
            bg="black",
            bd=0,
            highlightthickness=1,
            highlightbackground=COR_BORDA,
        )
        self.canvas_preview.pack(side="left", padx=(4, 0))

        self._rotulo(
            linha_b,
            "A borracha pinta de preto — o fundo do quadro é preto, então o efeito é apagar.",
            bg=COR_PAINEL,
            fg=COR_SECUNDARIO,
            font=("Segoe UI", 9),
        ).pack(side="right")

    def _botao_ferramenta(self, pai, texto: str, icone, ferramenta: str) -> tk.Button:
        botao = tk.Button(
            pai,
            text=texto,
            image=icone,
            compound="left",
            font=("Segoe UI", 10),
            bg=COR_CAMPO,
            fg=COR_TEXTO,
            activebackground=COR_BORDA,
            activeforeground=COR_TEXTO,
            relief="flat",
            bd=0,
            padx=8,
            pady=3,
            highlightthickness=2,
            highlightbackground=COR_BORDA,
            takefocus=False,
            cursor="hand2",
            command=lambda: self._selecionar_ferramenta(ferramenta),
        )
        botao.pack(side="left", padx=(0, 6))
        return botao

    def _corpo(self, pai) -> None:
        corpo = tk.Frame(pai, bg=COR_FUNDO, padx=10, pady=10)
        corpo.pack(side="top", fill="both", expand=True)

        self.quadro = QuadroNegro(corpo)
        self.quadro.pack(side="left", anchor="nw")
        self._selecionar_ferramenta("caneta")
        self._selecionar_cor("#ffffff")

        painel = tk.Frame(corpo, bg=COR_PAINEL, padx=14, pady=10)
        painel.pack(side="left", fill="y", padx=(14, 0))
        self._painel_resultado(painel)

    def _painel_resultado(self, painel) -> None:
        self._rotulo(painel, "Resultado", bg=COR_PAINEL, font=("Segoe UI", 13, "bold")).pack(anchor="w")

        self.rotulo_digito = self._rotulo(
            painel, "—", bg=COR_PAINEL, fg=COR_SECUNDARIO, font=("Consolas", 58, "bold")
        )
        self.rotulo_digito.pack(anchor="w", pady=(0, 2))

        self._rotulo(painel, "Confiança", bg=COR_PAINEL, fg=COR_SECUNDARIO).pack(anchor="w")
        barra_confianca = ttk.Progressbar(painel, maximum=100, length=240, mode="determinate")
        barra_confianca.pack(anchor="w", pady=(2, 0))
        self.barra_confianca = barra_confianca

        self.rotulo_confianca = self._rotulo(
            painel, "--", bg=COR_PAINEL, fg=COR_SECUNDARIO, font=("Segoe UI", 14, "bold")
        )
        self.rotulo_confianca.pack(anchor="w")

        self.botao_prever = tk.Button(
            painel,
            text="Prever dígito",
            font=("Segoe UI", 12, "bold"),
            bg=COR_DESTAQUE,
            fg="#ffffff",
            disabledforeground="#cfe0ff",  # desabilitado durante a requisição
            activebackground="#0a6fdc",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            padx=14,
            pady=8,
            cursor="hand2",
            command=self._prever,
        )
        self.botao_prever.pack(anchor="w", fill="x", pady=(12, 6))

        self.rotulo_mensagem = self._rotulo(
            painel,
            "Desenhe um dígito no quadro e clique em Prever.",
            bg=COR_PAINEL,
            fg=COR_SECUNDARIO,
            font=("Segoe UI", 9),
            wraplength=240,
            justify="left",
        )
        self.rotulo_mensagem.pack(anchor="w", pady=(0, 4))

        # O que a API realmente recebe (28x28, dígito branco sobre preto).
        self._rotulo(painel, "Imagem 28×28 enviada", bg=COR_PAINEL, fg=COR_SECUNDARIO,
                     font=("Segoe UI", 9)).pack(anchor="w")
        # O tamanho do tk.Label só acompanha a imagem se NÃO houver
        # width/height explícitos (com eles, viram pixels e cortam tudo).
        self._foto_miniatura = ImageTk.PhotoImage(
            Image.new("RGB", (MINIATURA_PX, MINIATURA_PX), "black")
        )
        self.rotulo_miniatura = tk.Label(
            painel,
            bg="black",
            bd=0,
            image=self._foto_miniatura,
            highlightbackground=COR_BORDA,
            highlightthickness=1,
        )
        self.rotulo_miniatura.pack(anchor="w", pady=(2, 4))

        self._rotulo(painel, "Probabilidade por dígito", bg=COR_PAINEL, fg=COR_SECUNDARIO,
                     font=("Segoe UI", 9)).pack(anchor="w")
        quadro_probs = tk.Frame(painel, bg=COR_PAINEL)
        quadro_probs.pack(anchor="w", pady=(2, 0))
        self.barras_prob = []
        self.rotulos_prob = []
        for digito in range(10):
            linha = tk.Frame(quadro_probs, bg=COR_PAINEL)
            linha.pack(fill="x")
            rotulo_digito = tk.Label(linha, text=str(digito), width=1, bg=COR_PAINEL,
                                     fg=COR_TEXTO, font=("Consolas", 10, "bold"))
            rotulo_digito.pack(side="left")
            barra = ttk.Progressbar(linha, maximum=100, length=150, mode="determinate")
            barra.pack(side="left", padx=6)
            rotulo_pct = tk.Label(linha, text="", width=6, anchor="e", bg=COR_PAINEL,
                                  fg=COR_SECUNDARIO, font=("Consolas", 9))
            rotulo_pct.pack(side="left")
            self._rotulos_digito.append(rotulo_digito)
            self.barras_prob.append(barra)
            self.rotulos_prob.append(rotulo_pct)

    # ---------------------------------------------------------- ferramentas

    def cor_atual(self) -> str:
        return getattr(self, "_cor_selecionada", "#ffffff")

    def _selecionar_ferramenta(self, ferramenta: str) -> None:
        self.quadro.definir_ferramenta(ferramenta)
        ativo, inativo = COR_DESTAQUE, COR_BORDA
        self.botao_caneta.config(highlightbackground=ativo if ferramenta == "caneta" else inativo)
        self.botao_borracha.config(highlightbackground=ativo if ferramenta == "borracha" else inativo)
        self._atualizar_preview()

    def _selecionar_cor(self, cor: str) -> None:
        self._cor_selecionada = cor
        self.quadro.definir_cor(cor)
        for botao, c in self._botoes_cor:
            botao.config(highlightbackground=COR_TEXTO if c == cor else COR_BORDA)
        # O ícone da caneta mostra a cor vigente.
        self._icone_caneta = ImageTk.PhotoImage(cursores.icone("caneta", cor=cor))
        self.botao_caneta.config(image=self._icone_caneta)
        self._atualizar_preview()

    def _escolher_cor(self) -> None:
        _, hexadecimal = colorchooser.askcolor(
            color=self.cor_atual(), title="Cor do traço", parent=self
        )
        if hexadecimal:
            self._selecionar_cor(hexadecimal)

    def _ao_mudar_espessura(self, valor: str) -> None:
        # O ttk.Scale dispara o comando durante a própria montagem da
        # janela (set() inicial), antes de o quadro e o rótulo existirem.
        if not hasattr(self, "quadro") or not hasattr(self, "rotulo_espessura"):
            return
        self.quadro.definir_espessura(float(valor))
        self.rotulo_espessura.config(text=f"{self.quadro.espessura} px")
        self._atualizar_preview()

    def _atualizar_preview(self) -> None:
        """Desenha na caixa preta um ponto do tamanho real da espessura."""
        if not hasattr(self, "canvas_preview") or not hasattr(self, "quadro"):
            return  # quadro e caixa de prévia ainda não existem (ordem de montagem)
        self.canvas_preview.delete("tudo")
        diametro = min(self.quadro.espessura, 40)
        cx, cy = 38, 22
        caixa = [cx - diametro / 2, cy - diametro / 2, cx + diametro / 2, cy + diametro / 2]
        if self.quadro.ferramenta == "borracha":
            # Borracha: só o contorno, porque o preto não aparece no preto.
            self.canvas_preview.create_oval(*caixa, outline="#ffffff", width=2)
        else:
            self.canvas_preview.create_oval(*caixa, fill=self.quadro.cor, outline="")

    def _limpar(self) -> None:
        self.quadro.limpar()
        self._limpar_resultado()
        self._mensagem("Quadro limpo.")

    # -------------------------------------------------------------- ações

    def _verificar_saude(self, *_evento) -> None:
        endereco = self.var_endereco.get().strip() or api.PADRAO
        if not endereco.startswith(("http://", "https://")):
            endereco = f"http://{endereco}"
            self.var_endereco.set(endereco)
        self.api.endereco = endereco.rstrip("/")
        self._mensagem(f"Verificando {self.api.endereco} …")
        self._assincrono(self.api.saude, self._ao_receber_saude)

    def _ao_receber_saude(self, valor, houve_erro: bool) -> None:
        if houve_erro:
            self.rotulo_conexao.config(text="● desconectado", fg=COR_ERRO)
            self._mensagem(f"Sem conexão: {valor}", erro=True)
            return
        if valor.get("model_loaded"):
            self.rotulo_conexao.config(text="● conectado · modelo na RAM", fg=COR_OK)
            self._mensagem("Conectado. Desenhe um dígito e clique em Prever.", ok=True)
        else:
            self.rotulo_conexao.config(text="● conectado · SEM modelo na RAM", fg=COR_AVISO)
            self._mensagem(
                "Servidor no ar, mas o modelo não carregou — /predict responderá 500.",
                erro=True,
            )

    def _prever(self, *_evento) -> None:
        try:
            pixels = self.quadro.extrair_pixels()
        except preprocessamento.QuadroVazioError as exc:
            self._mensagem(str(exc), erro=True)
            return

        # Mostra já o tratamento aplicado (recorte + 20x20 centralizado)
        # enquanto a requisição está em voo.
        self._mostrar_miniatura(self.quadro.miniatura())
        self._limpar_resultado()
        self._mensagem(f"Enviando {len(pixels)} pixels (28×28) …")
        self._assincrono(lambda: self.api.prever(pixels), self._ao_receber_previsao)

    def _ao_receber_previsao(self, valor, houve_erro: bool) -> None:
        if houve_erro:
            if isinstance(valor, api.ErroDaApi):
                texto = f"API respondeu erro: {valor.mensagem}"
            else:
                texto = f"Falha ao falar com a API: {valor}"
            self._mensagem(texto, erro=True)
            return

        digito = int(valor["digito"])
        confianca = float(valor["confianca"])
        self.rotulo_digito.config(text=str(digito), fg=COR_TEXTO)
        self.rotulo_confianca.config(text=f"{confianca * 100:.1f}%", fg=COR_TEXTO)
        self.barra_confianca.config(value=confianca * 100)

        probabilidades = valor.get("probabilidades") or []
        for i, barra in enumerate(self.barras_prob):
            prob = float(probabilidades[i]) if i < len(probabilidades) else 0.0
            barra.config(value=prob * 100)
            self.rotulos_prob[i].config(text=f"{prob * 100:.1f}%")
            self._rotulos_digito[i].config(fg=COR_DESTAQUE if i == digito else COR_TEXTO)

        self._mensagem(f"Previsão recebida: {digito} com {confianca * 100:.1f}% de confiança.", ok=True)

    def _limpar_resultado(self) -> None:
        self.rotulo_digito.config(text="—", fg=COR_SECUNDARIO)
        self.rotulo_confianca.config(text="--", fg=COR_SECUNDARIO)
        self.barra_confianca.config(value=0)
        for i, barra in enumerate(self.barras_prob):
            barra.config(value=0)
            self.rotulos_prob[i].config(text="")
            self._rotulos_digito[i].config(fg=COR_TEXTO)

    def _mostrar_miniatura(self, imagem: Image.Image) -> None:
        self._foto_miniatura = ImageTk.PhotoImage(
            imagem.resize((MINIATURA_PX, MINIATURA_PX), Image.NEAREST)
        )
        self.rotulo_miniatura.config(image=self._foto_miniatura)

    def _mensagem(self, texto: str, erro: bool = False, ok: bool = False) -> None:
        cor = COR_ERRO if erro else (COR_OK if ok else COR_SECUNDARIO)
        self.rotulo_mensagem.config(text=texto, fg=cor)

    # ------------------------------------------------------------ assíncrono

    def _assincrono(self, funcao, ao_concluir) -> None:
        """Roda `funcao` numa thread; o resultado volta pela fila.

        O Tk só pode ser tocado na thread principal, então a thread devolve
        o resultado numa Queue e o `_agendar` (via after) entrega à interface.

        Se já houver uma requisição em andamento, a nova entra numa fila
        de UMA vaga em vez de ser engolida — assim a checagem automática
        de conexão que dispara na abertura da janela não some quando o
        usuário clica Prever antes dela terminar.
        """
        if self._ocupado:
            self._pendente = (funcao, ao_concluir)
            self._mensagem("Requisição na fila — a resposta chega em seguida.")
            return
        self._ocupado = True
        self._botoes_de_acao(state="disabled")
        threading.Thread(target=self._rodar, args=(funcao, ao_concluir), daemon=True).start()
        self.after(50, self._agendar)

    def _rodar(self, funcao, ao_concluir) -> None:
        try:
            resultado = funcao()
        except Exception as exc:  # noqa: BLE001 - qualquer erro vira mensagem
            self._fila.put((ao_concluir, exc, True))
        else:
            self._fila.put((ao_concluir, resultado, False))

    def _agendar(self) -> None:
        if self._fechando:
            return
        try:
            ao_concluir, valor, houve_erro = self._fila.get_nowait()
        except queue.Empty:
            self.after(50, self._agendar)
            return

        self._ocupado = False
        ao_concluir(valor, houve_erro)

        if self._pendente is not None and not self._fechando:
            funcao, callback = self._pendente
            self._pendente = None
            self._assincrono(funcao, callback)
        else:
            self._botoes_de_acao(state="normal")

    def _botoes_de_acao(self, state: str) -> None:
        self.botao_prever.config(state=state)
        self.botao_conectar.config(state=state)

    # -------------------------------------------------------------- atalhos

    def _atalho(self, evento) -> None:
        """Atalhos de teclado — ignorados quando o foco está num campo de texto."""
        foco = self.focus_get()
        if isinstance(foco, (tk.Entry, tk.Text)):
            return
        tecla = evento.keysym.lower()
        if tecla == "c":
            self._selecionar_ferramenta("caneta")
        elif tecla == "b":
            self._selecionar_ferramenta("borracha")
        elif tecla in ("return", "kp_enter"):
            self._prever()
        elif tecla == "delete":
            self._limpar()

    def _fechar(self) -> None:
        self._fechando = True
        self.destroy()
