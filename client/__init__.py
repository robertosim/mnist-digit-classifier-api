"""Cliente desktop da MNIST Classifier API.

Responsabilidade: desenhar o dígito em um quadro negro, converter o
traço no formato aceito pela API (784 pixels, dígito branco sobre fundo
preto) e mostrar a previsão (dígito + confiança).

Módulos:

    cursores.py        ícones de caneta e borracha desenhados com Pillow
    preprocessamento.py quadro 560x560 -> 784 pixels no padrão MNIST
    api.py             cliente HTTP da API (somente biblioteca padrão)
    quadro.py          widget Tkinter do quadro de desenho
    interface.py       janela principal (ferramentas + painel de resultado)
    main.py            ponto de entrada

Como rodar (a partir da raiz do repositório):

    python -m client.main

O servidor precisa estar no ar (`uvicorn app.main:app`); o endereço
pode ser trocado na própria janela ou via `--servidor`.
"""
