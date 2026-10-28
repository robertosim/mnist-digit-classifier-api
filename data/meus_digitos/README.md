# Meus dígitos (Desafio C)

Imagens usadas no teste manual da API (`POST /predict`), organizadas na pasta
de dados exigida pela AVA (caminhos relativos, nada fora do repositório).

## Padrão dos arquivos

| Item | Valor |
|---|---|
| Nomes | `digito_0.png` ... `digito_9.png` |
| Tamanho | 28 × 28 pixels |
| Modo | escala de cinza (L) |
| Cores | fundo preto (`0`) e traço branco (`255`), como o MNIST |

## ⚠️ Arquivos provisórios

`digito_0.png` a `digito_6.png`, `digito_8.png` e `digito_9.png` foram
**gerados por código (PIL)** apenas para deixar a pasta completa e permitir
testar o pipeline. A AVA pede imagens digitalizadas **próprias** no Desafio C —
substitua pelos seus desenhos antes de entregar.

`digito_7.png` já existia na pasta e não foi alterado.

## Como substituir pelos seus desenhos

1. Desenhe o dígito (papel branco, traço grosso e centralizado).
2. Fotografe/recorte e reduza para **28 × 28**, escala de cinza.
3. Inverta se necessário: fundo preto, traço branco.
4. Salve com o mesmo nome (`digito_N.png`), sobrescrevendo o provisório.
