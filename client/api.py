"""Cliente HTTP da API MNIST — só biblioteca padrão do Python.

Usa `urllib` (não requests/httpx) de propósito: o desktop não deve
exigir dependência além do Pillow, que já está no requirements.txt.

Traduz as falhas de rede e os status da API em exceções legíveis, que a
interface mostra no painel de mensagens:

    ErroDeConexao  servidor fora do ar / endereço errado / timeout
    ErroDaApi      a API respondeu com erro (400, 422, 500, ...)
"""
import json
import urllib.error
import urllib.request

PADRAO = "http://127.0.0.1:8000"
N_PIXELS = 784
TIMEOUT_PADRAO = 15.0  # segundos


class ErroDeConexao(RuntimeError):
    """Não foi possível falar com o servidor."""


class ErroDaApi(RuntimeError):
    """O servidor respondeu, mas com erro (status >= 400)."""

    def __init__(self, status: int, mensagem: str):
        super().__init__(f"HTTP {status}: {mensagem}")
        self.status = status
        self.mensagem = mensagem


def _detalhe_legivel(texto: str) -> str:
    """Converte o corpo de erro da API em uma frase legível.

    O FastAPI devolve {"detail": "..."}; no 422 (contrato do Pydantic)
    `detail` é uma LISTA de erros, então aqui ela é resumida — jogar a
    lista crua na tela só polui o painel.
    """
    try:
        corpo = json.loads(texto)
    except (json.JSONDecodeError, TypeError):
        return (texto or "").strip()[:200] or "resposta sem corpo"

    # FastAPI devolve {"detail": ...}; alguns proxies passam a lista do
    # 422 direto, então os dois formatos são tratados.
    if isinstance(corpo, list):
        detalhe = corpo
    elif isinstance(corpo, dict):
        detalhe = corpo.get("detail")
    else:
        detalhe = None
    if isinstance(detalhe, list):
        partes = []
        for item in detalhe:
            if not isinstance(item, dict):
                partes.append(str(item))
                continue
            campo = ".".join(str(parte) for parte in item.get("loc", ()))
            partes.append(f"{campo}: {item.get('msg', '')}".strip(": "))
        return "; ".join(partes)
    if detalhe is not None:
        return str(detalhe)
    return "resposta inesperada da API"


class ClienteApi:
    """Fala com a MNIST Classifier API (GET /health e POST /predict)."""

    def __init__(self, endereco: str = PADRAO, timeout: float = TIMEOUT_PADRAO):
        self.endereco = endereco.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------ interno

    def _requisitar(self, metodo: str, caminho: str, corpo=None) -> dict:
        dados = None
        cabecalhos = {"Accept": "application/json"}
        if corpo is not None:
            dados = json.dumps(corpo).encode("utf-8")
            cabecalhos["Content-Type"] = "application/json"

        requisicao = urllib.request.Request(
            f"{self.endereco}{caminho}",
            data=dados,
            method=metodo,
            headers=cabecalhos,
        )

        try:
            with urllib.request.urlopen(requisicao, timeout=self.timeout) as resposta:
                texto = resposta.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            texto = exc.read().decode("utf-8", "replace")
            raise ErroDaApi(exc.code, _detalhe_legivel(texto)) from exc
        except urllib.error.URLError as exc:
            raise ErroDeConexao(
                f"Sem resposta de {self.endereco}: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise ErroDeConexao(
                f"{self.endereco} não respondeu em {self.timeout:.0f}s."
            ) from exc

        try:
            return json.loads(texto)
        except json.JSONDecodeError as exc:
            raise ErroDaApi(200, "a API devolveu um JSON inválido") from exc

    # ------------------------------------------------------------ público

    def saude(self) -> dict:
        """GET /health — confere conexão e se o modelo está na RAM."""
        return self._requisitar("GET", "/health")

    def prever(self, pixels: list[int]) -> dict:
        """POST /predict — envia 784 pixels e devolve dígito + confiança.

        Raises:
            ValueError: se a lista não tiver 784 pixels (falha do cliente,
                não da API — nem vale a pena gastar uma requisição).
            ErroDeConexao: servidor inacessível.
            ErroDaApi: status >= 400.
        """
        if len(pixels) != N_PIXELS:
            raise ValueError(f"esperados {N_PIXELS} pixels; recebidos {len(pixels)}.")
        retorno = self._requisitar("POST", "/predict", {"pixels": pixels})
        if not isinstance(retorno, dict) or "digito" not in retorno:
            raise ErroDaApi(200, "resposta sem o campo 'digito'")
        return retorno
