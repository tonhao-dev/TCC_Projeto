import asyncio
import itertools
import re
from datetime import datetime, timedelta, timezone
import httpx
from typing import AsyncGenerator
from app.config import settings
from app.utils.rate_limiter import RateLimiter
import structlog

logger = structlog.get_logger()

_BRASILIA = timezone(timedelta(hours=-3))  # sem horário de verão desde 2019


def rpm_vigente(agora: datetime | None = None) -> int:
    """Limite por chave no horário atual de Brasília (00h-06h tem cota maior)."""
    noite = settings.CGU_RATE_LIMIT_RPM_NOITE
    hora = (agora or datetime.now(_BRASILIA)).astimezone(_BRASILIA).hour
    if noite and hora < 6:
        return noite
    return settings.CGU_RATE_LIMIT_RPM

ESTADOS = {
    "ACRE": "AC", "ALAGOAS": "AL", "AMAPÁ": "AP", "AMAZONAS": "AM",
    "BAHIA": "BA", "CEARÁ": "CE", "DISTRITO FEDERAL": "DF",
    "ESPÍRITO SANTO": "ES", "GOIÁS": "GO", "MARANHÃO": "MA",
    "MATO GROSSO": "MT", "MATO GROSSO DO SUL": "MS",
    "MINAS GERAIS": "MG", "PARÁ": "PA", "PARAÍBA": "PB",
    "PARANÁ": "PR", "PERNAMBUCO": "PE", "PIAUÍ": "PI",
    "RIO DE JANEIRO": "RJ", "RIO GRANDE DO NORTE": "RN",
    "RIO GRANDE DO SUL": "RS", "RONDÔNIA": "RO", "RORAIMA": "RR",
    "SANTA CATARINA": "SC", "SÃO PAULO": "SP", "SERGIPE": "SE",
    "TOCANTINS": "TO",
}

FUNCOES = {
    "Legislativa": "01", "Judiciária": "02", "Essencial à Justiça": "03",
    "Administração": "04", "Defesa Nacional": "05", "Segurança Pública": "06",
    "Relações Exteriores": "07", "Assistência Social": "08",
    "Previdência Social": "09", "Saúde": "10", "Trabalho": "11",
    "Educação": "12", "Cultura": "13", "Direitos da Cidadania": "14",
    "Urbanismo": "15", "Habitação": "16", "Saneamento": "17",
    "Gestão Ambiental": "18", "Gestão ambiental": "18",
    "Ciência e Tecnologia": "19", "Agricultura": "20",
    "Organização Agrária": "21", "Indústria": "22",
    "Comércio e Serviços": "23", "Comunicações": "24",
    "Energia": "25", "Transporte": "26", "Desporto e Lazer": "27",
    "Encargos Especiais": "28", "Encargos especiais": "28",
    "Meio Ambiente": "18", "Turismo": "23",
}


class CGUCollector:
    """Coletor de dados de emendas do Portal da Transparência."""

    BASE_URL = "https://api.portaldatransparencia.gov.br/api-de-dados"
    # Bloqueios 401/403 do WAF duram 1 a 2 minutos: espera 15, 30, 45 e 60 s antes de
    # desistir da emenda (e aí ela vai para retry_doc e, se continuar, para a DLQ).
    MAX_BLOQUEIOS_DOCS = 4
    ESPERA_BLOQUEIO_DOCS_S = 15

    def __init__(self):
        # Collect all API keys for rotation
        keys = [k.strip() for k in settings.CGU_API_KEYS.split(",") if k.strip()] if settings.CGU_API_KEYS else []
        if settings.CGU_API_KEY and settings.CGU_API_KEY not in keys:
            keys.insert(0, settings.CGU_API_KEY)
        if not keys:
            keys = [settings.CGU_API_KEY]
        self._api_keys = keys
        self._key_index = 0
        self._key_lock = asyncio.Lock()
        # Rate limiter POR CHAVE — cada chave tem seu próprio limite
        rpm_per_key = rpm_vigente()
        self._rpm_atual = rpm_per_key
        self._rate_limiters = {key: RateLimiter(rpm_per_key) for key in keys}
        # Manter rate_limiter global para compatibilidade (endpoints que não usam _next_headers_with_limit)
        total_rpm = rpm_per_key * len(keys)
        self.rate_limiter = RateLimiter(total_rpm)
        self.headers = {"chave-api-dados": keys[0]}
        logger.info("cgu_collector_init", num_keys=len(keys),
                     rpm_per_key=rpm_per_key, total_rpm=total_rpm)

    def update_rpm_for_current_hour(self):
        """Recalculate RPM — agora é no-op pois cada chave tem seu próprio limiter."""
        total_rpm = settings.CGU_RATE_LIMIT_RPM * len(self._api_keys)
        self.rate_limiter = RateLimiter(total_rpm)
        logger.info("rpm_updated", rpm_per_key=settings.CGU_RATE_LIMIT_RPM,
                     total_rpm=total_rpm, num_keys=len(self._api_keys))

    def _next_headers(self) -> dict:
        """Rotate to next API key and return headers."""
        key = self._api_keys[self._key_index % len(self._api_keys)]
        self._key_index += 1
        return {"chave-api-dados": key}

    async def _acquire_and_get_headers(self) -> dict:
        """Rotate to next key AND acquire its per-key rate limiter."""
        async with self._key_lock:
            rpm = rpm_vigente()
            if rpm != self._rpm_atual:  # virada 00h/06h em Brasília
                self._rpm_atual = rpm
                for limiter in self._rate_limiters.values():
                    limiter.rpm, limiter.interval = rpm, 60.0 / rpm
                logger.info("rpm_ajustado", rpm_por_chave=rpm)
            key = self._api_keys[self._key_index % len(self._api_keys)]
            self._key_index += 1
        await self._rate_limiters[key].acquire()
        return {"chave-api-dados": key}

    async def get_shared_client(self) -> httpx.AsyncClient:
        """Return a shared httpx client (create once, reuse)."""
        if not hasattr(self, '_shared_client') or self._shared_client is None:
            self._shared_client = httpx.AsyncClient(
                timeout=60.0,
                limits=httpx.Limits(
                    max_connections=60,
                    max_keepalive_connections=40,
                ),
                follow_redirects=True,
            )
        return self._shared_client

    async def close_shared_client(self):
        """Close the shared client when done."""
        if hasattr(self, '_shared_client') and self._shared_client is not None:
            await self._shared_client.aclose()
            self._shared_client = None

    async def coletar_emendas(self, ano: int, start_page: int = 1) -> AsyncGenerator[dict, None]:
        """Coleta todas as emendas de um ano com paginação automática."""
        pagina = start_page
        self.current_page = start_page
        total_coletadas = 0

        async with httpx.AsyncClient(timeout=30.0) as client:
            while True:
                headers = await self._acquire_and_get_headers()

                try:
                    response = await client.get(
                        f"{self.BASE_URL}/emendas",
                        params={"ano": ano, "pagina": pagina},
                        headers=headers,
                    )

                    if response.status_code == 429:
                        logger.warning("rate_limit_hit", ano=ano, pagina=pagina)
                        await asyncio.sleep(10)
                        continue

                    response.raise_for_status()
                    data = response.json()

                    if not data:
                        logger.info("coleta_completa", ano=ano, total=total_coletadas)
                        break

                    for emenda in data:
                        yield self._normalizar_emenda(emenda, ano)
                        total_coletadas += 1

                    pagina += 1
                    self.current_page = pagina

                except httpx.HTTPStatusError as e:
                    logger.error("http_error", status=e.response.status_code,
                                 ano=ano, pagina=pagina)
                    if e.response.status_code >= 500:
                        await asyncio.sleep(30)
                        continue
                    raise
                except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError,
                        httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                    logger.warning("network_error", tipo=type(e).__name__,
                                   ano=ano, pagina=pagina)
                    await asyncio.sleep(10)
                    continue

    def _normalizar_emenda(self, raw: dict, ano: int) -> dict:
        """Normaliza campos da API CGU para o schema unificado."""
        return {
            "codigo_emenda": raw.get("codigoEmenda"),
            "nome_autor": (raw.get("nomeAutor") or "").strip().upper(),
            "ano": ano,
            "tipo_emenda": raw.get("tipoEmenda", ""),
            "numero_emenda": raw.get("numeroEmenda", ""),
            "funcao": self._extrair_codigo_funcao(raw.get("funcao", "")),
            "funcao_nome": raw.get("funcao", ""),
            "subfuncao": self._extrair_codigo_subfuncao(raw.get("subfuncao", "")),
            "subfuncao_nome": raw.get("subfuncao", ""),
            "localidade": raw.get("localidadeDoGasto", ""),
            "uf": self._extrair_uf(raw.get("localidadeDoGasto", "")),
            "valor_empenhado": self._parse_valor(raw.get("valorEmpenhado", "0")),
            "valor_liquidado": self._parse_valor(raw.get("valorLiquidado", "0")),
            "valor_pago": self._parse_valor(raw.get("valorPago", "0")),
            "valor_resto_inscrito": self._parse_valor(raw.get("valorRestoInscrito", "0")),
            "valor_resto_cancelado": self._parse_valor(raw.get("valorRestoCancelado", "0")),
            "valor_resto_pago": self._parse_valor(raw.get("valorRestoPago", "0")),
            "descricao": raw.get("nomeSubfuncao", ""),
            "fonte": "Portal da Transparência",
        }

    @staticmethod
    def _parse_valor(valor_str) -> float:
        """Converte string monetária BR para float.

        A API às vezes separa o sinal do número ("- 4.504,92"), então todo
        espaço é removido antes da conversão. Sem dígitos ("-") vale 0.
        """
        if not valor_str:
            return 0.0
        if isinstance(valor_str, (int, float)):
            return float(valor_str)
        limpo = "".join(str(valor_str).split())
        if not any(ch.isdigit() for ch in limpo):
            return 0.0
        return float(limpo.replace(".", "").replace(",", "."))

    @staticmethod
    def _extrair_uf(localidade: str) -> str:
        """Extrai sigla UF de 'ACRE (UF)' ou 'RIO BRANCO - AC'."""
        if not localidade:
            return ""
        # Formato "ESTADO (UF)"
        if localidade.endswith("(UF)"):
            nome = localidade.replace(" (UF)", "").strip()
            return ESTADOS.get(nome, "")
        # Formato "CIDADE - UF"
        match = re.search(r" - ([A-Z]{2})$", localidade)
        if match:
            return match.group(1)
        return ""

    @staticmethod
    def _extrair_codigo_funcao(funcao_str: str) -> str:
        """Extrai código da função a partir do nome."""
        funcao_lower = funcao_str.lower().strip()
        # Match exato primeiro, depois substring (mais longo primeiro para evitar
        # "cultura" casar com "agricultura")
        for nome, codigo in sorted(FUNCOES.items(), key=lambda x: -len(x[0])):
            if funcao_lower == nome.lower():
                return codigo
        for nome, codigo in sorted(FUNCOES.items(), key=lambda x: -len(x[0])):
            if nome.lower() in funcao_lower:
                return codigo
        return ""

    @staticmethod
    def _extrair_codigo_subfuncao(subfuncao_str: str, db=None) -> str:
        """Extrai código numérico da subfunção.

        Aceita formatos:
        - "244 - Assistência comunitária" (documentos) → extrai "244"
        - "Assistência comunitária" (emendas) → busca na tabela subfuncoes
        """
        if not subfuncao_str:
            return ""
        # Formato com código numérico prefixado
        match = re.match(r"^(\d{3})", subfuncao_str)
        if match:
            return match.group(1)
        # Formato apenas nome — busca na tabela subfuncoes se db disponível
        if db:
            from sqlalchemy import text
            try:
                row = db.execute(text(
                    "SELECT codigo FROM subfuncoes WHERE LOWER(nome) = :n LIMIT 1"
                ), {"n": subfuncao_str.lower().strip()}).fetchone()
                if row:
                    return row[0]
            except Exception:
                pass
        return ""

    async def coletar_documentos_emenda(self, codigo_emenda: str,
                                        max_network_retries: int = 5) -> list[dict]:
        """Coleta documentos vinculados a uma emenda pelo código."""
        documentos = []
        pagina = 1
        network_retries = 0
        bloqueios = 0

        async with httpx.AsyncClient(timeout=30.0) as client:
            while True:
                headers = await self._acquire_and_get_headers()
                try:
                    response = await client.get(
                        f"{self.BASE_URL}/emendas/documentos/{codigo_emenda}",
                        params={"pagina": pagina},
                        headers=headers,
                    )

                    if response.status_code == 429:
                        logger.warning("rate_limit_hit_docs", codigo=codigo_emenda)
                        await asyncio.sleep(10)
                        continue

                    if response.status_code in (401, 403):
                        # Bloqueio passageiro do WAF da CGU. Antes isto caía no break
                        # abaixo e a emenda ficava sem documentos, sem erro nem DLQ.
                        bloqueios += 1
                        if bloqueios > self.MAX_BLOQUEIOS_DOCS:
                            response.raise_for_status()  # vira exceção: retry_doc e DLQ
                        logger.warning("bloqueio_docs", status=response.status_code,
                                       codigo=codigo_emenda, tentativa=bloqueios)
                        await asyncio.sleep(self.ESPERA_BLOQUEIO_DOCS_S * bloqueios)
                        continue

                    if response.status_code == 404:
                        logger.info("sem_documentos", codigo=codigo_emenda)
                        break

                    response.raise_for_status()
                    data = response.json()

                    if not data:
                        break

                    for doc in data:
                        documentos.append({
                            "codigo_documento": str(doc.get("codigoDocumento", "")),
                            "fase": doc.get("fase", ""),
                            "valor": self._parse_valor(doc.get("valor", "0")),
                            "data_documento": doc.get("data"),
                            "tipo_documento": doc.get("especieTipo", "") or doc.get("tipoDocumento", ""),
                            "observacao": doc.get("observacao", ""),
                        })

                    pagina += 1
                    network_retries = 0  # reset on success
                    bloqueios = 0

                except httpx.HTTPStatusError as e:
                    logger.error("http_error_docs", status=e.response.status_code,
                                 codigo=codigo_emenda)
                    if e.response.status_code >= 500:
                        await asyncio.sleep(30)
                        continue
                    # Devolver a lista parcial aqui gravava a emenda como completa.
                    raise
                except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError,
                        httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                    network_retries += 1
                    logger.warning("network_error_docs", tipo=type(e).__name__,
                                   codigo=codigo_emenda, retry=network_retries)
                    if network_retries >= max_network_retries:
                        logger.error("max_retries_docs", codigo=codigo_emenda)
                        raise
                    await asyncio.sleep(10)
                    continue

        logger.info("documentos_coletados", codigo=codigo_emenda, total=len(documentos))
        return documentos

    async def coletar_beneficiarios_documento(self, codigo_documento: str,
                                              max_network_retries: int = 5) -> list[dict]:
        """Coleta favorecidos finais de um documento de despesa."""
        beneficiarios = []
        pagina = 1
        network_retries = 0

        async with httpx.AsyncClient(timeout=30.0) as client:
            while True:
                headers = await self._acquire_and_get_headers()
                try:
                    response = await client.get(
                        f"{self.BASE_URL}/despesas/favorecidos-finais-por-documento",
                        params={"codigoDocumento": codigo_documento, "pagina": pagina},
                        headers=headers,
                    )

                    if response.status_code == 429:
                        logger.warning("rate_limit_hit_benef", codigo=codigo_documento)
                        await asyncio.sleep(10)
                        continue

                    if response.status_code == 404:
                        break

                    response.raise_for_status()
                    data = response.json()

                    if not data:
                        break

                    for fav in data:
                        cpf_cnpj = fav.get("cnpjCpf", "") or fav.get("cpfCnpj", "")
                        tipo = "PJ" if len(cpf_cnpj.replace(".", "").replace("-", "").replace("/", "")) > 11 else "PF"
                        beneficiarios.append({
                            "codigo_pessoa": fav.get("codigoPessoa", ""),
                            "nome": (fav.get("nome", "") or fav.get("razaoSocial", "")).strip(),
                            "tipo_pessoa": tipo,
                            "cpf_cnpj": cpf_cnpj,
                            "valor_recebido": self._parse_valor(fav.get("valor", "0")),
                        })

                    pagina += 1
                    network_retries = 0  # reset on success

                except httpx.HTTPStatusError as e:
                    logger.error("http_error_benef", status=e.response.status_code,
                                 codigo=codigo_documento)
                    if e.response.status_code >= 500:
                        await asyncio.sleep(30)
                        continue
                    break
                except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError,
                        httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                    network_retries += 1
                    logger.warning("network_error_benef", tipo=type(e).__name__,
                                   codigo=codigo_documento, retry=network_retries)
                    if network_retries >= max_network_retries:
                        logger.error("max_retries_benef", codigo=codigo_documento)
                        break
                    await asyncio.sleep(10)
                    continue

        logger.info("beneficiarios_coletados", codigo=codigo_documento, total=len(beneficiarios))
        return beneficiarios

    async def coletar_detalhes_documento(self, codigo_documento: str,
                                        max_network_retries: int = 5,
                                        client: httpx.AsyncClient | None = None) -> dict | None:
        """Coleta detalhes completos de um documento de despesa."""
        network_retries = 0
        _client = client or await self.get_shared_client()
        while True:
            headers = await self._acquire_and_get_headers()
            try:
                response = await _client.get(
                    f"{self.BASE_URL}/despesas/documentos/{codigo_documento}",
                    headers=headers,
                )

                if response.status_code in (429, 401, 403, 405):
                    network_retries += 1
                    if network_retries <= 2:
                        logger.warning("rate_limit_or_auth_retry",
                                       status=response.status_code,
                                       codigo=codigo_documento,
                                       retry=network_retries)
                    if network_retries >= max_network_retries:
                        raise httpx.HTTPStatusError(
                            f"Max retries on {response.status_code}",
                            request=response.request,
                            response=response,
                        )
                    await asyncio.sleep(3 * network_retries)
                    continue

                if response.status_code == 404:
                    return None

                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                logger.error("http_error_doc_detalhe", status=e.response.status_code,
                             codigo=codigo_documento)
                if e.response.status_code >= 500:
                    await asyncio.sleep(30)
                    continue
                raise
            except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError,
                    httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                network_retries += 1
                logger.warning("network_error_detalhe", tipo=type(e).__name__,
                               codigo=codigo_documento, retry=network_retries)
                if network_retries >= max_network_retries:
                    logger.error("max_retries_detalhe", codigo=codigo_documento)
                    # None marcaria o documento como coletado sem detalhe (igual a 404).
                    raise
                await asyncio.sleep(10)
                continue

    # -----------------------------------------------------------------------
    # Sanções
    # -----------------------------------------------------------------------
    async def _coletar_paginado(self, endpoint: str, label: str,
                                start_page: int = 1,
                                params_extra: dict | None = None,
                                max_network_retries: int = 5) -> AsyncGenerator[dict, None]:
        """Coleta genérica paginada para qualquer endpoint da API CGU."""
        pagina = start_page
        self.current_page = start_page
        total = 0
        network_retries = 0

        async with httpx.AsyncClient(timeout=60.0) as client:
            while True:
                headers = await self._acquire_and_get_headers()
                params = {"pagina": pagina}
                if params_extra:
                    params.update(params_extra)

                try:
                    response = await client.get(
                        f"{self.BASE_URL}/{endpoint}",
                        params=params,
                        headers=headers,
                    )

                    if response.status_code == 429:
                        logger.warning("rate_limit_hit", endpoint=label, pagina=pagina)
                        await asyncio.sleep(10)
                        continue

                    if response.status_code in (301, 302, 303, 307, 308, 404):
                        logger.info("paginacao_fim", endpoint=label,
                                    pagina=pagina, status=response.status_code)
                        break

                    response.raise_for_status()
                    data = response.json()

                    if not data:
                        logger.info("coleta_completa", endpoint=label, total=total)
                        break

                    for item in data:
                        yield item
                        total += 1

                    pagina += 1
                    self.current_page = pagina
                    network_retries = 0

                except httpx.HTTPStatusError as e:
                    logger.error("http_error", endpoint=label,
                                 status=e.response.status_code, pagina=pagina)
                    if e.response.status_code >= 500:
                        await asyncio.sleep(30)
                        continue
                    raise
                except (httpx.ReadError, httpx.ConnectError,
                        httpx.RemoteProtocolError,
                        httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                    network_retries += 1
                    logger.warning("network_error", endpoint=label,
                                   tipo=type(e).__name__, pagina=pagina,
                                   retry=network_retries)
                    if network_retries >= max_network_retries:
                        logger.error("max_retries", endpoint=label, pagina=pagina)
                        break
                    await asyncio.sleep(10)
                    continue

    async def coletar_sancoes_ceis(self, start_page: int = 1) -> AsyncGenerator[dict, None]:
        """Coleta cadastro CEIS (empresas inidôneas e suspensas)."""
        async for item in self._coletar_paginado("ceis", "ceis", start_page):
            yield self._normalizar_sancao(item, "CEIS")

    async def coletar_sancoes_cnep(self, start_page: int = 1) -> AsyncGenerator[dict, None]:
        """Coleta cadastro CNEP (empresas punidas)."""
        async for item in self._coletar_paginado("cnep", "cnep", start_page):
            yield self._normalizar_sancao(item, "CNEP")

    async def coletar_sancoes_cepim(self, start_page: int = 1) -> AsyncGenerator[dict, None]:
        """Coleta cadastro CEPIM (entidades impedidas)."""
        async for item in self._coletar_paginado("cepim", "cepim", start_page):
            pj = self._obj(item, "pessoaJuridica")
            yield {
                "cnpj": self._primeiro(pj.get("cnpjFormatado"), item.get("cnpjSancionado"),
                                       item.get("cnpj")),
                "nome": self._primeiro(pj.get("nome"), pj.get("razaoSocialReceita"),
                                       item.get("nomeSancionado"), item.get("razaoSocial")),
                "motivo": self._flatten_field(item.get("motivo") or item.get("motivoImpedimento", "")),
                "orgao_maximo": self._primeiro(self._obj(item, "orgaoSuperior").get("nome"),
                                               self._flatten_field(item.get("orgaoEntidade", ""))),
            }

    async def coletar_sancoes_ceaf(self, start_page: int = 1) -> AsyncGenerator[dict, None]:
        """Coleta cadastro CEAF (servidores punidos)."""
        async for item in self._coletar_paginado("ceaf", "ceaf", start_page):
            punicao = self._obj(item, "punicao")
            pessoa = self._obj(item, "pessoa")
            yield {
                "cpf": self._primeiro(punicao.get("cpfPunidoFormatado"), pessoa.get("cpfFormatado"),
                                      item.get("cpfSancionado")),
                "nome": self._primeiro(punicao.get("nomePunido"), pessoa.get("nome"),
                                       item.get("nomeSancionado")),
                "tipo_punicao": self._primeiro(self._obj(item, "tipoPunicao").get("descricao"),
                                               self._flatten_field(item.get("descricaoPunicao", ""))),
                "orgao_lotacao": self._primeiro(self._obj(item, "orgaoLotacao").get("nome"),
                                                self._flatten_field(item.get("orgaoLotacao", ""))),
                "data_inicio": item.get("dataPublicacao"),
                # A API do CEAF não traz data de fim da punição.
                "data_fim": item.get("dataFimPunicao"),
            }

    @staticmethod
    def _obj(item: dict, chave: str) -> dict:
        """Devolve o objeto aninhado `item[chave]`, ou {} se faltar ou não for objeto."""
        val = item.get(chave)
        return val if isinstance(val, dict) else {}

    @staticmethod
    def _primeiro(*valores) -> str:
        """Primeiro valor de texto não vazio, já sem espaços nas pontas."""
        for v in valores:
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    @staticmethod
    def _flatten_field(val, preferred_keys=("descricao", "nome", "razaoSocial")) -> str:
        """Converte campo que pode ser dict/list para string."""
        if val is None:
            return ""
        if isinstance(val, str):
            return val.strip()
        if isinstance(val, dict):
            for k in preferred_keys:
                for dk in val:
                    if k.lower() in dk.lower() and val[dk]:
                        return str(val[dk]).strip()
            # fallback: primeiro valor string não-vazio
            for v in val.values():
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return str(val)
        if isinstance(val, list):
            partes = (CGUCollector._flatten_field(v, preferred_keys) for v in val)
            return "; ".join(p for p in partes if p)
        return str(val)

    def _normalizar_sancao(self, raw: dict, tipo: str) -> dict:
        """Normaliza dados de sanção CEIS/CNEP.

        A API manda quem foi punido nos objetos `sancionado` e `pessoa`
        (CeisDTO/CnepDTO). Os campos soltos antigos ficam como reserva.
        """
        sancionado = self._obj(raw, "sancionado")
        pessoa = self._obj(raw, "pessoa")
        cpf_cnpj = self._primeiro(sancionado.get("codigoFormatado"),
                                  pessoa.get("cnpjFormatado"), pessoa.get("cpfFormatado"),
                                  raw.get("cpfCnpjSancionado"), raw.get("codigoSancionado"),
                                  raw.get("cnpjCpf"))
        result = {
            "cpf_cnpj": cpf_cnpj,
            "nome": self._primeiro(sancionado.get("nome"), pessoa.get("nome"),
                                   raw.get("nomeSancionado")),
            "tipo_sancao": self._flatten_field(raw.get("tipoSancao", "")),
            "orgao_sancionador": self._flatten_field(raw.get("orgaoSancionador", "")),
            "data_inicio": raw.get("dataInicioSancao"),
            "data_fim": raw.get("dataFimSancao"),
        }
        if tipo == "CNEP":
            result["valor_multa"] = self._parse_valor(raw.get("valorMulta", "0"))
        else:
            result["fundamentacao"] = self._flatten_field(
                raw.get("fundamentacao") or raw.get("fundamentacaoLegal", ""))
        return result

    # -----------------------------------------------------------------------
    # Convênios
    # -----------------------------------------------------------------------
    async def coletar_convenios(self, data_inicial: str, data_final: str,
                                start_page: int = 1) -> AsyncGenerator[dict, None]:
        """Coleta convênios do Poder Executivo Federal."""
        params = {"dataInicial": data_inicial, "dataFinal": data_final}
        async for item in self._coletar_paginado("convenios", "convenios",
                                                  start_page, params):
            dim = item.get("dimConvenio") or {}
            convenente = item.get("convenente") or {}
            mun = item.get("municipioConvenente") or {}
            uf_obj = mun.get("uf") or {}
            orgao = item.get("orgao") or {}
            orgao_max = orgao.get("orgaoMaximo") or {}
            subfuncao_obj = item.get("subfuncao") or {}
            funcao_obj = subfuncao_obj.get("funcao") or {}

            cpf_cnpj = (convenente.get("cnpjFormatado")
                        or convenente.get("cpfFormatado") or "").strip()
            # API tem sigla/nome invertidos para UF
            uf_sigla = uf_obj.get("nome") or uf_obj.get("sigla", "")
            if len(uf_sigla) > 2:
                uf_sigla = uf_obj.get("sigla", uf_sigla)

            yield {
                "numero_convenio": str(dim.get("numero") or dim.get("codigo", "")),
                "situacao": item.get("situacao", ""),
                "orgao_concedente": orgao_max.get("nome") or orgao.get("nome", ""),
                "cpf_cnpj_convenente": cpf_cnpj,
                "nome_convenente": (convenente.get("nome") or "").strip(),
                "uf": uf_sigla,
                "municipio": mun.get("nomeIBGE", ""),
                "objeto": dim.get("objeto", ""),
                "valor_convenio": self._parse_valor(item.get("valor", 0)),
                "valor_liberado": self._parse_valor(item.get("valorLiberado", 0)),
                "data_inicio": item.get("dataInicioVigencia"),
                "data_fim": item.get("dataFinalVigencia"),
                "funcao": funcao_obj.get("codigoFuncao", ""),
                "subfuncao": subfuncao_obj.get("codigoSubfuncao", ""),
            }
