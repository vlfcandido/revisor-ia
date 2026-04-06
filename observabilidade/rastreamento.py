"""
observabilidade/rastreamento.py — LangSmith (Tracing e Monitoramento)

LangSmith é a plataforma de observabilidade da LangChain.
Permite rastrear (trace) cada chamada ao LLM, ver os inputs/outputs,
medir latência, e debugar problemas em apps de IA.

╔══════════════════════════════════════════════════════════════╗
║  POR QUE OBSERVABILIDADE EM LLM APPS?                       ║
║                                                              ║
║  LLMs são "caixas pretas" — você manda um prompt e recebe    ║
║  uma resposta, mas não sabe o que aconteceu no meio.         ║
║                                                              ║
║  LangSmith resolve isso mostrando:                           ║
║  - Cada chamada ao LLM (prompt completo, resposta, tempo)    ║
║  - Cadeia de chamadas (qual nó chamou qual)                  ║
║  - Custos (tokens usados por chamada)                        ║
║  - Erros (onde falhou e por quê)                             ║
║                                                              ║
║  É como ter o DevTools do Chrome, mas pra apps de IA.        ║
╚══════════════════════════════════════════════════════════════╝

Conceitos:
- Trace: registro completo de uma execução (do início ao fim)
- Run: cada operação dentro de um trace (uma chamada LLM, um RAG, etc.)
- Project: agrupa traces (ex: "revisor-ia-producao")
- Dashboard: UI web pra visualizar traces (smith.langchain.com)

Como usar:
    1. Crie conta em smith.langchain.com
    2. Gere uma API key
    3. Adicione no .env:
       LANGSMITH_API_KEY=lsv2_pt_xxxx
       LANGSMITH_PROJECT=revisor-ia

Setup:
    from observabilidade.rastreamento import configurar_langsmith
    configurar_langsmith()  # chame no startup da app
"""

import os
from typing import Optional

from modelos.configuracao import configuracao


def configurar_langsmith(
    projeto: Optional[str] = None,
    ativo: bool = True,
) -> bool:
    """
    Configura o LangSmith pra rastrear chamadas ao LLM.

    O LangSmith funciona via variáveis de ambiente — o LangChain
    detecta automaticamente e começa a enviar traces.

    Variáveis que o LangSmith usa:
    - LANGSMITH_TRACING: "true" pra ativar o tracing
    - LANGSMITH_API_KEY: chave de autenticação
    - LANGSMITH_PROJECT: nome do projeto no dashboard
    - LANGSMITH_ENDPOINT: URL da API (padrão: https://api.smith.langchain.com)

    Args:
        projeto: nome do projeto no LangSmith (padrão: "revisor-ia")
        ativo: True pra ativar, False pra desativar

    Returns:
        True se configurou com sucesso, False se falta a API key

    Exemplo:
        >>> configurar_langsmith()
        True  # se LANGSMITH_API_KEY está no .env
        False # se não tem API key (tracing desativado)
    """
    # Se desativado explicitamente, desliga tudo
    if not ativo:
        os.environ["LANGSMITH_TRACING"] = "false"
        print("⚠️  LangSmith desativado")
        return False

    # Verifica se tem API key configurada
    api_key = os.environ.get("LANGSMITH_API_KEY", "")

    if not api_key:
        # Sem API key — funciona sem LangSmith (sem tracing)
        os.environ["LANGSMITH_TRACING"] = "false"
        print("ℹ️  LangSmith não configurado (sem LANGSMITH_API_KEY)")
        print("   Pra ativar: adicione LANGSMITH_API_KEY no .env")
        print("   Crie conta em: https://smith.langchain.com")
        return False

    # ── Configura as variáveis de ambiente ──

    # Ativa o tracing — a partir daqui, toda chamada LangChain é rastreada
    os.environ["LANGSMITH_TRACING"] = "true"

    # Nome do projeto (aparece no dashboard)
    nome_projeto = projeto or os.environ.get("LANGSMITH_PROJECT", "revisor-ia")
    os.environ["LANGSMITH_PROJECT"] = nome_projeto

    # Endpoint da API (usa o padrão se não definido)
    if "LANGSMITH_ENDPOINT" not in os.environ:
        os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

    print(f"✅ LangSmith ativado!")
    print(f"   Projeto: {nome_projeto}")
    print(f"   Dashboard: https://smith.langchain.com/o/default/projects")

    return True


def criar_run_personalizado(
    nome: str,
    tipo: str = "chain",
    metadata: Optional[dict] = None,
) -> Optional[object]:
    """
    Cria um "run" personalizado no LangSmith.

    Um run é uma unidade de execução rastreada. Normalmente
    o LangChain cria runs automaticamente, mas você pode
    criar manualmente pra rastrear operações customizadas.

    Útil pra rastrear:
    - Operações que não usam LangChain (ex: busca no pgvector)
    - Lógica de negócio (ex: decisão do teste A/B)
    - Métricas customizadas

    Args:
        nome: nome do run (ex: "busca_rag", "teste_ab")
        tipo: tipo do run ("chain", "llm", "tool", "retriever")
        metadata: dados extras pra associar ao run

    Returns:
        Objeto RunTree se LangSmith está ativo, None se não

    Exemplo:
        >>> run = criar_run_personalizado(
        ...     nome="busca_rag",
        ...     tipo="retriever",
        ...     metadata={"query": "nomes de variáveis", "top_k": 5}
        ... )
    """
    # Verifica se o tracing está ativo
    if os.environ.get("LANGSMITH_TRACING") != "true":
        return None

    try:
        # Importa o RunTree do LangSmith SDK
        from langsmith import RunTree

        # Cria o run com os dados fornecidos
        run = RunTree(
            name=nome,
            run_type=tipo,
            extra={"metadata": metadata or {}},
        )

        return run
    except ImportError:
        # langsmith não instalado — retorna None silenciosamente
        print("⚠️  langsmith SDK não instalado. pip install langsmith")
        return None


def obter_url_trace() -> Optional[str]:
    """
    Retorna a URL do dashboard do LangSmith pro projeto atual.

    Útil pra incluir em logs ou respostas da API,
    permitindo que o dev vá direto pro dashboard.

    Returns:
        URL do dashboard ou None se LangSmith não está ativo
    """
    if os.environ.get("LANGSMITH_TRACING") != "true":
        return None

    projeto = os.environ.get("LANGSMITH_PROJECT", "revisor-ia")
    return f"https://smith.langchain.com/o/default/projects/p/{projeto}"


# ══════════════════════════════════════════════════════════════
# DECORADOR PARA RASTREAMENTO
#
# Alternativa ao RunTree manual — usa o decorator
# @rastrear() pra marcar funções que devem ser rastreadas.
#
# O LangSmith tem seu próprio decorator (@traceable),
# mas criamos um wrapper pra manter tudo em português
# e pra funcionar mesmo sem LangSmith instalado.
# ══════════════════════════════════════════════════════════════

def rastrear(nome: Optional[str] = None):
    """
    Decorador que rastreia a execução de uma função no LangSmith.

    Se o LangSmith estiver configurado, usa o @traceable.
    Se não, é um no-op (não faz nada).

    Args:
        nome: nome do trace (padrão: nome da função)

    Exemplo:
        @rastrear("buscar_praticas")
        async def buscar_boas_praticas(query: str):
            ...

        # No LangSmith dashboard, vai aparecer como "buscar_praticas"
    """
    def decorador(func):
        # Tenta usar o @traceable do LangSmith
        try:
            from langsmith import traceable

            # Aplica o decorator do LangSmith
            nome_trace = nome or func.__name__
            return traceable(name=nome_trace)(func)

        except ImportError:
            # LangSmith não instalado — retorna a função original
            return func

    return decorador
