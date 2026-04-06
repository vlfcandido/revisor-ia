"""
visoes/api.py — API REST (View do MVC)

No MVC, a View é a camada que o usuário interage.
Aqui usamos FastAPI pra criar endpoints REST.

FastAPI:
- Framework web moderno e rápido pra Python
- Validação automática com Pydantic (integração nativa)
- Documentação automática (Swagger UI em /docs)
- Suporte async nativo

Endpoints:
- POST /revisar      → revisa código usando o agente LangGraph
- POST /rag/ingerir  → adiciona documento ao knowledge base
- GET  /rag/buscar   → busca boas práticas no pgvector
- POST /teste-ab     → roda teste A/B entre estratégias
- GET  /saude        → health check
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from banco.conexao import criar_tabelas, obter_sessao
from modelos.esquemas import (
    CodigoEntrada,
    ReviewSaida,
    DocumentoEntrada,
    BuscaRAGSaida,
    ResultadoAB,
)
from controladores.revisor import revisar_codigo
from controladores.rag import ingerir, buscar
from controladores.teste_ab import executar_teste_ab
from rag.ingestao import popular_base_inicial


# ══════════════════════════════════════════════════════════════
# LIFESPAN — o que roda quando a API sobe/desce
#
# O lifespan é um context manager que roda:
# - ANTES da API aceitar requests (startup)
# - DEPOIS da API parar (shutdown)
#
# Usamos pra:
# - Criar as tabelas no banco (se não existem)
# - Popular a base com dados iniciais (seed)
# ══════════════════════════════════════════════════════════════

@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    """
    Ciclo de vida da aplicação.

    Startup: cria tabelas e popula base.
    Shutdown: (nada por enquanto).
    """
    print("🚀 Iniciando Revisor IA...")

    # Configura LangSmith (observabilidade/tracing) — respeitando feature flag
    from modelos.feature_flags import flags
    from observabilidade.rastreamento import configurar_langsmith
    configurar_langsmith(ativo=flags.FF_LANGSMITH_ATIVO)

    # Cria as tabelas no banco (CREATE TABLE IF NOT EXISTS)
    await criar_tabelas()
    print("✅ Tabelas criadas/verificadas")

    # Habilita a extensão pgvector no PostgreSQL
    from sqlalchemy import text
    from banco.conexao import engine
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    print("✅ Extensão pgvector habilitada")

    # Popula a base com boas práticas (se estiver vazia)
    from banco.conexao import fabrica_sessao
    async with fabrica_sessao() as sessao:
        await popular_base_inicial(sessao)

    print("🟢 Revisor IA pronto!")

    # yield = ponto onde a API começa a aceitar requests
    yield

    # Código após o yield roda no shutdown
    print("🔴 Encerrando Revisor IA...")


# ══════════════════════════════════════════════════════════════
# APP — instância do FastAPI
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title="Revisor IA",
    description="API de revisão de código com IA — projeto de estudo",
    version="0.1.0",
    lifespan=ciclo_de_vida,
)


# ══════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════


@app.get("/saude")
async def verificar_saude():
    """
    Health check — verifica se a API está no ar.

    Retorna um JSON simples. Útil pra monitoramento
    e pra testar se a API está respondendo.
    """
    return {"status": "ok", "servico": "Revisor IA"}


@app.post("/revisar", response_model=ReviewSaida)
async def endpoint_revisar(entrada: CodigoEntrada):
    """
    Revisa código usando o agente LangGraph.

    Fluxo:
    1. Analisa sintaxe
    2. Busca boas práticas no RAG
    3. Gera review (estratégia A ou B)
    4. Avalia qualidade (LLM-as-judge)
    5. Re-gera se nota < 7 (max 2x)

    Body:
        {"codigo": "def f(): pass", "linguagem": "python", "estrategia": "A"}
    """
    # Chama o controller que orquestra tudo
    resultado = await revisar_codigo(entrada)
    return resultado


@app.post("/rag/ingerir")
async def endpoint_ingerir(
    entrada: DocumentoEntrada,
    sessao: AsyncSession = Depends(obter_sessao),
):
    """
    Ingere um documento no knowledge base (RAG).

    Pipeline: texto → chunks → embeddings → pgvector

    Body:
        {"texto": "Use nomes descritivos...", "fonte": "clean_code"}
    """
    resultado = await ingerir(sessao, entrada)
    return resultado


@app.get("/rag/buscar", response_model=BuscaRAGSaida)
async def endpoint_buscar(
    q: str = Query(..., description="Texto de busca"),
    top_k: int = Query(default=5, ge=1, le=20, description="Quantidade de resultados"),
    sessao: AsyncSession = Depends(obter_sessao),
):
    """
    Busca boas práticas relevantes no RAG (pgvector).

    Faz busca por similaridade vetorial:
    - Gera embedding da query
    - Encontra os vetores mais próximos no banco
    - Retorna os textos correspondentes

    Params:
        q: texto de busca (ex: "como nomear variáveis?")
        top_k: quantos resultados retornar (1-20)
    """
    resultado = await buscar(sessao, q, top_k)
    return resultado


@app.post("/teste-ab", response_model=ResultadoAB)
async def endpoint_teste_ab(
    entrada: CodigoEntrada,
    rodadas: int = Query(default=2, ge=1, le=10, description="Número de rodadas"),
):
    """
    Executa teste A/B entre estratégias de prompt.

    Compara a estratégia A (detalhada) com a B (concisa)
    executando ambas com o mesmo código e comparando notas.

    Body:
        {"codigo": "def f(): pass", "linguagem": "python"}
    Params:
        rodadas: quantas vezes executar cada estratégia (padrão: 2)

    Nota: cada rodada faz 2+ chamadas ao LLM, pode demorar.
    """
    resultado = await executar_teste_ab(
        codigo=entrada.codigo,
        linguagem=entrada.linguagem,
        rodadas=rodadas,
    )
    return resultado


# ══════════════════════════════════════════════════════════════
# ENDPOINTS DE FEATURE FLAGS
#
# Permitem ver e alterar flags em runtime (sem restart).
# Em produção, isso seria protegido por auth.
# ══════════════════════════════════════════════════════════════

@app.get("/flags")
async def endpoint_listar_flags():
    """
    Lista todas as feature flags e seus valores atuais.

    Feature flags controlam quais funcionalidades estão ativas:
    - FF_LANGSMITH_ATIVO: tracing/observabilidade
    - FF_RAGAS_ATIVO: avaliação de RAG
    - FF_DEEPEVAL_ATIVO: avaliação geral de qualidade
    - FF_LOOP_REGENERACAO: re-geração quando nota baixa
    - FF_RAG_ATIVO: busca de boas práticas
    - FF_TESTE_AB_ATIVO: teste A/B de estratégias
    """
    from modelos.feature_flags import listar_flags
    return listar_flags()


@app.put("/flags/{nome_flag}")
async def endpoint_alterar_flag(nome_flag: str, valor: bool):
    """
    Altera uma feature flag em runtime.

    Exemplo: PUT /flags/FF_RAGAS_ATIVO?valor=false
    → Desliga a avaliação Ragas (fica mais rápido)

    NOTA: alterações são temporárias (voltam ao .env no restart).
    Em produção, usaria LaunchDarkly, Unleash, ou similar.
    """
    from modelos.feature_flags import flags

    if not hasattr(flags, nome_flag):
        return {"erro": f"Flag '{nome_flag}' não existe"}

    setattr(flags, nome_flag, valor)
    return {"flag": nome_flag, "valor": valor, "status": "atualizado"}


# ══════════════════════════════════════════════════════════════
# ENDPOINT DE AVALIAÇÃO AVULSA
#
# Permite rodar Ragas e DeepEval manualmente num review
# que já foi gerado. Útil pra testar e comparar as métricas.
# ══════════════════════════════════════════════════════════════

@app.post("/avaliar")
async def endpoint_avaliar(entrada: CodigoEntrada):
    """
    Avalia um código com TODAS as métricas disponíveis.

    Roda o review + Ragas + DeepEval + versões manuais.
    Útil pra comparar frameworks vs implementação manual.

    Retorna um relatório completo com todas as métricas.
    """
    # 1. Gera o review normalmente
    resultado_review = await revisar_codigo(entrada)

    relatorio = {
        "review": resultado_review.review,
        "nota_llm_judge": resultado_review.nota_qualidade,
        "estrategia": resultado_review.estrategia,
        "avaliacoes": {},
    }

    contextos = resultado_review.boas_praticas
    pergunta = f"Revise este código {entrada.linguagem}: {entrada.codigo[:200]}"

    # 2. Ragas (se ativo)
    try:
        from avaliacao.ragas_eval import avaliar_com_ragas, avaliar_rag_manual
        from modelos.feature_flags import flags

        if flags.FF_RAGAS_ATIVO:
            resultado_ragas = await avaliar_com_ragas(
                pergunta=pergunta,
                resposta=resultado_review.review,
                contextos=contextos,
            )
            if resultado_ragas:
                relatorio["avaliacoes"]["ragas"] = {
                    "faithfulness": resultado_ragas.faithfulness,
                    "answer_relevancy": resultado_ragas.answer_relevancy,
                    "context_precision": resultado_ragas.context_precision,
                    "context_recall": resultado_ragas.context_recall,
                    "media": resultado_ragas.media,
                }

        # Versão manual (sempre roda pra comparação)
        resultado_manual_rag = await avaliar_rag_manual(
            pergunta=pergunta,
            resposta=resultado_review.review,
            contextos=contextos,
        )
        relatorio["avaliacoes"]["rag_manual"] = resultado_manual_rag
    except Exception as e:
        relatorio["avaliacoes"]["ragas_erro"] = str(e)

    # 3. DeepEval (se ativo)
    try:
        from avaliacao.deepeval_eval import avaliar_com_deepeval, avaliar_qualidade_manual
        from modelos.feature_flags import flags

        if flags.FF_DEEPEVAL_ATIVO:
            resultado_deepeval = await avaliar_com_deepeval(
                pergunta=pergunta,
                resposta=resultado_review.review,
                contextos=contextos,
            )
            if resultado_deepeval:
                relatorio["avaliacoes"]["deepeval"] = {
                    "answer_relevancy": resultado_deepeval.answer_relevancy,
                    "faithfulness": resultado_deepeval.faithfulness,
                    "hallucination": resultado_deepeval.hallucination,
                    "geval_score": resultado_deepeval.geval_score,
                    "passou_tudo": resultado_deepeval.passou_tudo,
                }

        # Versão manual (sempre roda pra comparação)
        resultado_manual_qual = await avaliar_qualidade_manual(
            pergunta=pergunta,
            resposta=resultado_review.review,
            contextos=contextos,
        )
        relatorio["avaliacoes"]["qualidade_manual"] = resultado_manual_qual
    except Exception as e:
        relatorio["avaliacoes"]["deepeval_erro"] = str(e)

    return relatorio
