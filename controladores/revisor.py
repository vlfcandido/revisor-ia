"""
controladores/revisor.py — Controller Principal (Orquestra o Review)

No MVC, o Controller é quem orquestra tudo:
- Recebe a requisição (da View/API)
- Chama os Models e services necessários
- Retorna a resposta formatada

Este controller conecta o grafo LangGraph com a API FastAPI.
É a "cola" entre as camadas.

Agora também orquestra avaliações opcionais:
- Ragas: avalia qualidade do RAG (feature flag)
- DeepEval: avalia qualidade geral da resposta (feature flag)
"""

from modelos.esquemas import CodigoEntrada, ReviewSaida
from modelos.feature_flags import flags
from agentes.grafo_revisao import construir_grafo


async def revisar_codigo(entrada: CodigoEntrada) -> ReviewSaida:
    """
    Executa o fluxo completo de revisão de código.

    1. Constrói o grafo LangGraph
    2. Executa com o código de entrada
    3. (Opcional) Avalia com Ragas e DeepEval
    4. Formata a resposta

    Args:
        entrada: código + linguagem + estratégia (opcional)

    Returns:
        ReviewSaida com review, nota, justificativa, etc.
    """
    # Constrói o grafo de revisão
    grafo = construir_grafo()

    # Monta o estado inicial — só o que o usuário forneceu
    estado_inicial = {
        "codigo": entrada.codigo,
        "linguagem": entrada.linguagem,
    }

    # Se o usuário escolheu uma estratégia, usa ela
    # Se não, o grafo escolhe aleatoriamente
    if entrada.estrategia:
        estado_inicial["estrategia"] = entrada.estrategia

    # Executa o grafo — ele percorre todos os nós automaticamente
    # ainvoke = async invoke (execução assíncrona)
    resultado = await grafo.ainvoke(estado_inicial)

    # ── Avaliações opcionais (controladas por feature flags) ──
    review = resultado.get("review", "")
    contextos = resultado.get("boas_praticas", [])

    # Ragas: avalia qualidade do pipeline RAG
    if flags.FF_RAGAS_ATIVO and contextos:
        try:
            from avaliacao.ragas_eval import avaliar_com_ragas
            await avaliar_com_ragas(
                pergunta=f"Revise este código {entrada.linguagem}: {entrada.codigo[:200]}",
                resposta=review,
                contextos=contextos,
            )
        except Exception as e:
            print(f"⚠️  Ragas falhou (não bloqueia o review): {e}")

    # DeepEval: avalia qualidade geral da resposta
    if flags.FF_DEEPEVAL_ATIVO and review:
        try:
            from avaliacao.deepeval_eval import avaliar_com_deepeval
            await avaliar_com_deepeval(
                pergunta=f"Revise este código {entrada.linguagem}",
                resposta=review,
                contextos=contextos,
            )
        except Exception as e:
            print(f"⚠️  DeepEval falhou (não bloqueia o review): {e}")

    # Monta a resposta formatada (Pydantic valida automaticamente)
    return ReviewSaida(
        review=resultado.get("review", ""),
        nota_qualidade=resultado.get("nota_qualidade", 0),
        justificativa=resultado.get("justificativa", ""),
        estrategia=resultado.get("estrategia", "A"),
        tentativas=resultado.get("tentativas", 1),
        problemas_sintaxe=resultado.get("problemas_sintaxe", []),
        boas_praticas=resultado.get("boas_praticas", []),
    )
