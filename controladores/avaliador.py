"""
controladores/avaliador.py — Controller de Avaliação (LLM-as-Judge)

Encapsula a lógica de avaliação de qualidade de reviews.

Conceito: LLM-as-Judge
- Usar um LLM pra avaliar a saída de outro LLM
- Técnica comum pra medir qualidade automaticamente
- Alternativa a avaliação humana (mais rápido, mais barato)
- Limitação: o juiz pode ter os mesmos vieses do modelo avaliado
"""

from agentes.ferramentas import avaliar_review_com_llm


async def avaliar_review(codigo: str, review: str) -> dict:
    """
    Avalia a qualidade de um code review.

    Usa LLM-as-Judge: um LLM separado avalia o review
    e dá nota de 0 a 10 com justificativa.

    Args:
        codigo: código original que foi revisado
        review: texto do review a avaliar

    Returns:
        Dict com nota (float) e justificativa (str)
    """
    nota, justificativa = await avaliar_review_com_llm(codigo, review)

    return {
        "nota": nota,
        "justificativa": justificativa,
    }
