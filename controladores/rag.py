"""
controladores/rag.py — Controller do RAG (Ingestão + Busca)

Controller que expõe as operações do RAG pra API:
- Ingerir documentos (texto → chunks → embeddings → pgvector)
- Buscar boas práticas relevantes

No MVC, este controller faz a ponte entre a View (API)
e o Model (banco de dados + RAG).
"""

from sqlalchemy.ext.asyncio import AsyncSession

from modelos.esquemas import DocumentoEntrada, BuscaRAGSaida
from rag.ingestao import ingerir_documento
from rag.vetorial import buscar_similares


async def ingerir(
    sessao: AsyncSession,
    entrada: DocumentoEntrada,
) -> dict:
    """
    Ingere um documento no RAG (pipeline completo).

    Args:
        sessao: sessão do banco de dados
        entrada: texto + fonte do documento

    Returns:
        Dict com quantidade de chunks inseridos
    """
    # Executa o pipeline de ingestão
    total = await ingerir_documento(
        sessao=sessao,
        texto=entrada.texto,
        fonte=entrada.fonte,
    )

    return {"chunks_inseridos": total, "fonte": entrada.fonte}


async def buscar(
    sessao: AsyncSession,
    query: str,
    top_k: int = 5,
) -> BuscaRAGSaida:
    """
    Busca boas práticas relevantes no RAG.

    Args:
        sessao: sessão do banco
        query: texto de busca
        top_k: quantos resultados retornar

    Returns:
        BuscaRAGSaida com resultados e total
    """
    # Faz a busca vetorial no pgvector
    resultados = await buscar_similares(sessao, query, top_k)

    return BuscaRAGSaida(
        resultados=resultados,
        total=len(resultados),
    )
