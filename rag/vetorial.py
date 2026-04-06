"""
rag/vetorial.py — Operações Vetoriais no pgvector

Este módulo faz o CRUD (Create, Read) no pgvector —
insere vetores e busca os mais similares.

O pgvector é uma extensão do PostgreSQL que adiciona:
- Tipo de coluna 'vector' pra armazenar embeddings
- Operador '<->' pra calcular distância entre vetores
- Índice HNSW ou IVFFlat pra busca rápida

Busca por similaridade:
    Query: "como nomear variáveis?"
    → Gera embedding da query
    → Busca no pgvector os vetores mais próximos
    → Retorna os textos correspondentes

    SELECT texto
    FROM boas_praticas
    ORDER BY embedding <-> :query_embedding  -- distância de cosseno
    LIMIT 5;

Pipeline RAG:
    Texto → Chunks → Embeddings → [PGVECTOR] → Busca → Contexto
                                  ↑ você está aqui
"""

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from modelos.banco import BoaPratica
from rag.embeddings import gerar_embedding


async def inserir_boa_pratica(
    sessao: AsyncSession,
    texto: str,
    fonte: str,
    embedding: list[float],
) -> BoaPratica:
    """
    Insere uma boa prática com seu embedding no banco.

    Args:
        sessao: sessão do SQLAlchemy (transação com o banco)
        texto: texto da boa prática
        fonte: origem do texto (ex: "clean_code")
        embedding: vetor de 768 floats

    Returns:
        Objeto BoaPratica criado (com id preenchido)
    """
    # Cria o objeto ORM
    boa_pratica = BoaPratica(
        texto=texto,
        fonte=fonte,
        embedding=embedding,
    )

    # Adiciona à sessão (staging area do SQLAlchemy)
    sessao.add(boa_pratica)

    # Faz o commit (salva de verdade no banco)
    await sessao.commit()

    # Atualiza o objeto com dados gerados pelo banco (id, criado_em)
    await sessao.refresh(boa_pratica)

    return boa_pratica


async def buscar_similares(
    sessao: AsyncSession,
    query: str,
    top_k: int = 5,
) -> list[str]:
    """
    Busca as boas práticas mais similares a uma query.

    Este é o coração da busca RAG:
    1. Gera o embedding da query
    2. Usa o pgvector pra encontrar os vetores mais próximos
    3. Retorna os textos correspondentes

    Args:
        sessao: sessão do banco
        query: texto de busca (ex: "como nomear variáveis?")
        top_k: quantos resultados retornar (padrão: 5)

    Returns:
        Lista dos textos mais relevantes

    O operador '<->' calcula a distância de cosseno:
    - 0 = idênticos
    - 1 = ortogonais (sem relação)
    - 2 = opostos
    Quanto MENOR a distância, MAIS similar.
    """
    # Passo 1: Gera o embedding da query de busca
    query_embedding = await gerar_embedding(query)

    # Passo 2: Busca no pgvector usando distância de cosseno
    # O operador '<->' é do pgvector e calcula cosine distance
    # ORDER BY distância ASC = mais similar primeiro
    resultado = await sessao.execute(
        select(BoaPratica.texto)
        .order_by(
            BoaPratica.embedding.cosine_distance(query_embedding)
        )
        .limit(top_k)
    )

    # Passo 3: Extrai os textos dos resultados
    # scalars() retorna só a coluna 'texto' (não o objeto inteiro)
    textos = resultado.scalars().all()

    return list(textos)


async def contar_registros(sessao: AsyncSession) -> int:
    """
    Conta quantas boas práticas existem no banco.

    Útil pra saber se a base já foi populada (seed).
    """
    resultado = await sessao.execute(
        select(BoaPratica.id)
    )
    return len(resultado.scalars().all())
