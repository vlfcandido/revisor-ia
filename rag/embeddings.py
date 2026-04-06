"""
rag/embeddings.py — Geração de Embeddings com Ollama

Embeddings são representações numéricas de texto — cada texto
é convertido num vetor de 768 números (floats). Textos com
significados parecidos terão vetores próximos no espaço vetorial.

Modelo: nomic-embed-text (768 dimensões)
- Roda local via Ollama (zero custo)
- Bom pra textos em inglês e português
- Rápido pra gerar embeddings de chunks

Pipeline RAG:
    Texto → [chunking] → Chunks → [EMBEDDINGS] → Vetores → [pgvector]
                                   ↑ você está aqui

Conceitos:
- Embedding: vetor numérico que "captura o significado" do texto
- Distância de cosseno: mede quão parecidos são dois vetores (0=iguais, 2=opostos)
- O pgvector usa essa distância pra encontrar os textos mais relevantes
"""

import httpx

from modelos.configuracao import configuracao


async def gerar_embedding(texto: str) -> list[float]:
    """
    Gera o embedding (vetor numérico) de um texto usando Ollama.

    Faz uma chamada HTTP pro Ollama local, que roda o modelo
    nomic-embed-text e retorna um vetor de 768 floats.

    Args:
        texto: texto pra vetorizar (chunk ou query de busca)

    Returns:
        Lista de 768 floats representando o texto

    Exemplo:
        >>> vetor = await gerar_embedding("Use nomes descritivos")
        >>> len(vetor)  # → 768
        >>> type(vetor[0])  # → float
    """
    # Monta a URL da API de embeddings do Ollama
    url = f"{configuracao.OLLAMA_BASE_URL}/api/embed"

    # Payload da request — modelo e texto
    payload = {
        "model": configuracao.OLLAMA_MODELO_EMBEDDING,
        "input": texto,
    }

    # Faz a request HTTP pro Ollama
    # httpx é como o requests, mas com suporte async
    async with httpx.AsyncClient() as cliente:
        resposta = await cliente.post(
            url,
            json=payload,
            timeout=30.0,  # timeout de 30s (embedding é rápido)
        )

    # Verifica se deu certo
    resposta.raise_for_status()

    # O Ollama retorna: {"embeddings": [[0.012, -0.034, ...]]}
    dados = resposta.json()
    return dados["embeddings"][0]


async def gerar_embeddings_lote(textos: list[str]) -> list[list[float]]:
    """
    Gera embeddings pra vários textos de uma vez.

    Útil quando precisa vetorizar muitos chunks de uma vez
    (ex: na ingestão de um documento grande).

    Args:
        textos: lista de textos pra vetorizar

    Returns:
        Lista de vetores (um pra cada texto)

    Nota: O Ollama não suporta batch nativo pra embeddings,
    então fazemos um loop sequencial. Pra produção, seria
    melhor usar asyncio.gather() pra paralelizar.
    """
    embeddings = []

    for texto in textos:
        # Gera embedding de cada texto individualmente
        embedding = await gerar_embedding(texto)
        embeddings.append(embedding)

    return embeddings
