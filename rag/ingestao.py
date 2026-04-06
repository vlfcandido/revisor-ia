"""
rag/ingestao.py — Pipeline de Ingestão de Documentos

Orquestra o pipeline completo de ingestão RAG:
    Texto → Chunking → Embeddings → pgvector

Este módulo junta os outros módulos do RAG:
- chunking.py: divide texto em pedaços
- embeddings.py: converte chunks em vetores
- vetorial.py: salva vetores no banco

Também contém a função de seed (popular a base com dados iniciais).

Pipeline completo:
    1. Recebe texto bruto
    2. Divide em chunks (pedaços menores)
    3. Gera embedding de cada chunk (vetor numérico)
    4. Salva no pgvector (banco vetorial)
"""

import json
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from rag.chunking import dividir_em_chunks
from rag.embeddings import gerar_embedding
from rag.vetorial import inserir_boa_pratica, contar_registros


async def ingerir_documento(
    sessao: AsyncSession,
    texto: str,
    fonte: str = "manual",
    tamanho_chunk: int = 500,
    sobreposicao: int = 50,
) -> int:
    """
    Pipeline completo de ingestão: texto → chunks → embeddings → banco.

    Args:
        sessao: sessão do banco de dados
        texto: texto completo a ser ingerido
        fonte: origem do documento (pra referência)
        tamanho_chunk: tamanho de cada chunk em caracteres
        sobreposicao: overlap entre chunks

    Returns:
        Quantidade de chunks inseridos

    Exemplo:
        >>> n = await ingerir_documento(sessao, "texto longo...", fonte="clean_code")
        >>> print(f"Inseridos {n} chunks")
    """
    # Passo 1: Divide o texto em chunks
    chunks = dividir_em_chunks(texto, tamanho_chunk, sobreposicao)
    print(f"  📄 Dividido em {len(chunks)} chunks")

    # Passo 2 e 3: Pra cada chunk, gera embedding e salva
    for i, chunk in enumerate(chunks):
        # Gera o vetor (embedding) do chunk
        embedding = await gerar_embedding(chunk)

        # Salva no banco (pgvector)
        await inserir_boa_pratica(sessao, chunk, fonte, embedding)

        print(f"  ✅ Chunk {i + 1}/{len(chunks)} inserido")

    return len(chunks)


async def popular_base_inicial(sessao: AsyncSession) -> int:
    """
    Popula o banco com boas práticas a partir do arquivo seeds.

    Lê o arquivo seeds/boas_praticas.json e insere cada prática
    no banco com seu embedding. Só insere se a base estiver vazia.

    Returns:
        Quantidade de práticas inseridas (0 se base já populada)
    """
    # Verifica se já tem dados no banco
    total_existente = await contar_registros(sessao)
    if total_existente > 0:
        print(f"⚠️  Base já tem {total_existente} registros. Pulando seed.")
        return 0

    # Caminho do arquivo de seeds
    caminho_seeds = Path(__file__).parent.parent / "seeds" / "boas_praticas.json"

    if not caminho_seeds.exists():
        print("❌ Arquivo seeds/boas_praticas.json não encontrado!")
        return 0

    # Lê o JSON com as boas práticas
    with open(caminho_seeds, "r", encoding="utf-8") as f:
        praticas = json.load(f)

    print(f"🌱 Inserindo {len(praticas)} boas práticas...")

    # Insere cada prática
    total_inserido = 0
    for pratica in praticas:
        texto = pratica["texto"]
        fonte = pratica.get("fonte", "seed")

        # Gera embedding e insere
        embedding = await gerar_embedding(texto)
        await inserir_boa_pratica(sessao, texto, fonte, embedding)

        total_inserido += 1
        print(f"  ✅ {total_inserido}/{len(praticas)}: {texto[:60]}...")

    print(f"🎉 Base populada com {total_inserido} boas práticas!")
    return total_inserido
