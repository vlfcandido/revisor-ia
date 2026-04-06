"""
rag/chunking.py — Divisão de Texto em Chunks

O chunking é o primeiro passo do pipeline RAG:
Texto grande → pedaços menores (chunks) → cada chunk vira um embedding

Por que dividir em chunks?
- Modelos de embedding têm limite de tokens (~8000 pra nomic-embed-text)
- Chunks menores são mais precisos na busca (menos "ruído")
- Overlap (sobreposição) evita perder contexto nas bordas

Conceitos:
- Chunk: pedaço de texto com tamanho controlado
- Overlap: quantos caracteres se repetem entre chunks consecutivos
- Isso garante que uma frase cortada no meio ainda aparece inteira
  em pelo menos um chunk

Exemplo com tamanho=10, overlap=3:
    Texto: "ABCDEFGHIJKLMNOP"
    Chunk 1: "ABCDEFGHIJ"
    Chunk 2: "HIJKLMNOP"   ← "HIJ" repete (overlap de 3)
"""


def dividir_em_chunks(
    texto: str,
    tamanho: int = 500,
    sobreposicao: int = 50,
) -> list[str]:
    """
    Divide um texto em chunks menores com sobreposição.

    Args:
        texto: texto completo a ser dividido
        tamanho: tamanho máximo de cada chunk (em caracteres)
        sobreposicao: quantos caracteres se repetem entre chunks

    Returns:
        Lista de chunks (strings menores)

    Exemplo:
        >>> chunks = dividir_em_chunks("texto longo...", tamanho=100, sobreposicao=20)
        >>> len(chunks)  # → depende do tamanho do texto
    """
    # Se o texto é vazio ou só espaços, retorna lista vazia
    if not texto.strip():
        return []

    # Se o texto é menor que um chunk, retorna ele inteiro
    if len(texto) <= tamanho:
        return [texto.strip()]

    chunks = []
    inicio = 0

    while inicio < len(texto):
        # Pega um pedaço do texto do tamanho definido
        fim = inicio + tamanho
        chunk = texto[inicio:fim]

        # Só adiciona se o chunk não estiver vazio
        if chunk.strip():
            chunks.append(chunk.strip())

        # Avança a janela, mas volta 'sobreposicao' caracteres
        # Isso cria o overlap entre chunks consecutivos
        inicio += tamanho - sobreposicao

    return chunks


def dividir_por_paragrafos(texto: str, tamanho_max: int = 500) -> list[str]:
    """
    Divide texto respeitando parágrafos (quebras de linha duplas).

    Alternativa ao chunking por caracteres — tenta manter
    parágrafos inteiros, juntando parágrafos pequenos até
    atingir o tamanho máximo.

    Args:
        texto: texto com parágrafos separados por \\n\\n
        tamanho_max: tamanho máximo de cada chunk

    Returns:
        Lista de chunks respeitando parágrafos
    """
    # Separa o texto em parágrafos
    paragrafos = texto.split("\n\n")

    chunks = []
    chunk_atual = ""

    for paragrafo in paragrafos:
        paragrafo = paragrafo.strip()
        if not paragrafo:
            continue

        # Se adicionar este parágrafo ultrapassa o limite,
        # salva o chunk atual e começa um novo
        if len(chunk_atual) + len(paragrafo) > tamanho_max and chunk_atual:
            chunks.append(chunk_atual.strip())
            chunk_atual = ""

        # Adiciona o parágrafo ao chunk atual
        if chunk_atual:
            chunk_atual += "\n\n" + paragrafo
        else:
            chunk_atual = paragrafo

    # Não esquece do último chunk
    if chunk_atual.strip():
        chunks.append(chunk_atual.strip())

    return chunks
