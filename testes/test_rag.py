"""
testes/test_rag.py — Testes do RAG (Chunking)

Testa a funcionalidade de chunking (divisão de texto).
Esses testes são "unitários" — testam funções isoladas,
sem depender do banco ou do Ollama.

Conceitos:
- Testes unitários: testam uma função isolada
- Não precisam de infraestrutura (banco, LLM)
- São rápidos e confiáveis
- Rodam com: pytest testes/test_rag.py
"""

from rag.chunking import dividir_em_chunks, dividir_por_paragrafos


# ══════════════════════════════════════════════════════════════
# TESTES DO dividir_em_chunks
# ══════════════════════════════════════════════════════════════

def test_texto_pequeno_retorna_unico_chunk():
    """Texto menor que o tamanho do chunk retorna 1 chunk."""
    texto = "Texto curto."

    chunks = dividir_em_chunks(texto, tamanho=500)

    # Deve retornar exatamente 1 chunk
    assert len(chunks) == 1
    assert chunks[0] == "Texto curto."


def test_texto_grande_divide_em_multiplos_chunks():
    """Texto grande é dividido em vários chunks."""
    # Cria um texto de 1000 caracteres
    texto = "A" * 1000

    chunks = dividir_em_chunks(texto, tamanho=300, sobreposicao=50)

    # Deve ter mais de 1 chunk
    assert len(chunks) > 1

    # Cada chunk deve ter no máximo 'tamanho' caracteres
    for chunk in chunks:
        assert len(chunk) <= 300


def test_sobreposicao_funciona():
    """Testa que a sobreposição entre chunks existe."""
    texto = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Tamanho 10, sobreposição 5
    chunks = dividir_em_chunks(texto, tamanho=10, sobreposicao=5)

    # O final de um chunk deve aparecer no início do próximo
    # (overlap de 5 caracteres)
    assert len(chunks) >= 2

    # Verifica que há overlap: os últimos 5 chars do chunk 1
    # devem ser os primeiros 5 chars do chunk 2
    final_chunk1 = chunks[0][-5:]
    inicio_chunk2 = chunks[1][:5]
    assert final_chunk1 == inicio_chunk2


def test_texto_vazio_retorna_lista_vazia():
    """Texto vazio ou só espaços retorna lista vazia."""
    chunks = dividir_em_chunks("   ", tamanho=500)

    # Texto em branco é filtrado pelo strip()
    assert len(chunks) == 0


# ══════════════════════════════════════════════════════════════
# TESTES DO dividir_por_paragrafos
# ══════════════════════════════════════════════════════════════

def test_paragrafos_simples():
    """Divide texto com parágrafos claros."""
    texto = "Parágrafo 1.\n\nParágrafo 2.\n\nParágrafo 3."

    chunks = dividir_por_paragrafos(texto, tamanho_max=500)

    # Texto pequeno com 3 parágrafos → deve caber em 1 chunk
    assert len(chunks) == 1
    assert "Parágrafo 1" in chunks[0]


def test_paragrafos_grandes_divide():
    """Parágrafos que excedem o tamanho máximo são separados."""
    # Cria parágrafos de 200 chars cada
    paragrafos = [f"Parágrafo {'X' * 190}" for _ in range(5)]
    texto = "\n\n".join(paragrafos)

    chunks = dividir_por_paragrafos(texto, tamanho_max=300)

    # 5 parágrafos de 200 chars, max 300 → deve ter vários chunks
    assert len(chunks) > 1
