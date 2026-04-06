"""
testes/test_esquemas.py — Testes dos Schemas Pydantic

Testa se os schemas validam dados corretamente:
- Aceita dados válidos
- Rejeita dados inválidos (com mensagens claras)

Conceitos de teste:
- pytest: framework de testes do Python (alternativa ao unittest)
- assert: verifica se algo é verdadeiro (falha se não for)
- pytest.raises: verifica se uma exceção é lançada
- Cada função test_* é um teste independente
"""

import pytest
from pydantic import ValidationError

from modelos.esquemas import (
    CodigoEntrada,
    ReviewSaida,
    ResultadoAB,
    DocumentoEntrada,
)


# ══════════════════════════════════════════════════════════════
# TESTES DO CodigoEntrada
# ══════════════════════════════════════════════════════════════

def test_codigo_entrada_valido():
    """Testa que dados válidos são aceitos sem erro."""
    # Cria uma instância com dados válidos
    entrada = CodigoEntrada(
        codigo="def soma(a, b): return a + b",
        linguagem="python",
    )

    # Verifica que os dados foram armazenados corretamente
    assert entrada.codigo == "def soma(a, b): return a + b"
    assert entrada.linguagem == "python"
    assert entrada.estrategia is None  # padrão é None


def test_codigo_entrada_com_estrategia():
    """Testa que a estratégia opcional funciona."""
    entrada = CodigoEntrada(
        codigo="console.log('hello')",
        linguagem="javascript",
        estrategia="A",
    )

    assert entrada.estrategia == "A"


def test_codigo_entrada_linguagem_padrao():
    """Testa que a linguagem padrão é 'python'."""
    entrada = CodigoEntrada(codigo="print('hello')")

    # Se não passar linguagem, deve ser "python"
    assert entrada.linguagem == "python"


def test_codigo_entrada_vazio_rejeita():
    """Testa que código vazio é rejeitado pelo Pydantic."""
    # pytest.raises verifica que a exceção é lançada
    with pytest.raises(ValidationError):
        CodigoEntrada(codigo="")  # min_length=1, vazio não passa


# ══════════════════════════════════════════════════════════════
# TESTES DO ReviewSaida
# ══════════════════════════════════════════════════════════════

def test_review_saida_valido():
    """Testa que um review válido é aceito."""
    review = ReviewSaida(
        review="O código está bem escrito, mas...",
        nota_qualidade=7.5,
        justificativa="Review claro e completo.",
        estrategia="A",
        tentativas=1,
    )

    assert review.nota_qualidade == 7.5
    assert review.estrategia == "A"
    assert review.problemas_sintaxe == []  # lista vazia por padrão


def test_review_saida_nota_invalida():
    """Testa que nota fora do range (0-10) é rejeitada."""
    # Nota 11 ultrapassa o máximo (le=10)
    with pytest.raises(ValidationError):
        ReviewSaida(
            review="teste",
            nota_qualidade=11,  # inválido! máximo é 10
            justificativa="teste",
            estrategia="A",
            tentativas=1,
        )


def test_review_saida_nota_negativa():
    """Testa que nota negativa é rejeitada."""
    with pytest.raises(ValidationError):
        ReviewSaida(
            review="teste",
            nota_qualidade=-1,  # inválido! mínimo é 0
            justificativa="teste",
            estrategia="B",
            tentativas=1,
        )


# ══════════════════════════════════════════════════════════════
# TESTES DO ResultadoAB
# ══════════════════════════════════════════════════════════════

def test_resultado_ab_valido():
    """Testa que resultado do teste A/B é criado corretamente."""
    resultado = ResultadoAB(
        media_a=7.5,
        media_b=6.8,
        vencedor="A",
        rodadas=4,
        detalhes=[
            {"rodada": 1, "estrategia": "A", "nota": 7.5},
            {"rodada": 1, "estrategia": "B", "nota": 6.8},
        ],
    )

    assert resultado.vencedor == "A"
    assert resultado.media_a > resultado.media_b
    assert len(resultado.detalhes) == 2


# ══════════════════════════════════════════════════════════════
# TESTES DO DocumentoEntrada
# ══════════════════════════════════════════════════════════════

def test_documento_entrada_valido():
    """Testa ingestão de documento com dados válidos."""
    doc = DocumentoEntrada(
        texto="Use nomes descritivos para variáveis.",
        fonte="clean_code",
    )

    assert doc.fonte == "clean_code"


def test_documento_entrada_fonte_padrao():
    """Testa que a fonte padrão é 'manual'."""
    doc = DocumentoEntrada(texto="Algum texto sobre boas práticas.")

    assert doc.fonte == "manual"
