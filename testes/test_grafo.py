"""
testes/test_grafo.py — Testes do Grafo LangGraph

Testa a construção e estrutura do grafo de revisão.
Estes testes verificam que o grafo foi montado corretamente
(nós, arestas, decisões) sem precisar executá-lo de verdade.

Conceitos:
- Testar a ESTRUTURA do grafo (não a execução)
- Evita dependência de Ollama/banco nos testes
- Verifica que todos os nós estão registrados
- Verifica a lógica de decisão
"""

from modelos.esquemas import EstadoRevisao
from agentes.grafo_revisao import construir_grafo, decidir_proximo_passo


# ══════════════════════════════════════════════════════════════
# TESTES DA CONSTRUÇÃO DO GRAFO
# ══════════════════════════════════════════════════════════════

def test_grafo_compila():
    """Verifica que o grafo compila sem erros."""
    # Se construir_grafo() não lançar exceção, o grafo é válido
    grafo = construir_grafo()

    # O grafo compilado deve existir
    assert grafo is not None


# ══════════════════════════════════════════════════════════════
# TESTES DA FUNÇÃO DE DECISÃO
# ══════════════════════════════════════════════════════════════

def test_decisao_nota_alta_finaliza():
    """Nota >= 7 deve finalizar (retornar 'fim')."""
    estado: EstadoRevisao = {
        "codigo": "def f(): pass",
        "linguagem": "python",
        "nota_qualidade": 8.5,
        "tentativas": 1,
    }

    resultado = decidir_proximo_passo(estado)

    assert resultado == "fim"


def test_decisao_nota_baixa_regera():
    """Nota < 7 com tentativas sobrando deve re-gerar."""
    estado: EstadoRevisao = {
        "codigo": "def f(): pass",
        "linguagem": "python",
        "nota_qualidade": 5.0,
        "tentativas": 1,  # ainda tem tentativas (< 3)
    }

    resultado = decidir_proximo_passo(estado)

    assert resultado == "gerar_review"


def test_decisao_nota_baixa_max_tentativas_finaliza():
    """Nota < 7 mas com 3+ tentativas deve finalizar."""
    estado: EstadoRevisao = {
        "codigo": "def f(): pass",
        "linguagem": "python",
        "nota_qualidade": 4.0,
        "tentativas": 3,  # já tentou o máximo
    }

    resultado = decidir_proximo_passo(estado)

    assert resultado == "fim"


def test_decisao_nota_exata_7_finaliza():
    """Nota exatamente 7 deve finalizar (>= 7)."""
    estado: EstadoRevisao = {
        "codigo": "def f(): pass",
        "linguagem": "python",
        "nota_qualidade": 7.0,
        "tentativas": 1,
    }

    resultado = decidir_proximo_passo(estado)

    assert resultado == "fim"


def test_decisao_nota_6_9_regera():
    """Nota 6.9 (quase 7) deve re-gerar."""
    estado: EstadoRevisao = {
        "codigo": "def f(): pass",
        "linguagem": "python",
        "nota_qualidade": 6.9,
        "tentativas": 1,
    }

    resultado = decidir_proximo_passo(estado)

    assert resultado == "gerar_review"
