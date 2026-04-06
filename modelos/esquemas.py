"""
modelos/esquemas.py — Schemas Pydantic (Validação de Dados)

Define os "contratos" de dados do projeto usando Pydantic v2.
Cada schema valida automaticamente os dados de entrada/saída.

Conceitos importantes:
- BaseModel: classe base do Pydantic, valida dados no __init__
- Field: adiciona validações extras (min, max, descrição)
- TypedDict: usado pelo LangGraph pra definir estado do grafo
- Tipagem Python: str, list, float, Optional — Pydantic usa pra validar

No MVC, os schemas fazem parte do Model — definem a
"forma" dos dados que trafegam entre as camadas.
"""

from typing import Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════
# SCHEMAS DE ENTRADA/SAÍDA DA API
# ══════════════════════════════════════════════════════════════

class CodigoEntrada(BaseModel):
    """
    Schema de entrada: código que o usuário quer revisar.

    Exemplo de uso na API:
        POST /revisar
        {"codigo": "def soma(a, b): return a+b", "linguagem": "python"}
    """
    # O código fonte a ser revisado
    codigo: str = Field(
        ...,  # ... = obrigatório (não tem valor padrão)
        min_length=1,
        description="Código fonte a ser revisado",
    )

    # Linguagem do código (python, javascript, etc.)
    linguagem: str = Field(
        default="python",
        description="Linguagem de programação do código",
    )

    # Estratégia de review: "A" (detalhado) ou "B" (conciso)
    # Se None, o sistema escolhe aleatoriamente (útil pro teste A/B)
    estrategia: Optional[str] = Field(
        default=None,
        description="Estratégia de prompt: 'A' (detalhado) ou 'B' (conciso). None = aleatório.",
    )


class ReviewSaida(BaseModel):
    """
    Schema de saída: resultado da revisão de código.

    Contém o review gerado, a nota de qualidade,
    e metadados sobre o processo.
    """
    # Texto do code review gerado pelo agente
    review: str = Field(description="Texto do code review gerado")

    # Nota de qualidade do review (0 a 10), dada pelo LLM-as-judge
    nota_qualidade: float = Field(
        ge=0,    # ge = greater or equal (mínimo 0)
        le=10,   # le = less or equal (máximo 10)
        description="Nota de qualidade do review (0-10)",
    )

    # Justificativa da nota (por que o juiz deu essa nota)
    justificativa: str = Field(description="Justificativa da nota pelo avaliador")

    # Qual estratégia de prompt foi usada
    estrategia: str = Field(description="Estratégia usada: 'A' ou 'B'")

    # Quantas vezes o review foi re-gerado (se nota < 7)
    tentativas: int = Field(description="Número de tentativas de geração")

    # Problemas de sintaxe encontrados
    problemas_sintaxe: list[str] = Field(
        default_factory=list,
        description="Lista de problemas de sintaxe encontrados",
    )

    # Boas práticas relevantes encontradas no RAG
    boas_praticas: list[str] = Field(
        default_factory=list,
        description="Boas práticas relevantes encontradas no RAG",
    )


# ══════════════════════════════════════════════════════════════
# ESTADO DO LANGGRAPH
#
# TypedDict é usado pelo LangGraph pra definir o estado que
# flui entre os nós do grafo. É como um "formulário" que cada
# nó pode ler e preencher.
#
# Por que TypedDict e não BaseModel?
# → LangGraph usa TypedDict internamente pra controle de estado.
#   É uma convenção do framework.
# ══════════════════════════════════════════════════════════════

class EstadoRevisao(TypedDict, total=False):
    """
    Estado que flui pelo grafo de revisão (LangGraph).

    Cada nó do grafo recebe este estado, faz seu trabalho,
    e retorna as chaves que modificou. O LangGraph faz o
    merge automático.

    total=False significa que nem todas as chaves precisam
    existir desde o início — cada nó preenche as suas.
    """
    # --- Entrada (preenchido pelo usuário) ---
    codigo: str                     # código fonte a revisar
    linguagem: str                  # linguagem de programação

    # --- Preenchido pelo nó 'analisar_sintaxe' ---
    problemas_sintaxe: list[str]    # problemas encontrados

    # --- Preenchido pelo nó 'buscar_boas_praticas' ---
    boas_praticas: list[str]        # resultados do RAG

    # --- Preenchido pelo nó 'gerar_review' ---
    review: str                     # texto do review
    estrategia: str                 # "A" ou "B"

    # --- Preenchido pelo nó 'avaliar_qualidade' ---
    nota_qualidade: float           # 0-10 (LLM-as-judge)
    justificativa: str              # por que essa nota

    # --- Controle interno ---
    tentativas: int                 # quantas vezes re-gerou


# ══════════════════════════════════════════════════════════════
# SCHEMAS DO TESTE A/B
# ══════════════════════════════════════════════════════════════

class ResultadoAB(BaseModel):
    """
    Resultado de um teste A/B entre duas estratégias de prompt.

    Compara a média de notas entre a estratégia A (detalhada)
    e a estratégia B (concisa).
    """
    # Média de notas da estratégia A
    media_a: float = Field(description="Média de notas da estratégia A")

    # Média de notas da estratégia B
    media_b: float = Field(description="Média de notas da estratégia B")

    # Qual estratégia venceu
    vencedor: str = Field(description="Estratégia vencedora: 'A' ou 'B'")

    # Quantas rodadas foram executadas
    rodadas: int = Field(description="Número de rodadas do teste")

    # Detalhes de cada rodada
    detalhes: list[dict] = Field(
        default_factory=list,
        description="Nota de cada rodada [{estrategia, nota}]",
    )


# ══════════════════════════════════════════════════════════════
# SCHEMAS DO RAG
# ══════════════════════════════════════════════════════════════

class DocumentoEntrada(BaseModel):
    """Schema pra ingerir um documento no RAG."""
    # Texto do documento a ser processado
    texto: str = Field(
        ...,
        min_length=1,
        description="Texto do documento a ser ingerido no RAG",
    )

    # Fonte/origem do documento (pra referência)
    fonte: str = Field(
        default="manual",
        description="Origem do documento (ex: 'clean_code', 'solid')",
    )


class BuscaRAGSaida(BaseModel):
    """Schema de saída de uma busca no RAG."""
    # Textos mais relevantes encontrados
    resultados: list[str] = Field(description="Chunks mais relevantes")

    # Quantidade de resultados
    total: int = Field(description="Quantidade de resultados retornados")
