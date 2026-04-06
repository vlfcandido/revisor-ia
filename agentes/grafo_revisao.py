"""
agentes/grafo_revisao.py — Grafo de Revisão (LangGraph)

Este é o módulo principal dos agentes — define o grafo de estados
que orquestra o fluxo completo de code review.

╔══════════════════════════════════════════════════════════════╗
║  LANGGRAPH — Grafo de Estados                               ║
║                                                              ║
║  LangGraph permite criar "workflows" como grafos:            ║
║  - Nós: funções que processam o estado                       ║
║  - Arestas: conexões entre nós (quem chama quem)             ║
║  - Estado: dados que fluem pelo grafo (TypedDict)            ║
║  - Arestas condicionais: decisões (if/else no grafo)         ║
║                                                              ║
║  Fluxo deste grafo:                                          ║
║                                                              ║
║  [INÍCIO]                                                    ║
║      ↓                                                       ║
║  [analisar_sintaxe] — detecta problemas no código            ║
║      ↓                                                       ║
║  [buscar_boas_praticas] — RAG: busca no pgvector             ║
║      ↓                                                       ║
║  [gerar_review] — LLM gera o review (estratégia A ou B)      ║
║      ↓                                                       ║
║  [avaliar_qualidade] — LLM-as-judge dá nota 0-10             ║
║      ↓                                                       ║
║  {nota >= 7?} ──sim──→ [FIM] ✅                              ║
║      │                                                       ║
║      └──não──→ [gerar_review] (tenta de novo, max 2x)        ║
╚══════════════════════════════════════════════════════════════╝

Conceitos importantes:
- StateGraph: classe principal do LangGraph, define o grafo
- START/END: nós especiais (início e fim do fluxo)
- add_node: adiciona um nó (função) ao grafo
- add_edge: conecta dois nós (A → B)
- add_conditional_edges: conecta com lógica (if/else)
- compile: "compila" o grafo pra execução
"""

import random

from langgraph.graph import StateGraph, START, END

from modelos.esquemas import EstadoRevisao
from modelos.feature_flags import flags
from agentes.prompts import obter_prompt_estrategia
from agentes.ferramentas import (
    analisar_sintaxe_codigo,
    gerar_review_com_llm,
    avaliar_review_com_llm,
)
from banco.conexao import fabrica_sessao
from rag.vetorial import buscar_similares


# ══════════════════════════════════════════════════════════════
# NÓS DO GRAFO
#
# Cada nó é uma função async que:
# 1. Recebe o estado atual (EstadoRevisao)
# 2. Faz seu trabalho
# 3. Retorna um dict com as chaves que modificou
#
# O LangGraph faz o merge automático: o estado atualizado
# é passado pro próximo nó.
# ══════════════════════════════════════════════════════════════


async def no_analisar_sintaxe(estado: EstadoRevisao) -> dict:
    """
    Nó 1: Analisa o código e detecta problemas de sintaxe.

    Recebe: codigo, linguagem
    Preenche: problemas_sintaxe
    """
    print("🔍 Analisando sintaxe do código...")

    # Usa o LLM pra detectar problemas
    problemas = await analisar_sintaxe_codigo(
        estado["codigo"],
        estado["linguagem"],
    )

    print(f"   Encontrados {len(problemas)} problemas")

    # Retorna apenas as chaves que este nó modifica
    return {"problemas_sintaxe": problemas}


async def no_buscar_boas_praticas(estado: EstadoRevisao) -> dict:
    """
    Nó 2: Busca boas práticas relevantes no RAG (pgvector).

    Recebe: codigo
    Preenche: boas_praticas

    Aqui é onde o RAG entra no fluxo:
    - Gera embedding do código
    - Busca os vetores mais próximos no pgvector
    - Retorna os textos das práticas mais relevantes
    """
    # Feature flag: RAG pode estar desligado pra testes rápidos
    if not flags.FF_RAG_ATIVO:
        print("📚 RAG desativado (FF_RAG_ATIVO=False) — pulando busca")
        return {"boas_praticas": []}

    print("📚 Buscando boas práticas no RAG...")

    # Abre uma sessão com o banco pra fazer a busca
    async with fabrica_sessao() as sessao:
        # Busca as práticas mais relevantes pro código
        praticas = await buscar_similares(
            sessao,
            query=estado["codigo"],  # usa o próprio código como query
            top_k=flags.FF_RAG_TOP_K,
        )

    print(f"   Encontradas {len(praticas)} práticas relevantes")

    return {"boas_praticas": praticas}


async def no_gerar_review(estado: EstadoRevisao) -> dict:
    """
    Nó 3: Gera o code review usando o LLM.

    Recebe: codigo, linguagem, problemas_sintaxe, boas_praticas
    Preenche: review, estrategia, tentativas

    Escolhe a estratégia (A ou B) e gera o review.
    Se está re-gerando (tentativa > 1), mantém a estratégia.
    """
    # Define a estratégia — se A/B test desligado, sempre usa A
    estrategia = estado.get("estrategia")
    if not estrategia:
        if flags.FF_TESTE_AB_ATIVO:
            estrategia = random.choice(["A", "B"])
        else:
            estrategia = "A"

    # Conta tentativas
    tentativas = estado.get("tentativas", 0) + 1

    print(f"✍️  Gerando review (estratégia {estrategia}, tentativa {tentativas})...")

    # Pega o template de prompt da estratégia
    template = obter_prompt_estrategia(estrategia)

    # Chama o LLM pra gerar o review
    review = await gerar_review_com_llm(
        codigo=estado["codigo"],
        linguagem=estado["linguagem"],
        problemas=estado.get("problemas_sintaxe", []),
        boas_praticas=estado.get("boas_praticas", []),
        prompt_template=template,
    )

    return {
        "review": review,
        "estrategia": estrategia,
        "tentativas": tentativas,
    }


async def no_avaliar_qualidade(estado: EstadoRevisao) -> dict:
    """
    Nó 4: Avalia a qualidade do review (LLM-as-Judge).

    Recebe: codigo, review
    Preenche: nota_qualidade, justificativa

    Usa um prompt separado pra avaliar o review gerado.
    É como ter um "supervisor" que dá nota pro trabalho do revisor.
    """
    print("⚖️  Avaliando qualidade do review...")

    # Chama o LLM avaliador (juiz)
    nota, justificativa = await avaliar_review_com_llm(
        codigo=estado["codigo"],
        review=estado["review"],
    )

    print(f"   Nota: {nota}/10 — {justificativa}")

    return {
        "nota_qualidade": nota,
        "justificativa": justificativa,
    }


# ══════════════════════════════════════════════════════════════
# ARESTA CONDICIONAL — decide se re-gera ou finaliza
#
# Depois de avaliar, verifica:
# - Se nota >= 7 → finaliza (END)
# - Se nota < 7 E tentativas < 3 → re-gera (volta pra gerar_review)
# - Se nota < 7 E tentativas >= 3 → finaliza mesmo assim (END)
# ══════════════════════════════════════════════════════════════


def decidir_proximo_passo(estado: EstadoRevisao) -> str:
    """
    Função de decisão: o review é bom o suficiente?

    Esta função é chamada pelo LangGraph após o nó 'avaliar_qualidade'.
    Retorna o NOME do próximo nó a executar.

    Returns:
        "fim" se o review é bom ou já tentou demais
        "gerar_review" se precisa re-gerar
    """
    nota = estado.get("nota_qualidade", 0)
    tentativas = estado.get("tentativas", 0)

    # Feature flag: se loop de re-geração desligado, sempre finaliza
    if not flags.FF_LOOP_REGENERACAO:
        print(f"{'✅' if nota >= 7 else '⚠️'} Nota {nota}/10 (re-geração desligada)")
        return "fim"

    # Review bom (nota >= 7) → finaliza
    if nota >= 7:
        print(f"✅ Review aprovado! Nota {nota}/10")
        return "fim"

    # Já tentou o máximo → finaliza mesmo com nota baixa
    if tentativas >= flags.FF_MAX_TENTATIVAS:
        print(f"⚠️  Nota {nota}/10 após {tentativas} tentativas. Finalizando.")
        return "fim"

    # Nota baixa, ainda tem tentativas → re-gera
    print(f"🔄 Nota {nota}/10 — re-gerando review (tentativa {tentativas + 1})...")
    return "gerar_review"


# ══════════════════════════════════════════════════════════════
# CONSTRUÇÃO DO GRAFO
#
# Aqui montamos o grafo juntando todos os nós e arestas.
# O LangGraph precisa saber:
# 1. Quais são os nós (add_node)
# 2. Como eles se conectam (add_edge)
# 3. Onde tem decisão (add_conditional_edges)
# 4. Onde começa e termina (START, END)
# ══════════════════════════════════════════════════════════════


def construir_grafo() -> StateGraph:
    """
    Constrói e compila o grafo de revisão.

    Returns:
        Grafo compilado, pronto pra executar com .ainvoke()

    Uso:
        grafo = construir_grafo()
        resultado = await grafo.ainvoke({
            "codigo": "def soma(a, b): return a+b",
            "linguagem": "python",
        })
    """
    # Cria o grafo com o tipo de estado
    # O LangGraph usa o EstadoRevisao pra saber quais dados fluem
    grafo = StateGraph(EstadoRevisao)

    # ── Adiciona os nós ──
    grafo.add_node("analisar_sintaxe", no_analisar_sintaxe)
    grafo.add_node("buscar_boas_praticas", no_buscar_boas_praticas)
    grafo.add_node("gerar_review", no_gerar_review)
    grafo.add_node("avaliar_qualidade", no_avaliar_qualidade)

    # ── Arestas fixas (A → B, sempre) ──
    grafo.add_edge(START, "analisar_sintaxe")                 # INÍCIO → análise
    grafo.add_edge("analisar_sintaxe", "buscar_boas_praticas")  # análise → RAG
    grafo.add_edge("buscar_boas_praticas", "gerar_review")      # RAG → gera review
    grafo.add_edge("gerar_review", "avaliar_qualidade")          # gera → avalia

    # ── Aresta condicional (decisão) ──
    # Após avaliar, decide: finaliza ou re-gera?
    grafo.add_conditional_edges(
        "avaliar_qualidade",     # nó de origem
        decidir_proximo_passo,   # função de decisão
        {
            "fim": END,            # se retorna "fim" → finaliza
            "gerar_review": "gerar_review",  # se retorna "gerar_review" → re-gera
        },
    )

    # Compila o grafo — transforma em algo executável
    return grafo.compile()
