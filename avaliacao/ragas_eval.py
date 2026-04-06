"""
avaliacao/ragas_eval.py — Avaliação de RAG com Ragas

╔══════════════════════════════════════════════════════════════╗
║  O QUE É RAGAS?                                             ║
║                                                              ║
║  Ragas (RAG Assessment) é um framework que avalia a          ║
║  QUALIDADE do seu pipeline RAG com métricas padronizadas.    ║
║                                                              ║
║  Sem Ragas, você não sabe se:                                ║
║  - O retriever tá trazendo chunks relevantes                 ║
║  - O LLM tá usando o contexto pra responder                  ║
║  - A resposta é fiel ao contexto (ou tá inventando)          ║
║                                                              ║
║  É como ter testes unitários, mas pra qualidade de RAG.      ║
╚══════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════
COMO FUNCIONARIA NA MÃO (sem Ragas):

1. FAITHFULNESS (Fidelidade):
   - Você pegaria a resposta do LLM e o contexto do RAG
   - Mandaria pro LLM: "A resposta está baseada no contexto? 0 a 1"
   - Faria isso manualmente pra cada resposta
   → Ragas automatiza isso com um pipeline de NLI (Natural Language Inference)

2. ANSWER RELEVANCY (Relevância da Resposta):
   - Pegaria a pergunta original e a resposta
   - Mandaria pro LLM: "A resposta é relevante pra pergunta? 0 a 1"
   - Compararia embeddings da pergunta com embeddings da resposta
   → Ragas gera perguntas reversa a partir da resposta e compara com a original

3. CONTEXT PRECISION (Precisão do Contexto):
   - Pegaria os chunks retornados pelo RAG
   - Verificaria manualmente: "Esses chunks são úteis pra responder?"
   - Calcularia: chunks_úteis / total_chunks
   → Ragas usa LLM pra classificar cada chunk como útil ou não

4. CONTEXT RECALL (Cobertura do Contexto):
   - Compararia o que o RAG retornou com o que DEVERIA retornar
   - Precisaria de um "ground truth" (resposta esperada)
   - Calcularia: informações_encontradas / informações_esperadas
   → Ragas compara cada frase do ground truth com os chunks

NA PRÁTICA:
   Sem Ragas = você faz 4 prompts manuais + cálculos + parsing
   Com Ragas = uma chamada e ele faz tudo automatizado
═══════════════════════════════════════════════════════════════

Métricas do Ragas:
┌─────────────────────┬────────────────────────────────────────┐
│ Métrica              │ O que mede                             │
├─────────────────────┼────────────────────────────────────────┤
│ faithfulness         │ A resposta é fiel ao contexto?         │
│ answer_relevancy     │ A resposta é relevante pra pergunta?   │
│ context_precision    │ Os chunks retornados são úteis?        │
│ context_recall       │ O RAG trouxe toda informação necessária│
└─────────────────────┴────────────────────────────────────────┘

Cada métrica vai de 0.0 (péssimo) a 1.0 (perfeito).
"""

from dataclasses import dataclass

from modelos.feature_flags import flags


# ══════════════════════════════════════════════════════════════
# RESULTADO DA AVALIAÇÃO RAGAS
#
# Dataclass simples pra agrupar as métricas.
# Poderíamos usar Pydantic, mas dataclass é mais leve
# e suficiente pra dados internos.
# ══════════════════════════════════════════════════════════════

@dataclass
class ResultadoRagas:
    """Resultado da avaliação Ragas com todas as métricas."""
    faithfulness: float         # 0-1: resposta é fiel ao contexto?
    answer_relevancy: float     # 0-1: resposta é relevante?
    context_precision: float    # 0-1: chunks retornados são úteis?
    context_recall: float       # 0-1: RAG trouxe tudo que precisava?

    @property
    def media(self) -> float:
        """Média geral das 4 métricas."""
        return (
            self.faithfulness
            + self.answer_relevancy
            + self.context_precision
            + self.context_recall
        ) / 4


async def avaliar_com_ragas(
    pergunta: str,
    resposta: str,
    contextos: list[str],
    ground_truth: str = "",
) -> ResultadoRagas | None:
    """
    Avalia a qualidade do RAG usando métricas Ragas.

    Args:
        pergunta: a query original do usuário (ex: "revise este código")
        resposta: a resposta gerada pelo LLM (ex: o code review)
        contextos: os chunks retornados pelo RAG (boas práticas)
        ground_truth: resposta esperada (opcional, pra context_recall)

    Returns:
        ResultadoRagas com as 4 métricas, ou None se Ragas desativado

    ─────────────────────────────────────────────────────
    COMO FUNCIONA POR BAIXO:

    O Ragas faz isso internamente (simplificado):

    1. Faithfulness:
       - Extrai "afirmações" da resposta (usa LLM)
       - Pra cada afirmação, verifica se está no contexto
       - Score = afirmações_suportadas / total_afirmações

    2. Answer Relevancy:
       - Gera N perguntas a partir da resposta (engenharia reversa)
       - Calcula similaridade (embedding) entre perguntas geradas e original
       - Score = média das similaridades

    3. Context Precision:
       - Pra cada chunk, pergunta ao LLM: "esse chunk ajuda a responder?"
       - Score = chunks_úteis / total_chunks (com peso pela posição)

    4. Context Recall:
       - Divide o ground_truth em frases
       - Pra cada frase, verifica se algum chunk a suporta
       - Score = frases_suportadas / total_frases
    ─────────────────────────────────────────────────────

    Exemplo:
        >>> resultado = await avaliar_com_ragas(
        ...     pergunta="revise def f(x): return x+1",
        ...     resposta="O código usa nome pouco descritivo...",
        ...     contextos=["Use nomes descritivos pra variáveis..."],
        ... )
        >>> resultado.faithfulness  # → 0.85
        >>> resultado.media         # → 0.78
    """
    # Verifica feature flag
    if not flags.FF_RAGAS_ATIVO:
        print("ℹ️  Ragas desativado (FF_RAGAS_ATIVO=False)")
        return None

    try:
        # ── Importações do Ragas ──
        # Importamos aqui dentro pra:
        # 1. Não dar erro se Ragas não tiver instalado
        # 2. Não carregar módulos pesados se flag estiver desligada
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from ragas import EvaluationDataset, SingleTurnSample

        # ── Monta o dataset de avaliação ──
        # O Ragas espera um dataset com samples no formato específico
        # Cada sample tem: pergunta, resposta, contextos, ground_truth
        sample = SingleTurnSample(
            user_input=pergunta,
            response=resposta,
            retrieved_contexts=contextos,
            reference=ground_truth or resposta,  # se não tem ground truth, usa a resposta
        )

        dataset = EvaluationDataset(samples=[sample])

        # ── Configura o LLM pra avaliação ──
        # Ragas precisa de um LLM pra calcular as métricas
        # Usamos o mesmo Ollama do projeto
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from modelos.configuracao import configuracao

        llm_avaliador = ChatOllama(
            model=configuracao.OLLAMA_MODELO_LLM,
            base_url=configuracao.OLLAMA_BASE_URL,
        )

        embeddings_avaliador = OllamaEmbeddings(
            model=configuracao.OLLAMA_MODELO_EMBEDDING,
            base_url=configuracao.OLLAMA_BASE_URL,
        )

        # ── Roda a avaliação ──
        # O evaluate() faz TUDO automaticamente:
        # - Chama o LLM pra cada métrica
        # - Calcula os scores
        # - Retorna tudo organizado
        metricas = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

        resultado = evaluate(
            dataset=dataset,
            metrics=metricas,
            llm=llm_avaliador,
            embeddings=embeddings_avaliador,
        )

        # ── Extrai os scores ──
        scores = resultado.to_pandas().iloc[0]

        resultado_final = ResultadoRagas(
            faithfulness=float(scores.get("faithfulness", 0)),
            answer_relevancy=float(scores.get("answer_relevancy", 0)),
            context_precision=float(scores.get("context_precision", 0)),
            context_recall=float(scores.get("context_recall", 0)),
        )

        print(f"📊 Ragas — Faithfulness: {resultado_final.faithfulness:.2f} | "
              f"Relevancy: {resultado_final.answer_relevancy:.2f} | "
              f"Precision: {resultado_final.context_precision:.2f} | "
              f"Recall: {resultado_final.context_recall:.2f} | "
              f"Média: {resultado_final.media:.2f}")

        return resultado_final

    except ImportError:
        print("⚠️  Ragas não instalado. pip install ragas")
        return None
    except Exception as e:
        print(f"⚠️  Erro no Ragas: {e}")
        return None


async def avaliar_rag_manual(
    pergunta: str,
    resposta: str,
    contextos: list[str],
) -> dict:
    """
    Avaliação de RAG "na mão" — SEM framework.

    Esta função mostra como você faria a avaliação MANUALMENTE,
    sem o Ragas. É mais simples mas cobre os conceitos básicos.

    ─────────────────────────────────────────────────────
    COMPARAÇÃO:

    Com Ragas:
        resultado = evaluate(dataset, metrics=[faithfulness, ...])
        # Ragas faz 10+ chamadas LLM, extrai afirmações, calcula tudo

    Na mão (esta função):
        prompt = "A resposta é fiel ao contexto? Dê nota 0-1"
        resposta = await llm.ainvoke(prompt)
        # 1 chamada LLM, avaliação mais superficial mas funcional

    A diferença: Ragas é mais rigoroso e padronizado,
    mas esta versão manual te dá 80% do valor com 20% do esforço.
    ─────────────────────────────────────────────────────
    """
    from agentes.ferramentas import criar_llm

    llm = criar_llm()

    # Junta os contextos numa string
    contexto_texto = "\n".join(f"- {c}" for c in contextos)

    # ── Prompt manual pra avaliação ──
    prompt = f"""Avalie a qualidade desta interação RAG.

PERGUNTA: {pergunta}

CONTEXTO RETORNADO PELO RAG:
{contexto_texto}

RESPOSTA GERADA:
{resposta}

Avalie de 0.0 a 1.0 cada critério:

1. FIDELIDADE: A resposta é baseada no contexto? (não inventou informação?)
2. RELEVÂNCIA: A resposta é relevante pra pergunta?
3. PRECISÃO DO CONTEXTO: Os chunks retornados são úteis pra responder?
4. COBERTURA: O contexto cobre toda informação necessária?

Responda APENAS neste formato JSON:
{{"fidelidade": 0.0, "relevancia": 0.0, "precisao_contexto": 0.0, "cobertura": 0.0}}"""

    resposta_llm = await llm.ainvoke(prompt)

    try:
        import json
        scores = json.loads(resposta_llm.content)
        return scores
    except Exception:
        return {
            "fidelidade": 0.5,
            "relevancia": 0.5,
            "precisao_contexto": 0.5,
            "cobertura": 0.5,
        }
