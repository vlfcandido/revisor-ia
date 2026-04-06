"""
avaliacao/deepeval_eval.py — Avaliação com DeepEval

╔══════════════════════════════════════════════════════════════╗
║  O QUE É DEEPEVAL?                                           ║
║                                                              ║
║  DeepEval é um framework de testes pra LLM apps.             ║
║  Pense nele como "pytest pra IA" — roda métricas de          ║
║  qualidade e dá pass/fail, igual um teste unitário.          ║
║                                                              ║
║  Diferença pro Ragas:                                        ║
║  - Ragas: foco em avaliar RAG (retrieval + generation)       ║
║  - DeepEval: foco mais amplo — avalia QUALQUER saída de LLM  ║
║    (hallucination, toxicity, bias, relevância, etc.)          ║
║                                                              ║
║  Os dois se complementam no projeto.                          ║
╚══════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════
COMO FUNCIONARIA NA MÃO (sem DeepEval):

1. HALLUCINATION (Alucinação):
   - Você pegaria a resposta e o contexto
   - Perguntaria ao LLM: "A resposta contém informação que NÃO
     está no contexto? Se sim, é alucinação."
   - Contaria quantas frases são alucinação vs baseadas no contexto
   → DeepEval faz isso com NLI (Natural Language Inference):
     decompõe a resposta em claims, classifica cada uma como
     "suportada" ou "contradita" pelo contexto

2. ANSWER RELEVANCY (Relevância):
   - Pegaria a pergunta e a resposta
   - Perguntaria: "A resposta responde a pergunta? 0 a 1"
   - Simples mas subjetivo
   → DeepEval gera "statements" da resposta e verifica se cada
     statement é relevante pra pergunta original

3. G-EVAL (Avaliação Customizada):
   - Você definiria critérios (clareza, completude, tom)
   - Pra cada critério, mandaria pro LLM avaliar de 1 a 5
   - Calcularia a média
   → DeepEval automatiza isso com o GEval metric:
     você define critérios em texto e ele gera o rubric

4. TOXICITY (Toxicidade):
   - Verificaria se a resposta contém linguagem ofensiva
   - Na mão: prompt "essa resposta é tóxica? sim/não"
   → DeepEval usa classificadores especializados

NA PRÁTICA:
   Sem DeepEval = vários prompts manuais + parsing + threshold manual
   Com DeepEval = define métricas, roda assert_test, pass/fail automático

   # Sem DeepEval:
   nota = await perguntar_llm("é relevante? 0-1", resposta)
   assert float(nota) > 0.7  # threshold manual

   # Com DeepEval:
   metric = AnswerRelevancyMetric(threshold=0.7)
   assert_test(test_case, [metric])  # integra com pytest!
═══════════════════════════════════════════════════════════════

Métricas do DeepEval usadas neste projeto:
┌───────────────────────┬──────────────────────────────────────┐
│ Métrica                │ O que mede                           │
├───────────────────────┼──────────────────────────────────────┤
│ AnswerRelevancy        │ Resposta é relevante pra pergunta?   │
│ Faithfulness           │ Resposta é fiel ao contexto?         │
│ Hallucination          │ LLM inventou informação?             │
│ GEval (customizada)    │ Avaliação por critérios definidos    │
└───────────────────────┴──────────────────────────────────────┘
"""

from dataclasses import dataclass, field

from modelos.feature_flags import flags


@dataclass
class ResultadoDeepEval:
    """Resultado da avaliação DeepEval com métricas e pass/fail."""
    # Score de cada métrica (0.0 a 1.0)
    answer_relevancy: float = 0.0
    faithfulness: float = 0.0
    hallucination: float = 0.0  # 0 = sem alucinação (bom), 1 = tudo alucinado
    geval_score: float = 0.0

    # Pass/fail de cada métrica (threshold padrão: 0.7)
    passou_relevancy: bool = False
    passou_faithfulness: bool = False
    passou_hallucination: bool = False  # inverte: hallucination < 0.3 = passou
    passou_geval: bool = False

    # Detalhes (razões de cada métrica)
    detalhes: dict = field(default_factory=dict)

    @property
    def passou_tudo(self) -> bool:
        """Retorna True se TODAS as métricas passaram."""
        return (
            self.passou_relevancy
            and self.passou_faithfulness
            and self.passou_hallucination
            and self.passou_geval
        )


async def avaliar_com_deepeval(
    pergunta: str,
    resposta: str,
    contextos: list[str],
    threshold: float = 0.7,
) -> ResultadoDeepEval | None:
    """
    Avalia a qualidade da resposta usando métricas DeepEval.

    Args:
        pergunta: pergunta/input original
        resposta: resposta gerada pelo LLM
        contextos: chunks retornados pelo RAG
        threshold: nota mínima pra "passar" (padrão: 0.7)

    Returns:
        ResultadoDeepEval com scores e pass/fail, ou None se desativado

    ─────────────────────────────────────────────────────
    COMO O DEEPEVAL FUNCIONA POR BAIXO:

    1. Cria um "test case" com input, output, context
    2. Pra cada métrica, chama o LLM avaliador
    3. Compara o score com o threshold
    4. Retorna pass/fail (igual pytest assert)

    A integração com pytest permite rodar:
        pytest testes/test_deepeval.py -v
    E ver pass/fail de cada métrica no terminal.
    ─────────────────────────────────────────────────────
    """
    if not flags.FF_DEEPEVAL_ATIVO:
        print("ℹ️  DeepEval desativado (FF_DEEPEVAL_ATIVO=False)")
        return None

    try:
        from deepeval import evaluate as deepeval_evaluate
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            FaithfulnessMetric,
            HallucinationMetric,
            GEval,
        )
        from deepeval.metrics import BaseMetric

        # ── Monta o test case ──
        # LLMTestCase é o "container" de dados pro DeepEval
        # É como um TestCase do pytest, mas pra LLM
        test_case = LLMTestCase(
            input=pergunta,
            actual_output=resposta,
            retrieval_context=contextos,
        )

        # ── Define as métricas ──

        # 1. Answer Relevancy — a resposta é relevante pra pergunta?
        # Na mão: prompt "A resposta '{resposta}' é relevante pra '{pergunta}'?"
        # DeepEval: decompõe em statements e verifica relevância de cada um
        metrica_relevancy = AnswerRelevancyMetric(
            threshold=threshold,  # nota mínima pra "passar"
        )

        # 2. Faithfulness — a resposta é fiel ao contexto?
        # Na mão: prompt "Cada afirmação da resposta está no contexto?"
        # DeepEval: extrai claims da resposta, verifica cada um contra o contexto
        metrica_faithfulness = FaithfulnessMetric(
            threshold=threshold,
        )

        # 3. Hallucination — o LLM inventou informação?
        # Na mão: prompt "A resposta contém informação que NÃO está no contexto?"
        # DeepEval: usa NLI pra detectar contradições com o contexto
        # ATENÇÃO: hallucination score alto = RUIM (invertido)
        metrica_hallucination = HallucinationMetric(
            threshold=0.3,  # queremos < 0.3 (pouca alucinação)
        )

        # 4. GEval — avaliação customizada por critérios
        # Esta é a mais flexível: você define os critérios em texto
        # e o DeepEval gera automaticamente um rubric de avaliação.
        #
        # Na mão: você escreveria o rubric e mandaria pro LLM
        # DeepEval: faz Chain-of-Thought pra avaliar cada critério
        metrica_geval = GEval(
            name="qualidade_code_review",
            criteria=(
                "Avalie a qualidade deste code review considerando: "
                "1) Clareza: o review é fácil de entender? "
                "2) Completude: cobriu os problemas importantes? "
                "3) Acionabilidade: as sugestões são práticas?"
            ),
            evaluation_params=[
                # Quais dados do test_case usar na avaliação
                # LLMTestCase.INPUT = a pergunta
                # LLMTestCase.ACTUAL_OUTPUT = a resposta
            ],
            threshold=threshold,
        )

        # ── Roda a avaliação ──
        # evaluate() chama o LLM pra cada métrica e calcula scores
        metricas: list[BaseMetric] = [
            metrica_relevancy,
            metrica_faithfulness,
            metrica_hallucination,
            metrica_geval,
        ]

        # Mede cada métrica individualmente pra pegar scores
        for metrica in metricas:
            metrica.measure(test_case)

        # ── Monta o resultado ──
        resultado = ResultadoDeepEval(
            answer_relevancy=metrica_relevancy.score or 0,
            faithfulness=metrica_faithfulness.score or 0,
            hallucination=metrica_hallucination.score or 0,
            geval_score=metrica_geval.score or 0,
            passou_relevancy=metrica_relevancy.is_successful(),
            passou_faithfulness=metrica_faithfulness.is_successful(),
            passou_hallucination=metrica_hallucination.is_successful(),
            passou_geval=metrica_geval.is_successful(),
            detalhes={
                "relevancy_reason": metrica_relevancy.reason or "",
                "faithfulness_reason": metrica_faithfulness.reason or "",
                "hallucination_reason": metrica_hallucination.reason or "",
                "geval_reason": metrica_geval.reason or "",
            },
        )

        # Log dos resultados
        status = "✅ PASSOU" if resultado.passou_tudo else "❌ FALHOU"
        print(f"📊 DeepEval — {status}")
        print(f"   Relevancy: {resultado.answer_relevancy:.2f} "
              f"({'✅' if resultado.passou_relevancy else '❌'})")
        print(f"   Faithfulness: {resultado.faithfulness:.2f} "
              f"({'✅' if resultado.passou_faithfulness else '❌'})")
        print(f"   Hallucination: {resultado.hallucination:.2f} "
              f"({'✅' if resultado.passou_hallucination else '❌'})")
        print(f"   GEval: {resultado.geval_score:.2f} "
              f"({'✅' if resultado.passou_geval else '❌'})")

        return resultado

    except ImportError:
        print("⚠️  DeepEval não instalado. pip install deepeval")
        return None
    except Exception as e:
        print(f"⚠️  Erro no DeepEval: {e}")
        return None


async def avaliar_qualidade_manual(
    pergunta: str,
    resposta: str,
    contextos: list[str],
) -> dict:
    """
    Avaliação de qualidade "na mão" — SEM DeepEval.

    Mostra como você faria cada métrica manualmente com
    prompts diretos ao LLM. Mais simples mas funcional.

    ─────────────────────────────────────────────────────
    COMPARAÇÃO COM DEEPEVAL:

    DeepEval (por baixo dos panos):
    1. Extrai N "claims" da resposta via LLM
    2. Pra cada claim, classifica como suportada/contradita
    3. Calcula score = claims_suportadas / total
    4. Compara com threshold
    5. Retorna pass/fail + reason

    Na mão (esta função):
    1. Manda 1 prompt pedindo notas de 0-1
    2. Parseia o JSON
    3. Pronto

    O DeepEval é mais rigoroso porque avalia claim por claim.
    A versão manual avalia "no atacado" — menos precisa mas
    funciona bem pra ter uma noção geral.
    ─────────────────────────────────────────────────────
    """
    from agentes.ferramentas import criar_llm
    import json

    llm = criar_llm()

    contexto_texto = "\n".join(f"- {c}" for c in contextos)

    prompt = f"""Você é um avaliador de qualidade de respostas de IA.

PERGUNTA: {pergunta}

CONTEXTO (informação disponível):
{contexto_texto}

RESPOSTA GERADA:
{resposta}

Avalie de 0.0 a 1.0:

1. RELEVÂNCIA: A resposta responde a pergunta? (0=irrelevante, 1=perfeita)
2. FIDELIDADE: A resposta é baseada no contexto? (0=inventou tudo, 1=100% baseada)
3. ALUCINAÇÃO: Quanta informação foi inventada? (0=nada inventado, 1=tudo inventado)
4. QUALIDADE: Nota geral do review (0=péssimo, 1=excelente)

Responda APENAS neste JSON:
{{"relevancia": 0.0, "fidelidade": 0.0, "alucinacao": 0.0, "qualidade": 0.0}}"""

    resposta_llm = await llm.ainvoke(prompt)

    try:
        scores = json.loads(resposta_llm.content)
        scores["passou"] = (
            scores.get("relevancia", 0) >= 0.7
            and scores.get("fidelidade", 0) >= 0.7
            and scores.get("alucinacao", 0) <= 0.3
            and scores.get("qualidade", 0) >= 0.7
        )
        return scores
    except Exception:
        return {
            "relevancia": 0.5,
            "fidelidade": 0.5,
            "alucinacao": 0.5,
            "qualidade": 0.5,
            "passou": False,
        }
