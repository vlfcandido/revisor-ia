"""
agentes/prompts.py — Templates de Prompt (Estratégias A e B)

Define os prompts que o LLM usa pra gerar code reviews.
Temos duas estratégias diferentes pra testar no A/B test:

- Estratégia A (Detalhada): pede review completo com categorias
- Estratégia B (Concisa): pede review direto ao ponto

O teste A/B compara qual estratégia gera reviews de melhor qualidade.

Conceitos:
- Prompt template: texto com {variáveis} que são preenchidas em runtime
- System prompt: instrução que define o "papel" do LLM
- Few-shot: dar exemplos no prompt pra guiar o LLM (não usado aqui, mas é uma técnica)
"""

# ══════════════════════════════════════════════════════════════
# PROMPT DO SISTEMA — define o "papel" do LLM
#
# Este prompt é enviado como 'system message' e define
# como o LLM deve se comportar em todas as respostas.
# ══════════════════════════════════════════════════════════════
PROMPT_SISTEMA = """Você é um revisor de código sênior com 15 anos de experiência.
Sua função é revisar código e dar feedback construtivo, específico e acionável.
Sempre baseie seus comentários em boas práticas reconhecidas."""


# ══════════════════════════════════════════════════════════════
# ESTRATÉGIA A — Review Detalhado
#
# Pede uma análise organizada por categorias:
# legibilidade, segurança, performance, boas práticas.
# Gera reviews mais longos e completos.
# ══════════════════════════════════════════════════════════════
PROMPT_ESTRATEGIA_A = """Analise o seguinte código {linguagem} de forma DETALHADA.

## Código para revisão:
```{linguagem}
{codigo}
```

## Problemas de sintaxe encontrados:
{problemas_sintaxe}

## Boas práticas relevantes (base de conhecimento):
{boas_praticas}

## Instruções:
Faça um code review organizado nas seguintes categorias:

1. **Legibilidade**: nomes de variáveis, formatação, clareza
2. **Segurança**: vulnerabilidades, validação de entrada
3. **Performance**: otimizações possíveis, complexidade
4. **Boas Práticas**: aderência a padrões, SOLID, Clean Code
5. **Sugestões**: melhorias concretas com exemplos de código

Para cada ponto, explique o problema e dê uma sugestão específica de correção."""


# ══════════════════════════════════════════════════════════════
# ESTRATÉGIA B — Review Conciso
#
# Pede uma análise direta, focada nos problemas principais.
# Gera reviews mais curtos e objetivos.
# ══════════════════════════════════════════════════════════════
PROMPT_ESTRATEGIA_B = """Revise o código {linguagem} abaixo de forma CONCISA e DIRETA.

```{linguagem}
{codigo}
```

Problemas encontrados: {problemas_sintaxe}
Contexto relevante: {boas_praticas}

Liste apenas os TOP 3 problemas mais críticos, cada um com:
- O que está errado (1 frase)
- Como corrigir (snippet de código)

Seja direto. Sem introdução, sem conclusão."""


# ══════════════════════════════════════════════════════════════
# PROMPT DO AVALIADOR (LLM-as-Judge)
#
# Este prompt é usado por um LLM separado pra avaliar
# a qualidade do review gerado. É o "juiz".
#
# Conceito: LLM-as-Judge
# - Usar um LLM pra avaliar a saída de outro LLM
# - Dá uma nota de 0 a 10 com justificativa
# - Critérios claros evitam avaliações subjetivas
# ══════════════════════════════════════════════════════════════
PROMPT_AVALIADOR = """Você é um avaliador de code reviews. Sua tarefa é avaliar
a QUALIDADE de um code review feito por outra pessoa.

## Código original:
```
{codigo}
```

## Review a avaliar:
{review}

## Critérios de avaliação (0-10):
1. **Clareza** (0-10): O review é fácil de entender?
2. **Completude** (0-10): Cobriu os problemas importantes?
3. **Acionabilidade** (0-10): As sugestões são práticas e implementáveis?

## Responda EXATAMENTE neste formato JSON:
{{
    "nota": <média das 3 notas, arredondada pra 1 decimal>,
    "justificativa": "<1-2 frases explicando a nota>"
}}"""


def obter_prompt_estrategia(estrategia: str) -> str:
    """
    Retorna o template de prompt da estratégia escolhida.

    Args:
        estrategia: "A" (detalhado) ou "B" (conciso)

    Returns:
        Template de prompt como string

    Raises:
        ValueError: se estratégia não for "A" nem "B"
    """
    # Mapa de estratégias → prompts
    mapa = {
        "A": PROMPT_ESTRATEGIA_A,
        "B": PROMPT_ESTRATEGIA_B,
    }

    if estrategia not in mapa:
        raise ValueError(f"Estratégia inválida: '{estrategia}'. Use 'A' ou 'B'.")

    return mapa[estrategia]
