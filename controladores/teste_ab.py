"""
controladores/teste_ab.py — Teste A/B entre Estratégias de Prompt

Implementa um teste A/B simples que compara duas estratégias
de prompt pra geração de code reviews:

- Estratégia A: prompt detalhado (review por categorias)
- Estratégia B: prompt conciso (top 3 problemas)

Conceito: Teste A/B
- Técnica pra comparar duas variantes (A e B) de algo
- Executa as duas variantes com os mesmos dados
- Mede o resultado (neste caso, nota do LLM-as-judge)
- A variante com melhor resultado "vence"

Em produção, testes A/B usam estatística mais robusta
(p-value, intervalo de confiança). Aqui é simplificado
pra fins de estudo.
"""

from modelos.esquemas import CodigoEntrada, ResultadoAB
from controladores.revisor import revisar_codigo


async def executar_teste_ab(
    codigo: str,
    linguagem: str = "python",
    rodadas: int = 4,
) -> ResultadoAB:
    """
    Executa um teste A/B comparando as duas estratégias de prompt.

    Pra cada rodada, executa o review com estratégia A e B,
    coleta as notas, e compara as médias no final.

    Args:
        codigo: código fonte a revisar
        linguagem: linguagem do código
        rodadas: quantas vezes executar cada estratégia

    Returns:
        ResultadoAB com médias, vencedor, e detalhes

    Nota: cada rodada faz 2 chamadas ao LLM (A e B),
    então 4 rodadas = 8 chamadas. Com Ollama local, isso
    pode levar alguns minutos.
    """
    detalhes = []

    # Executa 'rodadas' vezes pra cada estratégia
    for i in range(rodadas):
        print(f"\n{'='*50}")
        print(f"📊 Rodada {i + 1}/{rodadas}")
        print(f"{'='*50}")

        # ── Executa estratégia A ──
        print("\n🅰️  Estratégia A (Detalhada):")
        entrada_a = CodigoEntrada(
            codigo=codigo,
            linguagem=linguagem,
            estrategia="A",
        )
        resultado_a = await revisar_codigo(entrada_a)
        detalhes.append({
            "rodada": i + 1,
            "estrategia": "A",
            "nota": resultado_a.nota_qualidade,
        })

        # ── Executa estratégia B ──
        print("\n🅱️  Estratégia B (Concisa):")
        entrada_b = CodigoEntrada(
            codigo=codigo,
            linguagem=linguagem,
            estrategia="B",
        )
        resultado_b = await revisar_codigo(entrada_b)
        detalhes.append({
            "rodada": i + 1,
            "estrategia": "B",
            "nota": resultado_b.nota_qualidade,
        })

    # ── Calcula médias ──
    notas_a = [d["nota"] for d in detalhes if d["estrategia"] == "A"]
    notas_b = [d["nota"] for d in detalhes if d["estrategia"] == "B"]

    media_a = sum(notas_a) / len(notas_a) if notas_a else 0
    media_b = sum(notas_b) / len(notas_b) if notas_b else 0

    # Determina o vencedor
    vencedor = "A" if media_a >= media_b else "B"

    print(f"\n{'='*50}")
    print(f"🏆 RESULTADO DO TESTE A/B")
    print(f"   Estratégia A (Detalhada): {media_a:.1f}/10")
    print(f"   Estratégia B (Concisa):   {media_b:.1f}/10")
    print(f"   Vencedor: Estratégia {vencedor}")
    print(f"{'='*50}")

    return ResultadoAB(
        media_a=round(media_a, 1),
        media_b=round(media_b, 1),
        vencedor=vencedor,
        rodadas=rodadas,
        detalhes=detalhes,
    )
