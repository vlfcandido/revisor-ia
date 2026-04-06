"""
agentes/ferramentas.py — Ferramentas (Tools) do Agente

Define as "ferramentas" que o agente LangGraph pode usar.
No LangChain, tools são funções que o LLM pode chamar
durante a execução — como "dar super poderes" ao modelo.

Conceitos:
- Tool: função decorada com @tool que o LLM pode invocar
- O LLM decide QUANDO chamar cada tool baseado na descrição
- Tools são o mecanismo de "ação" dos agentes — sem tools,
  o LLM só gera texto, com tools ele pode FAZER coisas

Neste projeto, os tools são mais simples (chamados diretamente
pelos nós do grafo), mas a estrutura é a mesma de agentes
autônomos.
"""

import json

from langchain_ollama import ChatOllama

from modelos.configuracao import configuracao
from agentes.prompts import PROMPT_SISTEMA, PROMPT_AVALIADOR


def criar_llm() -> ChatOllama:
    """
    Cria uma instância do LLM (Ollama) configurada.

    ChatOllama é a integração do LangChain com o Ollama.
    Permite usar o Ollama como se fosse qualquer outro LLM
    (OpenAI, Anthropic, etc.) — mesma interface.

    Returns:
        Instância do ChatOllama pronta pra uso
    """
    return ChatOllama(
        model=configuracao.OLLAMA_MODELO_LLM,
        base_url=configuracao.OLLAMA_BASE_URL,
        temperature=0.3,  # pouca criatividade → respostas mais consistentes
    )


async def analisar_sintaxe_codigo(codigo: str, linguagem: str) -> list[str]:
    """
    Analisa o código e identifica problemas de sintaxe/estilo.

    Usa o LLM pra detectar erros e problemas no código.
    Em produção, isso poderia usar linters reais (pylint, flake8),
    mas aqui usamos LLM pra praticar a integração.

    Args:
        codigo: código fonte a analisar
        linguagem: linguagem de programação

    Returns:
        Lista de problemas encontrados (strings)
    """
    # Cria o LLM
    llm = criar_llm()

    # Prompt pra análise de sintaxe
    prompt = f"""Analise o seguinte código {linguagem} e liste APENAS problemas
de sintaxe, estilo e erros óbvios. Responda como uma lista JSON de strings.

Código:
```{linguagem}
{codigo}
```

Responda APENAS com o JSON, sem markdown:
["problema 1", "problema 2"]

Se não houver problemas, responda: []"""

    # Chama o LLM
    resposta = await llm.ainvoke(prompt)

    # Tenta parsear o JSON da resposta
    try:
        problemas = json.loads(resposta.content)
        return problemas if isinstance(problemas, list) else []
    except json.JSONDecodeError:
        # Se o LLM não retornou JSON válido, retorna lista vazia
        return ["(análise de sintaxe não retornou formato válido)"]


async def gerar_review_com_llm(
    codigo: str,
    linguagem: str,
    problemas: list[str],
    boas_praticas: list[str],
    prompt_template: str,
) -> str:
    """
    Gera o code review usando o LLM com o prompt da estratégia.

    Args:
        codigo: código a revisar
        linguagem: linguagem do código
        problemas: problemas de sintaxe encontrados
        boas_praticas: práticas relevantes do RAG
        prompt_template: template de prompt (estratégia A ou B)

    Returns:
        Texto do code review gerado
    """
    # Cria o LLM
    llm = criar_llm()

    # Preenche o template com os dados
    prompt_preenchido = prompt_template.format(
        codigo=codigo,
        linguagem=linguagem,
        problemas_sintaxe="\n".join(f"- {p}" for p in problemas) or "Nenhum problema encontrado.",
        boas_praticas="\n".join(f"- {bp}" for bp in boas_praticas) or "Nenhuma prática relevante.",
    )

    # Envia pro LLM com o system prompt (define o "papel")
    from langchain_core.messages import SystemMessage, HumanMessage
    mensagens = [
        SystemMessage(content=PROMPT_SISTEMA),
        HumanMessage(content=prompt_preenchido),
    ]

    resposta = await llm.ainvoke(mensagens)
    return resposta.content


async def avaliar_review_com_llm(codigo: str, review: str) -> tuple[float, str]:
    """
    Avalia a qualidade de um code review usando LLM-as-Judge.

    Conceito: LLM-as-Judge
    - Um LLM separado avalia a saída de outro LLM
    - Dá nota de 0 a 10 com justificativa
    - Útil pra medir qualidade automaticamente

    Args:
        codigo: código original revisado
        review: texto do review a avaliar

    Returns:
        Tupla (nota, justificativa)
    """
    # Cria o LLM (poderia ser um modelo diferente em prod)
    llm = criar_llm()

    # Preenche o prompt do avaliador
    prompt = PROMPT_AVALIADOR.format(
        codigo=codigo,
        review=review,
    )

    # Chama o LLM avaliador
    resposta = await llm.ainvoke(prompt)

    # Parseia o JSON de resposta
    try:
        dados = json.loads(resposta.content)
        nota = float(dados.get("nota", 5.0))
        justificativa = dados.get("justificativa", "Sem justificativa")
        return nota, justificativa
    except (json.JSONDecodeError, ValueError):
        # Fallback se o LLM não retornou JSON válido
        return 5.0, "Avaliação não retornou formato válido"
