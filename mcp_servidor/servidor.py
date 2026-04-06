"""
mcp_servidor/servidor.py — Servidor MCP (Model Context Protocol)

MCP (Model Context Protocol) é um protocolo aberto da Anthropic
que permite expor "ferramentas" (tools) pra LLMs consumirem.

Pense no MCP como uma API, mas feita especificamente pra IAs:
- API REST: humanos chamam endpoints via HTTP
- MCP Server: IAs chamam tools via protocolo MCP

Benefícios:
- Claude, Cursor, e outros clientes MCP podem usar suas tools
- Padroniza como IAs interagem com serviços externos
- Desacopla o LLM da implementação (o LLM não precisa saber
  como o RAG funciona, só chama a tool)

Este servidor expõe 2 tools:
1. revisar_codigo — faz review completo usando o agente LangGraph
2. buscar_boas_praticas — busca no pgvector (RAG)

Pra rodar:
    python -m mcp_servidor.servidor
"""

from mcp.server.fastmcp import FastMCP

# ══════════════════════════════════════════════════════════════
# CRIAÇÃO DO SERVIDOR MCP
#
# FastMCP é a forma mais simples de criar um MCP server.
# Ele abstrai o protocolo e permite definir tools como
# funções Python normais, usando decorators.
# ══════════════════════════════════════════════════════════════

# Cria o servidor MCP com nome e descrição
mcp = FastMCP(
    "Revisor IA",
    description="Servidor MCP pra revisão de código com IA",
)


# ══════════════════════════════════════════════════════════════
# TOOL 1: revisar_codigo
#
# Expõe o fluxo completo de revisão como uma tool MCP.
# Qualquer cliente MCP (Claude, Cursor, etc.) pode chamar
# esta tool pra revisar código.
#
# O decorator @mcp.tool() registra a função como uma tool.
# O MCP gera automaticamente o schema a partir dos type hints
# e docstring.
# ══════════════════════════════════════════════════════════════

@mcp.tool()
async def revisar_codigo(codigo: str, linguagem: str = "python") -> str:
    """
    Revisa código usando agente IA com RAG e avaliação de qualidade.

    Args:
        codigo: código fonte a ser revisado
        linguagem: linguagem de programação (padrão: python)

    Returns:
        Review completo com nota de qualidade
    """
    # Importa aqui pra evitar circular import
    from modelos.esquemas import CodigoEntrada
    from controladores.revisor import revisar_codigo as _revisar

    # Monta a entrada e executa o review
    entrada = CodigoEntrada(codigo=codigo, linguagem=linguagem)
    resultado = await _revisar(entrada)

    # Formata a resposta como texto (MCP tools retornam texto)
    return (
        f"## Code Review\n\n"
        f"{resultado.review}\n\n"
        f"---\n"
        f"**Nota de Qualidade:** {resultado.nota_qualidade}/10\n"
        f"**Estratégia:** {resultado.estrategia}\n"
        f"**Tentativas:** {resultado.tentativas}\n"
        f"**Justificativa:** {resultado.justificativa}"
    )


# ══════════════════════════════════════════════════════════════
# TOOL 2: buscar_boas_praticas
#
# Expõe a busca RAG como uma tool MCP.
# Permite que clientes MCP busquem boas práticas
# no knowledge base sem precisar saber como o RAG funciona.
# ══════════════════════════════════════════════════════════════

@mcp.tool()
async def buscar_boas_praticas(query: str, quantidade: int = 5) -> str:
    """
    Busca boas práticas de código relevantes no knowledge base.

    Args:
        query: texto de busca (ex: "como nomear variáveis")
        quantidade: quantos resultados retornar (padrão: 5)

    Returns:
        Lista de boas práticas relevantes
    """
    from banco.conexao import fabrica_sessao
    from rag.vetorial import buscar_similares

    # Abre sessão e busca no pgvector
    async with fabrica_sessao() as sessao:
        resultados = await buscar_similares(sessao, query, quantidade)

    if not resultados:
        return "Nenhuma boa prática encontrada. A base pode estar vazia."

    # Formata os resultados como texto numerado
    texto = "## Boas Práticas Encontradas\n\n"
    for i, resultado in enumerate(resultados, 1):
        texto += f"{i}. {resultado}\n\n"

    return texto


# ══════════════════════════════════════════════════════════════
# EXECUÇÃO DO SERVIDOR
#
# Quando executado diretamente, inicia o servidor MCP.
# O servidor usa stdio (stdin/stdout) por padrão —
# é assim que clientes MCP se comunicam com servers.
#
# Pra testar: python -m mcp_servidor.servidor
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🚀 Iniciando servidor MCP — Revisor IA")
    print("   Tools disponíveis:")
    print("   - revisar_codigo(codigo, linguagem)")
    print("   - buscar_boas_praticas(query, quantidade)")
    print("   Aguardando conexões via stdio...")
    mcp.run()
