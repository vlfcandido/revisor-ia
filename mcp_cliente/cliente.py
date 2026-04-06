"""
mcp_cliente/cliente.py — Cliente MCP (Consome o Servidor)

Demonstra como consumir um MCP server programaticamente.
Conecta no nosso próprio servidor MCP e chama as tools.

Conceito:
- MCP Client: programa que se conecta a um MCP server
- StdioServerParameters: configura como conectar (via stdin/stdout)
- ClientSession: sessão de comunicação com o server
- O client descobre as tools disponíveis e pode chamá-las

Fluxo:
1. Inicia o servidor MCP como subprocesso
2. Conecta via stdio (stdin/stdout)
3. Lista as tools disponíveis
4. Chama uma tool e imprime o resultado

Pra rodar:
    python -m mcp_cliente.cliente
"""

import asyncio
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ══════════════════════════════════════════════════════════════
# CONFIGURAÇÃO DA CONEXÃO
#
# StdioServerParameters define como iniciar o servidor MCP.
# O client vai executar o comando e se comunicar via stdio.
# ══════════════════════════════════════════════════════════════

# Parâmetros pra conectar no nosso servidor MCP
parametros_servidor = StdioServerParameters(
    command="python",                         # comando pra executar
    args=["-m", "mcp_servidor.servidor"],     # argumentos (nosso server)
)


async def demonstrar_cliente_mcp():
    """
    Demonstra o uso do cliente MCP.

    1. Conecta no servidor
    2. Lista as tools disponíveis
    3. Chama 'buscar_boas_praticas'
    4. Chama 'revisar_codigo'
    """
    print("🔌 Conectando ao servidor MCP...")

    # stdio_client cria a conexão com o server
    # Ele inicia o server como subprocesso e conecta via stdio
    async with stdio_client(parametros_servidor) as (leitura, escrita):

        # Cria a sessão de comunicação
        async with ClientSession(leitura, escrita) as sessao:

            # Inicializa a sessão (handshake com o server)
            await sessao.initialize()
            print("✅ Conectado ao servidor MCP!\n")

            # ── Passo 1: Lista as tools disponíveis ──
            print("📋 Tools disponíveis:")
            resposta_tools = await sessao.list_tools()
            for tool in resposta_tools.tools:
                print(f"   - {tool.name}: {tool.description}")
            print()

            # ── Passo 2: Chama 'buscar_boas_praticas' ──
            print("🔍 Chamando 'buscar_boas_praticas'...")
            resultado_busca = await sessao.call_tool(
                "buscar_boas_praticas",
                arguments={
                    "query": "como nomear variáveis em Python",
                    "quantidade": 3,
                },
            )
            # O resultado vem como lista de Content objects
            for conteudo in resultado_busca.content:
                print(conteudo.text)
            print()

            # ── Passo 3: Chama 'revisar_codigo' ──
            print("✍️  Chamando 'revisar_codigo'...")
            codigo_exemplo = """
def calc(x, y, z):
    a = x + y
    b = a * z
    if b > 100:
        print("big")
    return b
"""
            resultado_review = await sessao.call_tool(
                "revisar_codigo",
                arguments={
                    "codigo": codigo_exemplo,
                    "linguagem": "python",
                },
            )
            for conteudo in resultado_review.content:
                print(conteudo.text)


# ══════════════════════════════════════════════════════════════
# EXECUÇÃO
#
# python -m mcp_cliente.cliente
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  MCP Client — Demonstração")
    print("=" * 60)
    print()
    asyncio.run(demonstrar_cliente_mcp())
    print("\n✅ Demonstração finalizada!")
