"""
modelos/configuracao.py — Configurações do Projeto (Pydantic Settings)

Usa Pydantic Settings v2 pra carregar variáveis de ambiente
do arquivo .env automaticamente. Isso evita hardcoding de
valores e facilita mudar configs sem alterar código.

Conceitos importantes:
- BaseSettings: classe base que lê do .env e das env vars do sistema
- model_config: configuração do Pydantic (onde achar o .env)
- Tipagem: cada campo tem um tipo, e o Pydantic valida automaticamente
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Configuracao(BaseSettings):
    """
    Configurações globais do projeto.

    O Pydantic Settings procura cada campo como variável de ambiente.
    Se não achar no ambiente, procura no arquivo .env.

    Exemplo:
        DATABASE_URL no .env → self.DATABASE_URL no Python
    """

    # --- Banco de Dados ---
    # URL de conexão com o PostgreSQL (driver asyncpg pra async)
    DATABASE_URL: str = "postgresql+asyncpg://revisor:revisor123@localhost:5432/revisor_ia"

    # --- Ollama (LLM local) ---
    # URL base do Ollama (roda na porta 11434 por padrão)
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Modelo de linguagem pra gerar texto (chat/review)
    OLLAMA_MODELO_LLM: str = "llama3.1"

    # Modelo de embeddings pra vetorizar texto (RAG)
    OLLAMA_MODELO_EMBEDDING: str = "nomic-embed-text"

    # ──────────────────────────────────────────────────────────
    # CONFIGURAÇÃO DO PYDANTIC SETTINGS
    #
    # env_file: qual arquivo .env carregar
    # env_file_encoding: encoding do arquivo (utf-8 padrão)
    # extra: "ignore" = não dá erro se tiver vars extras no .env
    # ──────────────────────────────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# ──────────────────────────────────────────────────────────────
# INSTÂNCIA GLOBAL — importar de qualquer lugar
#
# Uso:
#   from modelos.configuracao import configuracao
#   print(configuracao.OLLAMA_MODELO_LLM)  # → "llama3.1"
# ──────────────────────────────────────────────────────────────
configuracao = Configuracao()
