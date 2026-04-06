"""
banco/conexao.py — Conexão com o Banco de Dados

Este módulo configura a conexão async com o PostgreSQL usando
SQLAlchemy 2.0. Ele cria o engine, a session factory, e uma
função pra inicializar as tabelas.

Conceitos importantes:
- AsyncEngine: motor assíncrono (não bloqueia enquanto espera o banco)
- async_sessionmaker: fábrica de sessões (cada request da API usa uma)
- Base: classe base do SQLAlchemy pra definir tabelas (ORM)
"""

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from modelos.configuracao import configuracao


# ──────────────────────────────────────────────────────────────
# ENGINE — conexão com o banco PostgreSQL
#
# O engine é o "motor" que gerencia o pool de conexões.
# echo=True imprime os SQLs no console (útil pra debug/estudo).
# ──────────────────────────────────────────────────────────────
engine = create_async_engine(
    configuracao.DATABASE_URL,
    echo=True,  # imprime SQL no console (desligar em prod)
)

# ──────────────────────────────────────────────────────────────
# SESSION FACTORY — cria sessões pra cada operação
#
# Uma sessão é como uma "transação" com o banco.
# expire_on_commit=False evita erros ao acessar dados
# após o commit (comum em apps async).
# ──────────────────────────────────────────────────────────────
fabrica_sessao = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ──────────────────────────────────────────────────────────────
# BASE — classe base pra todos os models do SQLAlchemy
#
# Todos os models (tabelas) herdam desta classe.
# Isso permite que o SQLAlchemy descubra automaticamente
# todas as tabelas quando chamarmos criar_tabelas().
# ──────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    """Classe base do ORM. Todos os models herdam dela."""
    pass


async def criar_tabelas():
    """
    Cria todas as tabelas no banco de dados.

    Usa o engine pra se conectar e cria as tabelas
    que ainda não existem. Não apaga tabelas existentes.

    IMPORTANTE: precisa importar os models antes de chamar
    esta função, senão o SQLAlchemy não sabe quais tabelas criar.
    """
    # Importa os models pra que o SQLAlchemy os registre
    import modelos.banco  # noqa: F401 — import pro side-effect

    async with engine.begin() as conexao:
        # Executa o CREATE EXTENSION e CREATE TABLE
        await conexao.run_sync(Base.metadata.create_all)


async def obter_sessao():
    """
    Gera uma sessão do banco de dados.

    Usado como dependency injection no FastAPI:
        @app.get("/algo")
        async def algo(sessao: AsyncSession = Depends(obter_sessao)):
            ...

    O 'async with' garante que a sessão é fechada
    mesmo se der erro.
    """
    async with fabrica_sessao() as sessao:
        yield sessao
