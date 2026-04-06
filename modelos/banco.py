"""
modelos/banco.py — Models do SQLAlchemy (Tabelas do Banco)

Define as tabelas do banco de dados usando SQLAlchemy ORM.
A tabela principal usa pgvector pra armazenar embeddings.

Conceitos importantes:
- Mapped / mapped_column: forma moderna (SQLAlchemy 2.0) de definir colunas
- Vector: tipo de coluna do pgvector pra armazenar vetores de embeddings
- pgvector: extensão do PostgreSQL que permite busca por similaridade vetorial
  (encontrar os textos mais "parecidos" com uma query usando distância de cosseno)

No MVC, este é o Model — representa a estrutura dos dados no banco.
"""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from banco.conexao import Base


class BoaPratica(Base):
    """
    Tabela: boas_praticas — Armazena boas práticas de código.

    Cada registro contém um chunk de texto sobre uma boa prática
    e seu embedding vetorial (768 dimensões do nomic-embed-text).

    O pgvector usa esses vetores pra encontrar as práticas mais
    relevantes pra um dado código, usando distância de cosseno.

    Exemplo de um registro:
        texto: "Use nomes descritivos pra variáveis..."
        fonte: "clean_code"
        embedding: [0.012, -0.034, 0.056, ...]  (768 floats)
    """
    # Nome da tabela no banco
    __tablename__ = "boas_praticas"

    # Chave primária auto-incrementada
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # Texto da boa prática (o conteúdo legível)
    texto: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # Fonte/origem do texto (ex: "clean_code", "solid")
    fonte: Mapped[str] = mapped_column(
        String(100),
        default="manual",
    )

    # ──────────────────────────────────────────────────────────
    # EMBEDDING — vetor de 768 dimensões (nomic-embed-text)
    #
    # Este é o coração do RAG vetorial:
    # - Cada texto é convertido num vetor de 768 números
    # - Textos com significados parecidos têm vetores parecidos
    # - O pgvector busca os vetores mais próximos (similarity search)
    #
    # Vector(768): o número é a dimensão do modelo de embedding
    # nomic-embed-text gera vetores de 768 dimensões
    # ──────────────────────────────────────────────────────────
    embedding = mapped_column(
        Vector(768),  # 768 dimensões do nomic-embed-text
        nullable=True,
    )

    # Data de criação (preenchida automaticamente pelo banco)
    criado_em: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    def __repr__(self):
        """Representação legível do objeto (útil pra debug)."""
        return f"<BoaPratica id={self.id} fonte='{self.fonte}' texto='{self.texto[:50]}...'>"


class HistoricoReview(Base):
    """
    Tabela: historico_reviews — Salva reviews gerados.

    Guarda o histórico de reviews pra análise posterior.
    Útil pra comparar estratégias e ver evolução.
    """
    __tablename__ = "historico_reviews"

    # Chave primária
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # Código que foi revisado
    codigo: Mapped[str] = mapped_column(Text, nullable=False)

    # Linguagem do código
    linguagem: Mapped[str] = mapped_column(String(50), default="python")

    # Texto do review gerado
    review: Mapped[str] = mapped_column(Text, nullable=False)

    # Nota de qualidade (0-10)
    nota_qualidade: Mapped[float] = mapped_column(nullable=False)

    # Qual estratégia foi usada (A ou B)
    estrategia: Mapped[str] = mapped_column(String(1), nullable=False)

    # Quantas tentativas foram necessárias
    tentativas: Mapped[int] = mapped_column(Integer, default=1)

    # Data de criação
    criado_em: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    def __repr__(self):
        return f"<HistoricoReview id={self.id} nota={self.nota_qualidade} estrategia='{self.estrategia}'>"
