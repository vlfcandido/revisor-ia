"""
modelos/feature_flags.py — Feature Flags (Liga/Desliga Funcionalidades)

Feature flags permitem ligar e desligar funcionalidades sem
alterar código ou fazer deploy. É uma prática essencial em
projetos profissionais.

╔══════════════════════════════════════════════════════════════╗
║  POR QUE FEATURE FLAGS?                                      ║
║                                                              ║
║  1. Deploy seguro: lança código desligado, liga depois       ║
║  2. Teste A/B: liga pra 50% dos usuários                     ║
║  3. Kill switch: desliga se der problema em prod              ║
║  4. Dev local: liga features experimentais sem afetar prod    ║
║                                                              ║
║  Neste projeto, usamos feature flags pra controlar:          ║
║  - LangSmith (tracing) — pode ser caro em prod               ║
║  - Ragas (avaliação RAG) — pesado, nem sempre necessário      ║
║  - DeepEval — testes de qualidade opcionais                   ║
║  - Loop de re-geração — desliga pra respostas mais rápidas   ║
╚══════════════════════════════════════════════════════════════╝

Implementação:
- Feature flags são carregadas do .env via Pydantic Settings
- Cada flag é um bool (True/False)
- Consultadas em runtime com: flags.LANGSMITH_ATIVO

Em produção, ferramentas como LaunchDarkly ou Unleash fazem
isso com mais sofisticação (rollout gradual, segmentação, etc.).
Aqui é a versão simples pra estudo.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class FeatureFlags(BaseSettings):
    """
    Feature flags do projeto — liga/desliga funcionalidades.

    Cada flag é carregada do .env como variável de ambiente.
    Prefixo FF_ (Feature Flag) pra não conflitar com outras vars.

    Uso:
        from modelos.feature_flags import flags
        if flags.FF_LANGSMITH_ATIVO:
            configurar_langsmith()
    """

    # ── Observabilidade ──

    # Liga/desliga o tracing do LangSmith
    # Útil pra desligar em dev local (evita mandar dados pro LangSmith)
    FF_LANGSMITH_ATIVO: bool = False

    # ── Avaliação de Qualidade ──

    # Liga/desliga avaliação com Ragas (métricas de RAG)
    # Ragas é pesado — desliga pra respostas mais rápidas
    FF_RAGAS_ATIVO: bool = True

    # Liga/desliga avaliação com DeepEval
    # DeepEval roda métricas adicionais (hallucination, toxicity, etc.)
    FF_DEEPEVAL_ATIVO: bool = True

    # ── Comportamento do Agente ──

    # Liga/desliga o loop de re-geração quando nota < 7
    # Se False, o agente gera o review uma vez e retorna (mais rápido)
    FF_LOOP_REGENERACAO: bool = True

    # Número máximo de tentativas de re-geração
    # Só funciona se FF_LOOP_REGENERACAO = True
    FF_MAX_TENTATIVAS: int = 3

    # ── RAG ──

    # Liga/desliga a busca RAG no fluxo de review
    # Se False, o agente gera review sem contexto de boas práticas
    FF_RAG_ATIVO: bool = True

    # Quantidade padrão de resultados RAG
    FF_RAG_TOP_K: int = 5

    # ── Teste A/B ──

    # Liga/desliga o teste A/B nos endpoints
    # Se False, sempre usa a estratégia A
    FF_TESTE_AB_ATIVO: bool = True

    # ── Configuração do Pydantic Settings ──
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# ──────────────────────────────────────────────────────────────
# INSTÂNCIA GLOBAL — importar de qualquer lugar
#
# Uso:
#   from modelos.feature_flags import flags
#
#   if flags.FF_LANGSMITH_ATIVO:
#       configurar_langsmith()
#
#   if flags.FF_RAGAS_ATIVO:
#       resultado_ragas = await avaliar_com_ragas(...)
# ──────────────────────────────────────────────────────────────
flags = FeatureFlags()


def listar_flags() -> dict[str, bool | int]:
    """
    Retorna todas as feature flags e seus valores atuais.

    Útil pra debug e pra expor num endpoint da API.

    Returns:
        Dict com nome da flag → valor atual

    Exemplo:
        >>> listar_flags()
        {
            "FF_LANGSMITH_ATIVO": False,
            "FF_RAGAS_ATIVO": True,
            "FF_DEEPEVAL_ATIVO": True,
            ...
        }
    """
    return {
        campo: getattr(flags, campo)
        for campo in flags.model_fields
    }
