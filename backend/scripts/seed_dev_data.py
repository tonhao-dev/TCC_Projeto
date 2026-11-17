"""Seed de dados de desenvolvimento para a plataforma Fiscalia.

Popula o banco com um conjunto pequeno e realista de parlamentares e emendas
(várias UFs, anos e áreas funcionais) e gera os embeddings vetoriais via o
EmbeddingGenerator do projeto, de modo que o pipeline RAG híbrido (SQL + vetorial)
retorne resultados reais em ambiente local.

Uso:
    cd backend && python -m scripts.seed_dev_data
    (requer DATABASE_URL apontando para um Postgres com pgvector e o schema já criado)
"""
from app.database import SessionLocal, init_db
from app.models.parlamentar import Parlamentar
from app.models.emenda import Emenda
from app.services.indexing.embedding_generator import EmbeddingGenerator
import structlog

logger = structlog.get_logger()

PARLAMENTARES = [
    {"cod_autor": 1001, "nome": "Maria Souza", "partido": "PT", "uf": "PA"},
    {"cod_autor": 1002, "nome": "João Pereira", "partido": "PL", "uf": "SP"},
    {"cod_autor": 1003, "nome": "Ana Lima", "partido": "MDB", "uf": "BA"},
    {"cod_autor": 1004, "nome": "Carlos Andrade", "partido": "PSDB", "uf": "PA"},
    {"cod_autor": 1005, "nome": "Beatriz Rocha", "partido": "PP", "uf": "MG"},
]

# (cod_autor, nome_autor, ano, funcao, funcao_nome, subfuncao, subfuncao_nome, uf, valor, descricao)
EMENDAS = [
    (1001, "Maria Souza", 2023, "10", "Saúde", "301", "Atenção Básica", "PA",
     2_500_000.00, "Construção de unidade básica de saúde no município de Belém"),
    (1004, "Carlos Andrade", 2023, "10", "Saúde", "302", "Assistência Hospitalar", "PA",
     4_100_000.00, "Aquisição de equipamentos hospitalares para hospital regional"),
    (1001, "Maria Souza", 2022, "10", "Saúde", "301", "Atenção Básica", "PA",
     1_800_000.00, "Ampliação de posto de saúde e compra de ambulância"),
    (1002, "João Pereira", 2023, "12", "Educação", "361", "Ensino Fundamental", "SP",
     3_200_000.00, "Reforma de escola municipal e laboratório de informática"),
    (1003, "Ana Lima", 2023, "12", "Educação", "364", "Ensino Superior", "BA",
     5_600_000.00, "Apoio a universidade estadual e bolsas de pesquisa"),
    (1005, "Beatriz Rocha", 2023, "26", "Transporte", "782", "Transporte Rodoviário", "MG",
     7_900_000.00, "Pavimentação de estrada vicinal e construção de ponte"),
    (1002, "João Pereira", 2024, "12", "Educação", "361", "Ensino Fundamental", "SP",
     2_750_000.00, "Construção de creche e aquisição de mobiliário escolar"),
    (1005, "Beatriz Rocha", 2024, "10", "Saúde", "301", "Atenção Básica", "MG",
     3_400_000.00, "Programa de saúde da família e medicamentos essenciais"),
]


def run():
    init_db()
    db = SessionLocal()
    try:
        if db.query(Emenda).count() > 0:
            logger.info("seed_ja_aplicado", emendas=db.query(Emenda).count())
            return

        for p in PARLAMENTARES:
            db.merge(Parlamentar(**p))
        db.commit()

        for i, (cod, nome, ano, func, func_nome, sub, sub_nome, uf, valor, desc) in enumerate(EMENDAS, 1):
            db.add(Emenda(
                codigo_emenda=f"DEV-{ano}-{i:04d}",
                cod_autor=cod,
                nome_autor=nome,
                ano=ano,
                tipo_emenda="Emenda Individual",
                funcao=func,
                funcao_nome=func_nome,
                subfuncao=sub,
                subfuncao_nome=sub_nome,
                localidade=uf,
                uf=uf,
                valor_empenhado=valor,
                valor_liquidado=valor * 0.9,
                valor_pago=valor * 0.8,
                descricao=desc,
                fonte="Seed de desenvolvimento",
            ))
        db.commit()
        logger.info("emendas_inseridas", total=len(EMENDAS))

        # Gera embeddings reais (SQL + vetorial dependem disso)
        gen = EmbeddingGenerator()
        gen.atualizar_emendas(db)
        logger.info("seed_concluido")
    finally:
        db.close()


if __name__ == "__main__":
    run()
