from pathlib import Path


PASTA_PROJETO = Path(__file__).resolve().parents[2]

PASTA_DADOS = PASTA_PROJETO / "dados"

# coloque abaixo o caminho para os arquivos de dados de seu projeto
DADOS_ORIGINAIS = PASTA_DADOS / "creditcard.zip"
DADOS_TRATADOS = PASTA_DADOS / "crediticard_mult_var_treino"
# DADOS_TRATADOS_ORIGINAL = PASTA_DADOS / "creditcard_original.parquet"

# coloque abaixo o caminho para os arquivos de modelos de seu projeto
PASTA_MODELOS = PASTA_PROJETO / "modelos"

# coloque abaixo outros caminhos que você julgar necessário
PASTA_RELATORIOS = PASTA_PROJETO / "relatorios"
PASTA_IMAGENS = PASTA_RELATORIOS / "imagens"
