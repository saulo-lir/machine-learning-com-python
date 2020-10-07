import pandas as pd

# Ler arquivo .csv
base = pd.read_csv('../data_bases/credit_data.csv') # A vairável base será do tipo DataFrame

# Exibir dados estatísticos da planilha
print(base.describe())

# 1) Tratar dados inconsistentes. Ex.: Idade negativa.
# Localizar na base de dados, no atributo 'age', os valores negativos
print(base.loc[base['age'] < 0])

# Método 1: Apagando a coluna
base.drop('age', 1, inplace=True) # 1 = Apagar coluna inteira. inplace = Não precisa atribuir a uma variável, apenas executa o comando
print(base)

# Método 2: Apagar somente os registros com problema
base.drop(base[base.age < 0].index, inplace=True)
print(base)

# Método 3: Preencher os valores com a média das idades de toda a base de dados, com excessão das idades negativas (mais recomendado)
media_idades = base['age'][base.age > 0].mean()
print(media_idades)
base.loc[base.age < 0, 'age'] = media_idades # Subistituir todas as idades negativas pela média encontrada
print(base.loc[base['age'] == 40.92770044906149])