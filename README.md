# SkillSync-Project
Código do projeto de ML.

print("--- INICIANDO SCRIPT ---")
print

try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    import sys
    print("ETAPA 1: Bibliotecas carregadas com sucesso.")
except ImportError as e:
    print(f"!!! ERRO FATAL NA ETAPA 1 !!!")
    print(f"Biblioteca faltando. Por favor, no terminal, execute: pip install {e.name}")

print("\n--- ETAPA 2: Carregando e Juntando Dados ---")
try:
    df_progresso = pd.read_csv('cursos_progresso (1).csv', sep=';')
    df_funcionarios = pd.read_csv('rh_funcionarios (1).csv', sep=';')
    df_cursos = pd.read_csv('dim_cursos (1).csv', sep=';')

    df_master = pd.merge(df_progresso, df_funcionarios, on='ID_Funcionario')
    df_master = pd.merge(df_master, df_cursos, on='ID_Curso')
    print(f"Dados carregados e juntados. {len(df_master)} linhas no total.")
except FileNotFoundError:
    print("!!! ERRO FATAL NA ETAPA 2 !!!")
    print("Arquivo CSV não encontrado.")
    print("Por favor, VERIFIQUE se os 3 arquivos CSV ('cursos_progresso (1).csv', 'rh_funcionarios (1).csv', 'dim_cursos (1).csv') estão EXATAMENTE NA MESMA PASTA deste notebook.")
except Exception as e:
    print(f"ERRO inesperado na ETAPA 2: {e}")

print("\n--- ETAPA 3: Iniciando Engenharia de Features ---")
try:
    df_master['CONCLUIDO'] = np.where(df_master['Status_Curso'] == 'Concluído', 1, 0)

    data_referencia = pd.to_datetime('today').normalize()
    df_master['Data_Admissao'] = pd.to_datetime(df_master['Data_Admissao'], format='%d/%m/%Y', errors='coerce')
    df_master['Tempo_de_Casa_Dias'] = (data_referencia - df_master['Data_Admissao']).dt.days
    
    media_dias = df_master['Tempo_de_Casa_Dias'].mean()
    if pd.isna(media_dias):
        media_dias = 0
    df_master['Tempo_de_Casa_Dias'].fillna(media_dias, inplace=True)

    df_master['Cargo_Oficial'].fillna('Desconhecido', inplace=True)
    df_master['Nivel_Skill'].fillna('Desconhecido', inplace=True)
    
    le_cargo = LabelEncoder()
    le_nivel_skill = LabelEncoder()
    df_master['Cargo_Num'] = le_cargo.fit_transform(df_master['Cargo_Oficial'])
    df_master['Nivel_Skill_Num'] = le_nivel_skill.fit_transform(df_master['Nivel_Skill'])

    carga_cursos = df_master.groupby('ID_Funcionario')['ID_Matricula'].transform('count')
    df_master['Carga_Total_Cursos'] = carga_cursos
    print("Engenharia de Features concluída.")
except Exception as e:
    print(f"!!! ERRO FATAL NA ETAPA 3 !!!")
    print(f"Erro ao criar as features (colunas): {e}")

print("\n--- ETAPA 4: Treinando o Modelo ---")
try:
    features = ['Tempo_de_Casa_Dias', 'Cargo_Num', 'Nivel_Skill_Num', 'Carga_Total_Cursos', 'ID_Departamento']
    target = 'CONCLUIDO'

    X = df_master[features]
    y = df_master[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Modelo treinado com sucesso! Acurácia: {accuracy * 100:.2f}%")
except Exception as e:
    print(f"!!! ERRO FATAL NA ETAPA 4 !!!")
    print(f"Erro ao treinar o modelo: {e}")

print("\n--- ETAPA 5: Gerando Previsões de Risco ---")
try:
    df_prever = df_master[df_master['CONCLUIDO'] == 0].copy()
    X_para_prever = df_prever[features]

    probabilidades = model.predict_proba(X_para_prever)
    df_prever['Probabilidade_Abandono'] = probabilidades[:, 0] # prob_de_0 = Abandono

    colunas_finais = ['ID_Matricula', 'ID_Funcionario', 'Nome_Completo', 'Cargo_Oficial', 'Nome_Curso', 'Status_Curso', 'Probabilidade_Abandono']
    df_resultado_ml = df_prever[colunas_finais]

    df_resultado_ml.to_csv('previsoes_risco.csv', index=False, sep=';', decimal=',')

    print("\n--- SCRIPT CONCLUÍDO COM SUCESSO! ---")
    print(f"Arquivo 'previsoes_risco.csv' salvo com {len(df_resultado_ml)} previsões.")
    print("Agora você pode carregar este arquivo no Power BI para a Página 2.")
except Exception as e:
    print(f"!!! ERRO FATAL NA ETAPA 5 !!!")
    print(f"Erro ao gerar o arquivo CSV de previsões: {e}")
