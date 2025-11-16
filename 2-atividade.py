import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np
import tensorflow as tf
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamelog 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score, roc_curve, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(layout="wide", page_title="Análise NBA - Regressão e MLP")


HEADERS = {
    'Accept': 'application/json, text/plain, */*',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection': 'keep-alive',
    'Host': 'stats.nba.com',
    'Origin': 'https://www.nba.com',
    'Referer': 'https://www.nba.com/',
    'Sec-Ch-Ua': '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
    'Sec-Ch-Ua-Mobile': '?0',
    'Sec-Ch-Ua-Platform': '"Windows"',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-site',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
}

# Configuração do TensorFlow
tf.random.set_seed(42)
np.random.seed(42)


def get_team_id(team_name):
    """Obtém o ID de um time pelo nome completo."""
    nba_teams = teams.get_teams()
    try:
        team = next(t for t in nba_teams if t['full_name'].lower() == team_name.lower())
        return team['id']
    except StopIteration:
        return None

# Função de carregamento de dados (mantida e melhorada ligeiramente)
@st.cache_data
def load_data(team_id, season='2023-24'):
    """Carrega o log de jogos de um time para uma temporada."""
    df = pd.DataFrame()
    
    try:
        log = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star='Regular Season',
            headers=HEADERS,
            timeout=30
        )
        raw_data = json.loads(log.get_json())
        if "resultSets" in raw_data and raw_data["resultSets"]:
            result_set = raw_data["resultSets"][0]
            if result_set["rowSet"]:
                df = pd.DataFrame(result_set["rowSet"], columns=result_set["headers"])
    except Exception:
        # Tenta novamente sem filtro de Season Type se a primeira falhar
        try:
            log = leaguegamelog.LeagueGameLog(
                season=season,
                headers=HEADERS,
                timeout=30
            )
            raw_data = json.loads(log.get_json())
            if "resultSets" in raw_data and raw_data["resultSets"]:
                result_set = raw_data["resultSets"][0]
                if result_set["rowSet"]:
                    df = pd.DataFrame(result_set["rowSet"], columns=result_set["headers"])
        except Exception as e:
            st.error(f"Falha ao buscar dados da NBA: {e}")
            return pd.DataFrame()
        
    if df.empty:
        return pd.DataFrame()
    
    df['TEAM_ID'] = pd.to_numeric(df['TEAM_ID'], errors='coerce')
    df = df[df['TEAM_ID'] == team_id].copy()

    cols_to_numeric = [
        'PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
        'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST',
        'STL', 'BLK', 'TOV', 'PF'
    ]
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['WIN'] = df['WL'].apply(lambda x: 1 if x == 'W' else 0)

    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(by='GAME_DATE')
    
    # Features de histórico (mantidas)
    df['AVG_PTS_LAST_5'] = df['PTS'].rolling(window=5, min_periods=1).mean().shift(1).fillna(df['PTS'].mean())
    df['AVG_REB_LAST_5'] = df['REB'].rolling(window=5, min_periods=1).mean().shift(1).fillna(df['REB'].mean())
    df['AVG_AST_LAST_5'] = df['AST'].rolling(window=5, min_periods=1).mean().shift(1).fillna(df['AST'].mean())

    df = df.dropna(subset=['AVG_PTS_LAST_5', 'FG_PCT', 'AST', 'REB', 'WIN']).reset_index(drop=True)

    return df


def prepare_data_for_mlp(df, x_vars, y_var, test_size=0.2, random_state=42):
    """Prepara os dados, escalona e divide para o treinamento da MLP."""
    X = df[x_vars]
    y = df[y_var]
    
    # 1. Divisão
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 2. Escalonamento
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def create_mlp_model(input_dim):
    """Cria a arquitetura da MLP."""
    model = Sequential([
        # Primeira Camada Oculta: 16 Neurônios, Ativação ReLU
        Dense(16, activation='relu', input_shape=(input_dim,)),
        # Regularização: Dropout
        Dropout(0.2), 
        # Segunda Camada Oculta: 8 Neurônios, Ativação ReLU
        Dense(8, activation='relu'),
        # Camada de Saída: 1 Neurônio, Ativação Sigmoide (para classificação binária)
        Dense(1, activation='sigmoid') 
    ])
    return model

def train_mlp(model, X_train, y_train, X_val, y_val, optimizer_choice, epochs=200):
    """Compila e treina a MLP com Early Stopping."""
    
    if optimizer_choice == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    elif optimizer_choice == 'RMSProp':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    else: # Adam
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Early Stopping: monitora a perda de validação, paciência=10 épocas
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping],
                        verbose=0)
    
    return model, history


# --- Streamlit App Layout ---

st.sidebar.title("🏀 Configurações da Análise")
# Defina "Denver Nuggets" como o time padrão
team_name_input = st.sidebar.text_input("Digite o nome do time:", "Denver Nuggets")
season_input = st.sidebar.selectbox("Escolha a Temporada:", ['2023-24', '2022-23', '2021-22', '2024-25'])
team_id = get_team_id(team_name_input)

if not team_id:
    st.sidebar.error("Time não encontrado. Verifique o nome.")
    st.stop()

df = load_data(team_id, season_input)

if df.empty:
    st.error("Nenhum dado encontrado para este time e temporada. Tente outra temporada.")
    st.stop()

st.sidebar.success(f"Dados carregados para: {team_name_input} ({season_input})")
st.sidebar.info(f"{len(df)} jogos encontrados.")

# Definindo as features para Classificação (MLP e Logística)
X_VARS_CLASSIFICATION = ['AVG_PTS_LAST_5', 'FG_PCT', 'AST', 'REB']
Y_VAR_CLASSIFICATION = 'WIN'


st.title(f"Análise Preditiva NBA: {team_name_input} - Atividade 2")

tab1, tab2, tab3, tab4 = st.tabs([
    "📃 Relatório (Atividade 2 - Parte 1)", 
    "📈 Regressão Logística (Base)",
    "🧠 MLP (Previsão de Vitória)",
    "🏆 Comparação Final (Atividade 2 - Parte 2)"
])

# --- TAB 1: RELATÓRIO MLP (PARTE 1) ---

with tab1:
    st.header("Relatório da Arquitetura e Treinamento da MLP")
    
    st.markdown("""
    Este relatório detalha a arquitetura, parâmetros e técnicas de regularização utilizadas para o treinamento da Rede Neural Perceptron Multicamadas (MLP).
    As *features* (variáveis independentes) selecionadas para a classificação de Vitória (WIN) foram as mesmas com melhor desempenho na Regressão Logística: **Média de Pontos nos Últimos 5 Jogos**, **Percentual de Arremessos de Campo (FG_PCT)**, **Assistências (AST)** e **Rebotes (REB)**.
    """)
    st.markdown("---")
    
    st.subheader("Configuração da Rede (Arquitetura)")
    st.markdown("""
    | Questão | Configuração Implementada | Justificativa |
    | :--- | :--- | :--- |
    | **Função de Ativação** | **Ocultas:** ReLU. **Saída:** Sigmoide. | **ReLU** (Rectified Linear Unit) é usada nas camadas ocultas por ser computacionalmente eficiente e prevenir o *vanishing gradient*. A **Sigmoide** é obrigatória na camada de saída para classificação binária, pois comprime o resultado entre 0 e 1 (probabilidade de vitória). |
    | **Camadas Ocultas** | **Duas** (1ª com 16 neurônios, 2ª com 8 neurônios). | Duas camadas oferecem poder de representação suficiente para aprender relações não-lineares nos dados, sendo um bom equilíbrio entre complexidade e risco de *overfitting* para este volume de dados. |
    | **Neurônios (Entrada)** | **4** (Uma para cada *feature*: `AVG_PTS_LAST_5`, `FG_PCT`, `AST`, `REB`). | O número de neurônios de entrada deve ser igual ao número de *features* (variáveis independentes) utilizadas. |
    | **Neurônios (Oculta)** | **16 (1ª)** e **8 (2ª)**. | Escolha empírica, seguindo a regra de ter mais neurônios no início e afunilando para a saída, permitindo que a rede aprenda padrões progressivamente mais complexos. |
    | **Neurônios (Saída)** | **1**. | Necessário apenas um neurônio para a classificação binária, que retorna a probabilidade de a equipe vencer (classe '1'). |
    """)

    st.subheader("Parâmetros de Treinamento")
    st.markdown("""
    | Parâmetro | Configuração | Justificativa |
    | :--- | :--- | :--- |
    | **Épocas** | **200 (Máximo)** | Um número alto o suficiente para garantir a convergência, mas o treinamento é controlado pelo **Early Stopping**. |
    | **Camadas/Técnicas** | `Dense`, `Dropout`, `optimizer`, `loss` | **Dense** (camadas totalmente conectadas) é o core da MLP. **Dropout** é usado para regularização. O **Optimizer** e **Loss** são definidos abaixo. |
    | **Função de Perda (`loss`)** | `binary_crossentropy` | Métrica padrão e mais adequada para problemas de classificação binária (0 ou 1). |
    | **Divisão de Dados** | **80% Treino / 20% Teste** (utilizado como `validation split` para a função `fit`). | Divisão comum para garantir que o modelo aprenda com a maioria dos dados e seja avaliado em um conjunto não visto. |
    """)

    st.subheader("Prevenção de Overfitting")
    st.markdown("""
    Para evitar que o modelo decore os dados de treino, foram aplicadas as seguintes técnicas:
    1.  **Early Stopping:** Monitora a `val_loss` (perda nos dados de validação). Se a perda não diminuir por **10 épocas** (`patience=10`), o treinamento é interrompido e os melhores pesos são restaurados.
    2.  **Dropout (0.2):** Aleatoriamente, 20% dos neurônios são desativados em algumas camadas. Isso força a rede a não depender de um subconjunto específico de neurônios, melhorando a generalização.
    3.  **Escalonamento (StandardScaler):** As *features* são padronizadas antes do treino, o que melhora a estabilidade e velocidade da convergência.
    """)

    st.subheader("Métricas de Regressão na Classificação")
    st.markdown("""
    - **Calculou MAE, RMSE e R2 para cada estatística?** **Não.**
    - **Justificativa:** MAE, RMSE e R2 são métricas para modelos de **Regressão** (previsão de valores contínuos como pontos). Para a classificação binária (Vitória/Derrota), utilizamos **Acurácia** e **AUC-ROC** como métricas de desempenho.
    """)


# --- TAB 2: REGRESSÃO LOGÍSTICA (Base) ---

with tab2:
    st.header("Regressão Logística para Classificação")
    
    # Treina o modelo Logístico
    if not df.empty and Y_VAR_CLASSIFICATION in df.columns:
        X_log = df[X_VARS_CLASSIFICATION]
        y_log = df[Y_VAR_CLASSIFICATION]

        # Garantir que LogReg tenha classes suficientes
        if y_log.nunique() < 2:
            st.warning("Dados insuficientes para Regressão Logística (apenas uma classe de resultado).")
            st.stop()
            
        X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
            X_log, y_log, test_size=0.2, random_state=42, stratify=y_log
        )
        
        # Escala os dados de Logística para boas práticas (embora não seja essencial como na MLP)
        scaler_log = StandardScaler()
        X_train_log_scaled = scaler_log.fit_transform(X_train_log)
        X_test_log_scaled = scaler_log.transform(X_test_log)
        
        model_logistic = LogisticRegression(max_iter=1000)
        model_logistic.fit(X_train_log_scaled, y_train_log)

        y_pred_log = model_logistic.predict(X_test_log_scaled)
        y_proba_log = model_logistic.predict_proba(X_test_log_scaled)[:, 1]

        acc_log = accuracy_score(y_test_log, y_pred_log)
        auc_log = roc_auc_score(y_test_log, y_proba_log)
        
        st.subheader("Resultados da Regressão Logística")
        colA, colB = st.columns(2)
        with colA:
            st.metric("Acurácia", f"{acc_log:.2%}")
        with colB:
            st.metric("AUC-ROC", f"{auc_log:.3f}")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Matriz de Erros (Logística)")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_test_log, y_pred_log, ax=ax, cmap='Blues', display_labels=['Derrota', 'Vitória'])
            st.pyplot(fig)
        with col2:
            st.subheader("Curva ROC (Logística)")
            fpr, tpr, _ = roc_curve(y_test_log, y_proba_log)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {auc_log:.3f}")
            ax.plot([0, 1], [0, 1], 'r--')
            ax.set_title("Curva ROC")
            ax.legend()
            st.pyplot(fig)
            
        st.session_state['log_reg_proba'] = y_proba_log
        st.session_state['y_test'] = y_test_log
    else:
        st.warning("Dados não carregados ou coluna 'WIN' ausente.")


# --- TAB 3: MLP (PARTE 1 - GRÁFICOS E TREINAMENTO) ---

with tab3:
    st.header("MLP: Treinamento e Diagnóstico")
    
    st.subheader("Teste de Otimizadores")
    optimizer_choice = st.selectbox("Escolha o Otimizador para Treinamento:", 
                                    ['Adam (Padrão)', 'Stochastic Gradient Descent (SGD)', 'RMSProp'])
    
    # Prepara os dados uma única vez
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data_for_mlp(
        df, X_VARS_CLASSIFICATION, Y_VAR_CLASSIFICATION
    )
    
    if st.button("Treinar MLP e Gerar Gráficos"):
        with st.spinner(f"Treinando MLP com {optimizer_choice}..."):
            input_dim = X_train_scaled.shape[1]
            model_mlp = create_mlp_model(input_dim)
            
            optimizer_name = optimizer_choice.split(' ')[0] # Pega Adam, SGD ou RMSProp
            
            model_mlp, history = train_mlp(
                model_mlp, 
                X_train_scaled, 
                y_train.values, 
                X_test_scaled, 
                y_test.values, 
                optimizer_name
            )

        st.success("Treinamento Concluído!")
        
        # --- Gráfico de Evolução do Erro ---
        st.subheader("Evolução do Erro (Loss) Durante o Treinamento")
        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(history.history['loss'], label='Loss de Treino')
        ax_loss.plot(history.history['val_loss'], label='Loss de Validação')
        ax_loss.set_title(f'Erro (Loss) por Época ({optimizer_name})')
        ax_loss.set_xlabel('Época')
        ax_loss.set_ylabel('Binary Cross-Entropy Loss')
        ax_loss.legend()
        st.pyplot(fig_loss)
        st.markdown("""
        * **Interpretação:** Se a linha de validação subir enquanto a de treino cai, há **overfitting**. Se ambas caírem e estabilizarem, o modelo convergiu. O **Early Stopping** parou o treino no ponto ideal.
        """)

        # --- Matriz de Erros (2 Gráficos: Confusão e ROC) ---
        y_proba_mlp = model_mlp.predict(X_test_scaled).flatten()
        y_pred_mlp = (y_proba_mlp > 0.5).astype(int)
        
        acc_mlp = accuracy_score(y_test, y_pred_mlp)
        auc_mlp = roc_auc_score(y_test, y_proba_mlp)

        st.subheader("Métricas de Desempenho da MLP")
        colC, colD = st.columns(2)
        with colC:
            st.metric("Acurácia da MLP", f"{acc_mlp:.2%}")
        with colD:
            st.metric("AUC-ROC da MLP", f"{auc_mlp:.3f}")


        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Matriz de Erros (MLP)")
            fig_conf, ax_conf = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred_mlp, ax=ax_conf, cmap='Reds', display_labels=['Derrota', 'Vitória'])
            st.pyplot(fig_conf)
        
        with col4:
            st.subheader("Curva ROC (MLP)")
            fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_proba_mlp)
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr_mlp, tpr_mlp, label=f"AUC MLP = {auc_mlp:.3f}")
            ax_roc.plot([0, 1], [0, 1], 'k--')
            ax_roc.set_title("Curva ROC")
            ax_roc.legend()
            st.pyplot(fig_roc)

        # Armazenar resultados para a comparação final
        st.session_state['mlp_proba'] = y_proba_mlp
        st.session_state['mlp_acc'] = acc_mlp
        st.session_state['log_acc'] = acc_log
        st.session_state['y_test_final'] = y_test
        st.session_state['is_trained'] = True

    
# --- TAB 4: COMPARAÇÃO FINAL (PARTE 2) ---

with tab4:
    st.header("Comparação de Modelos: Regressão Logística vs. MLP")
    st.markdown("Esta seção atende à **Atividade 2 – Parte 2**, comparando o desempenho da Regressão Logística (Atividade 1) com a MLP (Atividade 2) no mesmo conjunto de teste.")

    if 'is_trained' in st.session_state and st.session_state['is_trained']:
        y_test_final = st.session_state['y_test_final']
        log_reg_proba = st.session_state['log_reg_proba']
        mlp_proba = st.session_state['mlp_proba']
        
        # Cria o DataFrame de comparação
        df_comp = pd.DataFrame({
            'Jogo': range(len(y_test_final)),
            'Resultado Real': y_test_final.values,
            'Probabilidade (LogReg)': log_reg_proba,
            'Probabilidade (MLP)': mlp_proba
        })
        
        st.subheader("Acurácia por Modelo")
        col1, col2 = st.columns(2)
        col1.metric("Acurácia Regressão Logística", f"{st.session_state['log_acc']:.2%}")
        col2.metric("Acurácia MLP", f"{st.session_state['mlp_acc']:.2%}")

        st.subheader("Gráfico Comparativo: Probabilidade de Vitória no Conjunto de Teste")
        
        df_melt = df_comp.melt(id_vars=['Jogo', 'Resultado Real'], 
                                value_vars=['Probabilidade (LogReg)', 'Probabilidade (MLP)'], 
                                var_name='Modelo', value_name='Probabilidade')

        # Visualização: Previsão vs Realidade
        fig_comp, ax_comp = plt.subplots(figsize=(12, 6))
        
        # Plota as probabilidades previstas por modelo
        sns.barplot(x='Jogo', y='Probabilidade', hue='Modelo', data=df_melt, ax=ax_comp, alpha=0.7)
        
        # Adiciona a linha do resultado real (0 ou 1)
        ax_comp.plot(df_comp['Resultado Real'], 'k--', label='Resultado Real (1=Vitória, 0=Derrota)', linewidth=2)
        
        # Adiciona o limiar de decisão (0.5)
        ax_comp.axhline(0.5, color='gray', linestyle=':', label='Limiar de Decisão (0.5)')

        ax_comp.set_title("Comparação de Probabilidades de Vitória (Logística vs. MLP)")
        ax_comp.set_xlabel("Jogos de Teste")
        ax_comp.set_ylabel("Probabilidade de Vitória")
        ax_comp.legend(loc='upper right')
        st.pyplot(fig_comp)
        
        st.markdown("""
        **Interpretação do Gráfico:**
        - O eixo horizontal (`Jogo`) representa os jogos do conjunto de teste.
        - As barras mostram a **Probabilidade de Vitória** prevista por cada modelo.
        - A linha tracejada preta (`Resultado Real`) mostra o resultado verdadeiro: 1 (Vitória) ou 0 (Derrota).
        - **Resultado Ideal:** As barras deveriam estar próximas de 1 quando o Resultado Real for 1, e próximas de 0 quando o Resultado Real for 0.
        """)
        
    else:
        st.warning("Treine a MLP na aba 'MLP' antes de visualizar a comparação.")

# --- Tab 1: Regressão Linear (Mantida para fins de Atividade 1) ---

with st.expander("Regressão Linear Múltipla (Modelo de Estatísticas Contínuas - Não Classificação)"):
    st.header("Regressão Linear Múltipla")
    available_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col not in ['GAME_ID', 'WIN', 'TEAM_ID']]

    default_y_index = 0
    if 'PTS' in available_cols:
        default_y_index = available_cols.index('PTS')
        
    y_var_linear = st.selectbox("Variável Dependente (Y):", available_cols, index=default_y_index, key='linear_y')
    
    default_x_vars = ['AVG_PTS_LAST_5', 'FG3A', 'AST']
    default_x_vars = [col for col in default_x_vars if col in available_cols and col != y_var_linear]

    x_vars_linear = st.multiselect("Variáveis Independentes (X):",
                                   [col for col in available_cols if col != y_var_linear],
                                   default=default_x_vars, key='linear_x')

    if x_vars_linear:
        X = df[x_vars_linear]
        y = df[y_var_linear]

        if len(X) < 2 or len(y) < 2:
            st.warning("Dados insuficientes para dividir em treino e teste.")
        else:
            X_train, X_test, y_train, y_test_lin = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if X_test.empty or y_test_lin.empty:
                st.warning("Divisão de teste resultou em conjunto vazio.")
            else:
                model_linear = LinearRegression().fit(X_train, y_train)
                y_pred_lin = model_linear.predict(X_test)

                st.subheader("Resultados")
                coefs_str = " + ".join([f"({coef:.2f} * {name})" for coef, name in zip(model_linear.coef_, x_vars_linear)])
                st.latex(f"{y_var_linear} = {model_linear.intercept_:.2f} + {coefs_str} + \\epsilon")

                mse = mean_squared_error(y_test_lin, y_pred_lin)
                rmse = mse ** 0.5
                
                st.metric("R²", f"{r2_score(y_test_lin, y_pred_lin):.3f}")
                st.metric("RMSE", f"{rmse:.3f}") 

                st.write("**Coeficientes:**")
                st.dataframe(pd.DataFrame(model_linear.coef_, index=x_vars_linear, columns=['Impacto']))

                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots()
                    ax.scatter(y_test_lin, y_pred_lin, alpha=0.7, edgecolors='k')
                    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
                    ax.set_xlabel("Reais")
                    ax.set_ylabel("Previstos")
                    ax.set_title("Previsão vs. Realidade (Linear)")
                    st.pyplot(fig)

                with col2:
                    selected_x = st.selectbox("Escolha uma variável X:", x_vars_linear, key='linear_scatter')
                    fig, ax = plt.subplots()
                    sns.regplot(data=df, x=selected_x, y=y_var_linear, ax=ax, line_kws={"color": "red"})
                    ax.set_title(f"Tendência: {selected_x} vs. {y_var_linear}")
                    st.pyplot(fig)
    else:
        st.warning("Selecione ao menos uma variável independente (X) para Regressão Linear.")
