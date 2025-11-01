import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamelog 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score, roc_curve, ConfusionMatrixDisplay

st.set_page_config(layout="wide", page_title="Análise NBA - Regressão")


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


def get_team_id(team_name):
    """Obtém o ID de um time pelo nome completo."""
    nba_teams = teams.get_teams()
    try:
        team = next(t for t in nba_teams if t['full_name'].lower() == team_name.lower())
        return team['id']
    except StopIteration:
        return None

def load_data(team_id, season='2023-24'):
    """Carrega o log de jogos de um time para uma temporada usando LEAGUEGAMELOG."""
    st.info(f"Buscando dados para Team ID: {team_id}, Season: {season} (usando LeagueGameLog)")
    df = pd.DataFrame()
    
    try:
        st.info("Tentativa 1: Usando 'season_type_all_star'...")
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
        except Exception as e:
            st.warning(f"Falha na Tentativa 1: {e}")
            pass 
        if df.empty:
            st.info("Tentativa 2: Buscando TODOS os tipos de jogos (sem filtro)...")
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
                 st.error(f"Falha na Tentativa 2: {e}")
                 return pd.DataFrame() 

        if df.empty:
            st.warning(f"Nenhum dado retornado pela API (LeagueGameLog) para {season}.")
            return pd.DataFrame()
        
        st.info(f"Filtrando dados da liga para Team ID: {team_id}")
        df['TEAM_ID'] = pd.to_numeric(df['TEAM_ID'], errors='coerce')
        df = df[df['TEAM_ID'] == team_id].copy()


        if df.empty:
            st.warning(f"Dados da liga foram baixados, mas o Team ID {team_id} não foi encontrado.")
            return pd.DataFrame()

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
        
        df['AVG_PTS_LAST_5'] = df['PTS'].rolling(window=5, min_periods=1).mean().shift(1).fillna(0)
        df['AVG_REB_LAST_5'] = df['REB'].rolling(window=5, min_periods=1).mean().shift(1).fillna(0)
        df['AVG_AST_LAST_5'] = df['AST'].rolling(window=5, min_periods=1).mean().shift(1).fillna(0)

        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()


st.sidebar.title("🏀 Configurações da Análise")
team_name_input = st.sidebar.text_input("Digite o nome do time:", "Boston Celtics")
season_input = st.sidebar.selectbox("Escolha a Temporada:", ['2024-25', '2023-24', '2022-23', '2021-22'])
team_id = get_team_id(team_name_input)

if not team_id:
    st.sidebar.error("Time não encontrado. Verifique o nome.")
    st.stop()

df = load_data(team_id, season_input)

if df.empty:
    st.error("Nenhum dado encontrado para este time e temporada.")
    st.stop()

st.sidebar.success(f"Dados carregados para: {team_name_input} ({season_input})")
st.sidebar.info(f"{len(df)} jogos encontrados.")


st.title(f"Análise Preditiva NBA: {team_name_input}")

tab1, tab2 = st.tabs(["📊 Regressão Linear", "📈 Regressão Logística"])

with tab1:
    st.header("Regressão Linear Múltipla")
    available_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col not in ['Game_ID', 'WIN', 'TEAM_ID', 'GAME_ID']]

    default_y_index = 0
    if 'PTS' in available_cols:
        default_y_index = available_cols.index('PTS')
        
    y_var_linear = st.selectbox("Variável Dependente (Y):", available_cols, index=default_y_index)
    
    default_x_vars = ['AVG_PTS_LAST_5', 'FG3A', 'AST']

    default_x_vars = [col for col in default_x_vars if col in available_cols and col != y_var_linear]

    x_vars_linear = st.multiselect("Variáveis Independentes (X):",
                                   [col for col in available_cols if col != y_var_linear],
                                   default=default_x_vars)

    if x_vars_linear:
        X = df[x_vars_linear]
        y = df[y_var_linear]

        if len(X) < 2 or len(y) < 2:
            st.warning("Dados insuficientes para dividir em treino e teste.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if X_test.empty or y_test.empty:
                st.warning("Divisão de teste resultou em conjunto vazio. Não é possível avaliar o modelo.")
            else:
                model_linear = LinearRegression().fit(X_train, y_train)
                y_pred = model_linear.predict(X_test)

                st.subheader("Resultados")
                coefs_str = " + ".join([f"({coef:.2f} * {name})" for coef, name in zip(model_linear.coef_, x_vars_linear)])
                st.latex(f"{y_var_linear} = {model_linear.intercept_:.2f} + {coefs_str} + \\epsilon")

                mse = mean_squared_error(y_test, y_pred)
                rmse = mse ** 0.5
                
                st.metric("R²", f"{r2_score(y_test, y_pred):.3f}")
                st.metric("RMSE", f"{rmse:.3f}") 

                st.write("**Coeficientes:**")
                st.dataframe(pd.DataFrame(model_linear.coef_, index=x_vars_linear, columns=['Impacto']))

                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots()
                    ax.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
                    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
                    ax.set_xlabel("Reais")
                    ax.set_ylabel("Previstos")
                    ax.set_title("Previsão vs. Realidade")
                    st.pyplot(fig)

                with col2:
                    selected_x = st.selectbox("Escolha uma variável X:", x_vars_linear)
                    fig, ax = plt.subplots()
                    sns.regplot(data=df, x=selected_x, y=y_var_linear, ax=ax, line_kws={"color": "red"})
                    ax.set_title(f"Tendência: {selected_x} vs. {y_var_linear}")
                    st.pyplot(fig)
    else:
        st.warning("Selecione ao menos uma variável independente (X).")

with tab2:
    st.header("Regressão Logística")
    y_var_logistic = 'WIN'
    

    available_cols_logistic = [col for col in available_cols if pd.api.types.is_numeric_dtype(df[col]) and col not in ['Game_ID', 'WIN', 'TEAM_ID', 'GAME_ID']]
    
    default_x_vars_log = ['AVG_PTS_LAST_5', 'FG_PCT', 'AST', 'REB']

    default_x_vars_log = [col for col in default_x_vars_log if col in available_cols_logistic]

    x_vars_logistic = st.multiselect("Variáveis Independentes (X):",
                                     available_cols_logistic, 
                                     default=default_x_vars_log)

    if x_vars_logistic:
        X = df[x_vars_logistic]
        y = df[y_var_logistic]
        
        if len(X) < 2 or len(y) < 2:
            st.warning("Dados insuficientes para dividir em treino e teste.")

        elif y.nunique() < 2:
            st.warning(f"Não é possível treinar a Regressão Logística. Todos os dados são da mesma classe (ex: apenas vitórias ou apenas derrotas).")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            if X_test.empty or y_test.empty:
                st.warning("Divisão de teste resultou em conjunto vazio. Não é possível avaliar o modelo.")
            else:
                model_logistic = LogisticRegression(max_iter=1000)
                model_logistic.fit(X_train, y_train)

                y_pred = model_logistic.predict(X_test)
                y_proba = model_logistic.predict_proba(X_test)[:, 1]

                acc = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_proba)
                st.metric("Acurácia", f"{acc:.2%}")
                st.metric("AUC-ROC", f"{auc:.3f}")

                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots()
                    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues', display_labels=['Derrota', 'Vitória'])
                    st.pyplot(fig)
                with col2:
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
                    ax.plot([0, 1], [0, 1], 'r--')
                    ax.legend()
                    st.pyplot(fig)

                # Importância das variáveis
                st.subheader("Importância das Variáveis")
                coef_df = pd.DataFrame(model_logistic.coef_[0], index=x_vars_logistic, columns=['Coeficiente'])
                st.bar_chart(coef_df.sort_values('Coeficiente'))

                st.subheader("Previsão do Último Jogo")
                last_game = X.tail(1)
                if not last_game.empty:
                    prob = model_logistic.predict_proba(last_game)[0][1]
                    pred = "Vitória" if model_logistic.predict(last_game)[0] == 1 else "Derrota"
                    st.write(f"Probabilidade prevista de vitória: **{prob:.1%}** → {pred}")
    else:
        st.warning("Selecione ao menos uma variável independente (X).")

