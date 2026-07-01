import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json

# ==========================================
# STEP 1: training our model with wc history
# ==========================================


# we are going to split cronologically our data, to train and to test
# from 1990 to 2014 we be use to train and from 2018 to 2022 to test our model 
df_ready = pd.read_csv('../data/processed/matches_con_features.csv')

data_training = df_ready[df_ready['year'] <= 2014] #836 matches
data_testing = df_ready[df_ready['year'] > 2014] #64 matches


X_train = data_training.drop(['result', 'home_team_score', 'away_team_score'], axis=1) #independent variables
Y_train= data_training['result'] #dependent variable 


X_test = data_testing.drop(['result', 'home_team_score', 'away_team_score'], axis=1)
y_test = data_testing['result']

#ONE-HOT ENCONDING = Data preprocessing technique used to convert categorical variables into a binary format (0s and 1s)
X_train_num = pd.get_dummies(X_train)
X_test_num = pd.get_dummies(X_test)

X_train_num, X_test_num = X_train_num.align(X_test_num, join='left', axis=1, fill_value=0) 

forest = RandomForestClassifier(n_estimators=100, random_state=42) #n_estimators= number of decision trees
forest.fit(X_train_num, Y_train)

y_pred = forest.predict(X_test_num)

precision = accuracy_score(y_test, y_pred)
print(f"¡training complete! Precision: {precision * 100:.2f}%")


# ==========================================
# STEP 2: predicting world cup 2026
# ==========================================

#we will update the average goals for each team with the data of 2026
average_goals = joblib.load('../data/processed/promedios_historicos.pkl') 
with open('../data/raw-newWc/worldcup.json', 'r', encoding='utf-8') as f:
    datos_reales = json.load(f)['matches']

for partido in datos_reales:

        
    if 'score' in partido and partido['score'] is not None:
        equipo_1 = partido['team1']
        equipo_2 = partido['team2']

        goles_eq1 = partido['score']['ft'][0]
        goles_eq2 = partido['score']['ft'][1]
        
        historia_eq1 = average_goals.get(equipo_1, 0.0)
        historia_eq2 = average_goals.get(equipo_2, 0.0)
        
        # we average the historical data with the new one
        average_goals[equipo_1] = (historia_eq1 + goles_eq1) / 2
        average_goals[equipo_2] = (historia_eq2 + goles_eq2) / 2
    
# we will use our adapted model to predict the future winners
def predict_game(equipo_local, equipo_visitante):
    avg_local = average_goals.get(equipo_local, 0.0)
    avg_visit = average_goals.get(equipo_visitante, 0.0)
    
    df_partido = pd.DataFrame({
        'home_team_name': [equipo_local],
        'away_team_name': [equipo_visitante],
        'home_team_avg_goals': [avg_local],
        'away_team_avg_goals': [avg_visit]
    })
    
    df_partido_num = pd.get_dummies(df_partido)
    df_partido_num = df_partido_num.reindex(columns=X_train_num.columns, fill_value=0)
    
    prediccion = forest.predict(df_partido_num)
    return prediccion[0]    
        
winners= {}
losers= {}

with open('../data/raw-newWc/worldcup.json', 'r', encoding='utf-8') as f:
    datos_reales = json.load(f)['matches']

    df_2026= pd.read_csv('../data/processed/calendario_2026_limpio.csv')

    for index, row in datos_reales:
        home_team = row['home_team_name']
        away_team = row['away_team_name']
        stage = row['stage']
        
        num_match = index + 1
        
        if str(home_team).startswith('W'):
            id_prev = int(home_team[1:])
            home_team = winners[id_prev] 
        elif str(home_team).startswith('L'):
            id_prev = int(home_team[1:])
            home_team = losers[id_prev] 
            
        if str(away_team).startswith('W'):
            id_prev = int(away_team[1:])
            away_team = winners[id_prev]
        elif str(away_team).startswith('L'):
            id_prev = int(away_team[1:])
            away_team = losers[id_prev]
            
        print(f' {stage} | {home_team} vs {away_team}')
            
        result = predict_game(home_team, away_team)
        
        if result == 'home team win':
            winner = home_team
            loser = away_team
        elif result == 'away team win':
            winner = away_team
            loser = home_team
        else:
            if average_goals.get(home_team, 0) > average_goals.get(away_team, 0):
                winner = home_team
                loser = away_team
            else:
                winner = away_team
                loser = home_team
                
        losers[num_match] = loser # we save the loser
        winners[num_match] = winner # we save the winner
        
        print(f" Goes to the next round: {winner}\n")
        
        