import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
# we are going to split cronologically our data, to train and to test
# from 1990 to 2014 we be use to train and from 2018 to 2022 to test our model 
df_ready = pd.read_csv('../data/processed/matches_con_features.csv')


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