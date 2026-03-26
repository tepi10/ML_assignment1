
import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


conn = sqlite3.connect("database.sqlite")
df = pd.read_sql_query("SELECT * FROM Match", conn)


def get_result(row):
    if row['home_team_goal'] > row['away_team_goal']:
        return "Win"
    elif row['home_team_goal'] < row['away_team_goal']:
        return "Loss"
    else:
        return "Draw"

df['result'] = df.apply(get_result, axis=1)


df['goal_diff'] = df['home_team_goal'] - df['away_team_goal']

features = ['goal_diff']


df = df[features + ['result']].dropna()

X = df[features]
y = df['result']


le = LabelEncoder()
y = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestClassifier(n_estimators=100, random_state=42)


def train_model():
    model.fit(X_train, y_train)
    print("✅ Model trained successfully!")

def test_model():
    y_pred = model.predict(X_test)
    print("\n📊 Accuracy:", accuracy_score(y_test, y_pred))
    print("\n📄 Classification Report:\n")
    print(classification_report(y_test, y_pred))

def show_confusion_matrix():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(len(cm)):
        for j in range(len(cm[0])):
            plt.text(j, i, cm[i][j], ha='center', va='center')

    plt.colorbar()
    plt.show()

def show_feature_importance():
    importance = model.feature_importances_
    print("\n📌 Feature Importance:")
    for i, col in enumerate(features):
        print(col, ":", importance[i])

def predict_custom():
    try:
        val = float(input("Enter goal difference (home - away): "))
        sample = pd.DataFrame([[val]], columns=features)  # ✅ FIXED
        pred = model.predict(sample)
        print("⚽ Predicted Result:", le.inverse_transform(pred))
    except:
        print("❌ Invalid input!")


def menu():
    while True:
        print("\n===== ⚽ FOOTBALL MATCH PREDICTION =====")
        print("1. Train Model")
        print("2. Test Model")
        print("3. Show Confusion Matrix")
        print("4. Show Feature Importance")
        print("5. Predict Custom Input")
        print("6. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            train_model()

        elif choice == '2':
            test_model()

        elif choice == '3':
            show_confusion_matrix()

        elif choice == '4':
            show_feature_importance()

        elif choice == '5':
            predict_custom()

        elif choice == '6':
            print("👋 Exiting program...")
            break

        else:
            print("❌ Invalid choice! Try again.")


menu()
