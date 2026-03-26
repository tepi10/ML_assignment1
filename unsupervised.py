
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


file_path = "Career Mode player datasets - FIFA 15-21.xlsx"


df = pd.read_excel(file_path, sheet_name="FIFA 21")

print("Columns available:\n", df.columns)


features = [
    'overall',
    'pace',
    'shooting',
    'passing',
    'dribbling',
    'defending',
    'physic'
]


X = df[features]

# Handle missing values
X = X.fillna(X.mean())

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clusters_created = False




def show_data_info():
    print("\n📊 Dataset Info:\n")
    print(df[features].describe())


def elbow_method():
    inertia = []

    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    plt.plot(range(1, 10), inertia)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.show()


def apply_kmeans():
    global clusters_created

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    clusters_created = True

    print("\n✅ Clustering Completed!")
    print("\n📌 Cluster Summary:\n")
    print(df.groupby('Cluster')[features].mean())


def visualize_clusters():
    if not clusters_created:
        print("❌ Run clustering first!")
        return

    plt.scatter(df['pace'], df['shooting'], c=df['Cluster'])
    plt.xlabel("Pace")
    plt.ylabel("Shooting")
    plt.title("Player Clusters")
    plt.show()


def pca_visualization():
    if not clusters_created:
        print("❌ Run clustering first!")
        return

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'])
    plt.title("PCA Cluster Visualization")
    plt.show()




def menu():
    while True:
        print("\n===== ⚽ PLAYER CLUSTERING (FIFA DATA) =====")
        print("1. Show Dataset Info")
        print("2. Elbow Method")
        print("3. Apply K-Means Clustering")
        print("4. Visualize Clusters")
        print("5. PCA Visualization")
        print("6. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            show_data_info()

        elif choice == '2':
            elbow_method()

        elif choice == '3':
            apply_kmeans()

        elif choice == '4':
            visualize_clusters()

        elif choice == '5':
            pca_visualization()

        elif choice == '6':
            print("👋 Exiting...")
            break

        else:
            print(" Invalid choice!")



menu()
