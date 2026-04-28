import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import os
import warnings
import sys

if __name__ == "__main__":
    # Mengabaikan peringatan agar tampilan terminal lebih bersih
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # 1. Menangani pengambilan file dataset
    # Jika ada argumen ke-3 dari terminal, gunakan itu. Jika tidak, cari "train_pca.csv" di folder yang sama.
    if len(sys.argv) > 3:
        file_path = sys.argv[3]
    else:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_pca.csv")
    
    # Membaca data
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error: Gagal membaca file. Pastikan {file_path} ada di folder tersebut.")
        sys.exit(1)

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("Credit_Score", axis=1),
        data["Credit_Score"],
        random_state=42,
        test_size=0.2
    )
    input_example = X_train[0:5]

    # 3. Menangkap Parameter dari MLproject (sys.argv)
    # sys.argv[1] adalah n_estimators, sys.argv[2] adalah max_depth
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 37

    # 4. Proses MLflow
    # Set Tracking URI (Opsional jika dijalankan lewat 'mlflow run', tapi bagus untuk jaga-jaga)
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    #mlflow.set_experiment("Latihan Credit Scoring")

    with mlflow.start_run():
        # A. Log Parameter agar muncul di tabel Dashboard
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # B. Inisialisasi dan Training
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)

        # C. Log Model sebagai Artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

        # D. Log Metrics (Hasil ukur)
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

        # E. Print output ke terminal
        print("-" * 40)
        print("EKSEKUSI MLPROJECT BERHASIL")
        print(f"Parameter: n_estimators={n_estimators}, max_depth={max_depth}")
        print(f"Hasil Akurasi: {accuracy:.4f}")
        print("-" * 40)