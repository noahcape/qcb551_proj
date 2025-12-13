from main import parse_embeddings_and_type, plot_umap_structural
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

for model_name in ['esm2_8M', 'esm2_35M', 'esm2_150M', 'prot_bert']:
    df = parse_embeddings_and_type(f"./data/{model_name}.csv")
    #print(df['type'].value_counts())

    N_PER_CLASS = 200
    sampled_df = (
        df.groupby("type", group_keys=False)
        .apply(lambda x: x.sample(n=min(N_PER_CLASS, len(x)), random_state=42))
        .reset_index(drop=True)
    )
    print(sampled_df)

    # Train supervised models
    x_full = np.array(df['embedding'].tolist())
    y_full = df['type'].tolist()
    unique_strings = sorted(list(set(y_full)))
    string_to_int_map = {name: uid for uid, name in enumerate(unique_strings)}
    y_full = [string_to_int_map[name] for name in y_full]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        x_full, y_full, test_size=0.2, random_state=42
    )

    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Biophysical features Accuracy RF: {accuracy:.2f}")

    rf_model = SVC(random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Biophysical features Accuracy SVC: {accuracy:.2f}")

    rf_model = LogisticRegression(random_state=42, max_iter=10000)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Biophysical features Accuracy LR: {accuracy:.2f}")