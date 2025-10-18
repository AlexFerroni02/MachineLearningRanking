import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# 1. Caricamento dati
print("=" * 60)
print("CARICAMENTO DATI")
print("=" * 60)

try:
    # Carica i dati di training
    X_train_df = pd.read_csv('../data/X_train.csv')
    X_train = X_train_df.values
    y_train = pd.read_csv('../data/y_train.csv')['y'].values
    metadata_df = pd.read_csv('../data/training_metadata.csv')

    # Carica le feature columns
    feature_cols = X_train_df.columns.tolist()

    # Carica il dataset LOINC originale
    loinc_df = pd.read_excel('../data/loinc_dataset-v2.xlsx')

    print(f"✅ Dataset caricato con successo:")
    print(f"   - X_train shape: {X_train.shape}")
    print(f"   - y_train shape: {y_train.shape}")
    print(f"   - Feature: {len(feature_cols)}")
    print(f"   - Dataset LOINC: {len(loinc_df)} documenti")
    print(f"   - Metadati: {len(metadata_df)} coppie")

except FileNotFoundError as e:
    print(f"❌ ERRORE: File non trovato - {e}")
    print("Controlla che tutti i file siano nella directory ../data/")
    exit(1)

# 2. Training del modello pairwise (come nel paper di Joachims)
print("\n" + "=" * 60)
print("TRAINING MODELLO PAIRWISE - JOACHIMS APPROACH")
print("=" * 60)

# Split train/test come nel paper
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Training set: {X_train_split.shape[0]} coppie")
print(f"Test set: {X_test_split.shape[0]} coppie")

# Modelli da testare (SVM è quello del paper originale)
models = {
    'SVM_RBF': SVC(kernel='rbf', random_state=42, probability=True),
    'SVM_Linear': SVC(kernel='linear', random_state=42, probability=True),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
}

best_model = None
best_test_acc = 0
results = {}

for name, model in models.items():
    print(f"\n--- Training {name} ---")

    # Training
    model.fit(X_train_split, y_train_split)

    # Predizioni
    train_pred = model.predict(X_train_split)
    test_pred = model.predict(X_test_split)

    # Accuratezza
    train_acc = accuracy_score(y_train_split, train_pred)
    test_acc = accuracy_score(y_test_split, test_pred)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_split, y_train_split, cv=5)

    results[name] = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'model': model
    }

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_model = model

# 3. Selezione del modello finale
print("\n" + "=" * 60)
print("SELEZIONE MODELLO FINALE")
print("=" * 60)

print("Risultati completi:")
print("-" * 40)
for name, result in results.items():
    print(f"{name:18}: Train={result['train_accuracy']:.4f} | "
          f"Test={result['test_accuracy']:.4f} | "
          f"CV={result['cv_mean']:.4f}±{result['cv_std']:.4f}")

# Criterio di selezione: Test Accuracy
best_model_name = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
best_model = results[best_model_name]['model']

print(f"\n🏆 MODELLO SELEZIONATO: {best_model_name}")
print(f"   Test Accuracy: {best_test_acc:.4f}")

# Analisi SVM (come nel paper di Joachims)
svm_rbf_results = results.get('SVM_RBF', None)
if svm_rbf_results:
    print(f"\n📊 SVM RBF (paper originale): Test={svm_rbf_results['test_accuracy']:.4f}")

# Salva modelli
os.makedirs('../models', exist_ok=True)
joblib.dump(best_model, '../models/joachims_ranking_model.pkl')

print(f"\n💾 Modello salvato: joachims_ranking_model.pkl")


# 4. Funzione di ranking (cuore del paper di Joachims)
def extract_features_for_query(query_text, documents_df):
    """Estrae le stesse feature usate nel training"""
    query_terms = query_text.lower().split()
    df = documents_df.copy()

    # Feature base
    df['len_name'] = df['long_common_name'].astype(str).str.len()
    df['word_count'] = df['long_common_name'].astype(str).str.split().str.len()

    loinc_columns = ['long_common_name', 'component', 'system', 'property']

    for col in loinc_columns:
        df[col] = df[col].fillna('').astype(str)

        df[f'{col}_query_terms'] = df[col].str.lower().apply(
            lambda x: sum(1 for term in query_terms if term in x)
        )
        df[f'{col}_len'] = df[col].str.len()
        df[f'{col}_words'] = df[col].str.split().str.len()

        try:
            texts = [query_text] + df[col].tolist()
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            similarities = cosine_similarity(tfidf_matrix[1:], tfidf_matrix[0:1]).flatten()
            df[f'{col}_similarity'] = similarities
        except:
            df[f'{col}_similarity'] = 0.0

    return df


def rank_documents_joachims(query_text, documents_df, model, feature_cols):
    """
    Implementa il ranking come nel paper di Joachims:
    Per ogni coppia di documenti, predice quale è preferito
    Il ranking finale è basato su quante "vittorie" ha ogni documento
    """
    print(f"Ranking documenti per query: '{query_text}'")

    # Estrai feature
    df = extract_features_for_query(query_text, documents_df)

    # Algoritmo di ranking pairwise (Joachims method)
    n_docs = len(df)
    doc_scores = np.zeros(n_docs)

    print(f"Confrontando {n_docs} documenti in {n_docs * (n_docs - 1) // 2} coppie...")

    for i in range(n_docs):
        for j in range(i + 1, n_docs):
            # Estrai feature per coppia i,j
            doc_i_features = df.iloc[i][feature_cols].values.astype(float)
            doc_j_features = df.iloc[j][feature_cols].values.astype(float)

            # Differenza di feature (come nel training)
            diff_features = doc_i_features - doc_j_features

            # Predizione: 1 se doc_i preferito a doc_j
            preference = model.predict([diff_features])[0]
            confidence = model.predict_proba([diff_features])[0][1]

            if preference == 1:  # doc_i vince
                doc_scores[i] += confidence
            else:  # doc_j vince
                doc_scores[j] += (1 - confidence)

    # Crea ranking finale
    ranking_results = []
    for i in range(n_docs):
        ranking_results.append({
            'rank': 0,  # sarà calcolato dopo
            'loinc_num': df.iloc[i]['loinc_num'],
            'long_common_name': df.iloc[i]['long_common_name'],
            'component': df.iloc[i]['component'],
            'system': df.iloc[i]['system'],
            'property': df.iloc[i]['property'],
            'score': doc_scores[i],
            'long_common_name_similarity': df.iloc[i]['long_common_name_similarity']
        })

    # Ordina per score e assegna rank
    ranking_df = pd.DataFrame(ranking_results)
    ranking_df = ranking_df.sort_values('score', ascending=False).reset_index(drop=True)
    ranking_df['rank'] = range(1, len(ranking_df) + 1)

    return ranking_df


# 5. Test del ranking con query di esempio
print("\n" + "=" * 60)
print("TEST RANKING - JOACHIMS METHOD")
print("=" * 60)

# Crea directory per i risultati
os.makedirs('../results', exist_ok=True)

# Query di test (puoi modificare queste)
test_queries = [
    "glucose blood test",
    "hemoglobin measurement",
    "cholesterol level",
    "protein urine"
]

all_rankings = {}

for query_text in test_queries:
    print(f"\n📊 Ranking per query: '{query_text}'")

    # Prendi un sottocampione del dataset per il test (primi 50 documenti)
    test_docs = loinc_df.head(50).copy()

    ranked_docs = rank_documents_joachims(query_text, test_docs, best_model, feature_cols)
    all_rankings[query_text] = ranked_docs

    print(f"Top 5 documenti:")
    top5 = ranked_docs.head(5)[['rank', 'loinc_num', 'long_common_name', 'score', 'long_common_name_similarity']]
    for _, row in top5.iterrows():
        print(f"  {row['rank']:2d}. {row['loinc_num']} - {row['long_common_name'][:60]}...")
        print(f"      Score: {row['score']:.3f}, Similarity: {row['long_common_name_similarity']:.3f}")

    # Salva risultati per questa query
    safe_query_name = query_text.replace(' ', '_').replace('/', '_')
    ranked_docs.to_csv(f"../results/ranking_{safe_query_name}.csv", index=False)

# 6. Analisi feature importance
if hasattr(best_model, 'feature_importances_'):
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("Top 10 feature più importanti:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:35} : {row['importance']:.4f}")

    feature_importance.to_csv('../results/feature_importance_joachims.csv', index=False)

# 7. Confronto con baseline
print("\n" + "=" * 60)
print("CONFRONTO CON BASELINE")
print("=" * 60)

comparison_results = []

for query_text in test_queries:
    ml_ranking = all_rankings[query_text]

    # Baseline: ranking solo per similarità TF-IDF
    baseline_docs = loinc_df.head(50).copy()
    baseline_docs = extract_features_for_query(query_text, baseline_docs)
    baseline_ranking = baseline_docs.sort_values(
        'long_common_name_similarity', ascending=False
    ).reset_index(drop=True)

    print(f"\nQuery: '{query_text}'")
    print("Top 3 ML Ranking vs Baseline:")

    for i in range(min(3, len(ml_ranking))):
        ml_doc = ml_ranking.iloc[i]
        baseline_doc = baseline_ranking.iloc[i]

        print(f"{i + 1}. ML: {ml_doc['loinc_num']} ({ml_doc['score']:.3f}) | "
              f"Baseline: {baseline_doc['loinc_num']} ({baseline_doc['long_common_name_similarity']:.3f})")

        comparison_results.append({
            'query': query_text,
            'rank': i + 1,
            'ml_loinc': ml_doc['loinc_num'],
            'ml_score': ml_doc['score'],
            'baseline_loinc': baseline_doc['loinc_num'],
            'baseline_score': baseline_doc['long_common_name_similarity']
        })

# Salva confronto
comparison_df = pd.DataFrame(comparison_results)
comparison_df.to_csv('../results/ml_vs_baseline_comparison.csv', index=False)

# 8. Statistiche finali
print("\n" + "=" * 60)
print("JOACHIMS RANKING SYSTEM COMPLETATO!")
print("=" * 60)

print(f"📊 Statistiche finali:")
print(f"   - Modello utilizzato: {best_model_name}")
print(f"   - Test Accuracy: {best_test_acc:.4f}")
print(f"   - Query processate: {len(test_queries)}")
print(f"   - Documenti testati: 50 per query")
print(f"   - Feature utilizzate: {len(feature_cols)}")

print(f"\n📁 File generati:")
print(f"   - ../models/joachims_ranking_model.pkl")
print(f"   - ../results/ranking_*.csv (uno per query)")
print(f"   - ../results/feature_importance_joachims.csv")
print(f"   - ../results/ml_vs_baseline_comparison.csv")

print(f"\n✅ Implementazione del paper di Joachims (2002) completata!")
