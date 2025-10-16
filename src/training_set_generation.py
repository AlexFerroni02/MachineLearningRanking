import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

# Configuration paths
DOCS_FILE = "../data/loinc_dataset-v2.xlsx"      # Excel file with our documents (dataset)
QUERIES_FILE = "../data/queries.xlsx"            # Excel file with queries (first run create_query_file.py)
OUTPUT_PAIRS_FILE = "../data/pairwise_tfidf_ranking.xlsx"

# combine LOINC fields into text 
def combine_doc_text(df):
    return df.apply(lambda row: " ".join([
        str(row.get('component', '')),
        str(row.get('system', '')),
        str(row.get('property', ''))
    ]).strip(), axis=1)

#  Given a query, it ranks with tf-idf and cosine similarity all the documents wrt to that query
def rank_documents_for_query(query_str, docs_df):

    doc_texts = combine_doc_text(docs_df)
    
    # Fit TF-IDF on query + documents (handled as a list)
    corpus = [query_str] + doc_texts.tolist()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # The query is on the first column, the docs on all others
    query_vec = tfidf_matrix[0]
    doc_vecs = tfidf_matrix[1:]
    
    similarities = cosine_similarity(query_vec, doc_vecs)[0]
    
    ranked_docs = docs_df.copy()
    ranked_docs['similarity'] = similarities
    ranked_docs = ranked_docs.sort_values(by='similarity', ascending=False).reset_index(drop=True)
    
    return ranked_docs

# After computing the rankings we can "simulate" through the rankings the clicks on one document or another
# (higher ranking = it gets clicked)
# but first we must create all the possible pairs. We only keep the pairs where the rankings are different
# and where the first doc has a higher ranking than the second (keeping also the other ones would 
# just give us duplicate data)
# We must also keep track for all the generated pairs of which query has generated their ranking, so that 
# we'll save this information in the training dataset
def create_pairwise_from_ranking(ranked_docs, query):
    pairs = []
    for i, j in combinations(range(len(ranked_docs)), 2):
        d1 = ranked_docs.iloc[i]
        d2 = ranked_docs.iloc[j]

        if d1['similarity'] == d2['similarity']:
            continue
    
        pair = {
            'query': query,   # track which query generated the pair
            'doc1_id': d1['loinc_num'],
            'doc2_id': d2['loinc_num'],
            'doc1_text': " ".join([str(d1.get('component', '')),
                                   str(d1.get('system', '')),
                                   str(d1.get('property', ''))]),
            'doc2_text': " ".join([str(d2.get('component', '')),
                                   str(d2.get('system', '')),
                                   str(d2.get('property', ''))]),
            'doc1_similarity': d1['similarity'],
            'doc2_similarity': d2['similarity'],
            'label': +1
        }
        pairs.append(pair)
    return pairs

# Main: loads documents and queries from excel, calls the functions we previoulsy defined
# computing for each query the ranking in all documents and geenrating all pairs for each query
# Then it saves the new dataset
def main():
    df_docs = pd.read_excel(DOCS_FILE, dtype=str, header=2)
    
    df_queries = pd.read_excel(QUERIES_FILE, dtype=str)
    queries = df_queries['query'].tolist()
    
    all_pairs = []
    
    for query in queries:
        print(f"Processing query: {query}")
        ranked_docs = rank_documents_for_query(query, df_docs)
        pairs = create_pairwise_from_ranking(ranked_docs, query)  
        all_pairs.extend(pairs)
    
    # Save the pairwise training dataset for the SVM to Excel
    pair_df = pd.DataFrame(all_pairs)
    pair_df.to_excel(OUTPUT_PAIRS_FILE, index=False)
    print(f"Saved pairwise dataset with {len(pair_df)} pairs to {OUTPUT_PAIRS_FILE}")

if __name__ == "__main__":
    main()
