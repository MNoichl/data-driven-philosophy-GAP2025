from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
import numpy as np

# Initialize KeyBERT with our embedding model

vectorizer = KeyphraseCountVectorizer()
kw = KeyBERT(model=model)
print('KeyBERT initialized')

# Get unique cluster labels
unique_clusters = np.unique(cluster_labels)

# Dictionary to store keywords for each cluster
cluster_keywords = {}

for cluster_id in unique_clusters:
    # Get indices of documents in this cluster
    cluster_mask = cluster_labels == cluster_id
    cluster_docs = [text_data[i][0:3000] for i in range(len(text_data)) if cluster_mask[i]]
    cluster_embeddings = embeddings[cluster_mask]
    
    # Compute centroid embedding for this cluster
    centroid = np.mean(cluster_embeddings, axis=0)
    print('Centroid computed')
    # Create doc_embeddings array where each doc is represented by the cluster centroid
    doc_embs = np.tile(centroid, (len(cluster_docs), 1))
    
    print('Doc embeddings created')
    # Extract keywords for this cluster
    keywords = kw.extract_keywords(
        cluster_docs,
        vectorizer=vectorizer,
        stop_words="english",
        doc_embeddings=doc_embs,
        use_mmr=True, 
        diversity=0.4, 
        top_n=10
    )
    
    # Combine keywords from all documents in this cluster
    keyword_scores = {}
    for doc_keywords in keywords:
        for keyword, score in doc_keywords:
            if keyword in keyword_scores:
                keyword_scores[keyword] += score
            else:
                keyword_scores[keyword] = score
    
    # Sort by combined scores and keep top keywords
    combined_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)[:50]
    
    cluster_keywords[cluster_id] = ', '.join([x[0] for x in combined_keywords])
    print(f"Cluster {cluster_id} ({np.sum(cluster_mask)} docs):")
    print(f"  Top keywords: {cluster_keywords[cluster_id]}")
    print()