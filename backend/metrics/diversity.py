"""
Diversity Metrics: Misura la variabilità dei dati sintetici
"""
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def get_numeric_feature_cols(df):
    """
    Restituisce la lista di colonne numeriche da usare per le metriche:
    - esclude colonne chiamate 'id'
    - esclude colonne con tutti valori unici (probabili identificatori)
    - esclude colonne costanti (nunique <= 1)
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    filtered = []
    for c in num_cols:
        name_lower = c.lower()
        nunique = df[c].nunique(dropna=True)
        if name_lower == 'id':
            continue
        if nunique == len(df):  # tutti unici -> probabile id
            continue
        if nunique <= 1:  # costante -> inutile per le metriche
            continue
        filtered.append(c)
    return filtered


def calculate_diversity_metrics(df_original, df_augmented):
    """
    Calcola tutte le metriche di diversity del dataset aumentato
    
    Args:
        df_original: DataFrame del dataset originale
        df_augmented: DataFrame del dataset aumentato
        
    Returns:
        dict: Dizionario con tutte le metriche di diversity
    """
    metrics = {}
    
    # Seleziona solo colonne numeriche comuni
    numeric_cols = df_original.select_dtypes(include=[np.number]).columns
    common_cols = numeric_cols.intersection(df_augmented.select_dtypes(include=[np.number]).columns)
    
    if len(common_cols) == 0:
        return {"error": "Nessuna colonna numerica comune trovata"}
    
    df_orig_num = df_original[common_cols].dropna()
    df_aug_num = df_augmented[common_cols].dropna()
    
    # 1. Feature Entropy (entropia delle distribuzioni)
    metrics['feature_entropy'] = calculate_feature_entropy(df_aug_num)
    
    # 2. Coverage (copertura dello spazio dei dati originali)
    metrics['coverage'] = calculate_coverage(df_orig_num, df_aug_num)
    
    # 3. Cluster Spread (distribuzione nei cluster)
    metrics['cluster_spread'] = calculate_cluster_spread(df_orig_num, df_aug_num)
    
    # 4. Intra-diversity (diversità interna al dataset aumentato)
    metrics['intra_diversity'] = calculate_intra_diversity(df_aug_num)
    
    # 5. Feature Range Coverage
    metrics['range_coverage'] = calculate_range_coverage(df_orig_num, df_aug_num)

    # 6. Silhouette Score
    metrics['silhouette_score'] = calculate_silhouette_score(df_orig_num, df_aug_num)

    # 7. Davies-Bouldin Index
    metrics['davies_bouldin_index'] = calculate_davies_bouldin(df_orig_num, df_aug_num)

    # 8. Intra-Class Compactness (ICC)
    metrics['intra_class_compactness'] = calculate_icc(df_orig_num, df_aug_num)
    
    return metrics


def calculate_feature_entropy(df):
    """Calcola l'entropia per ogni feature (maggiore = più diversità)"""
    entropies = {}
    eps = 1e-10

    for col in df.columns:
        arr = df[col].dropna().values
        # Se troppi pochi valori o varianza zero -> entropia = 0
        if arr.size < 2 or np.nanstd(arr) == 0:
            entropies[col] = 0.0
            continue

        # Usa density=False e normalizza manualmente (più robusto)
        hist, bin_edges = np.histogram(arr, bins=50, density=False)
        hist = hist.astype(float)
        total = hist.sum()
        if total == 0:
            entropies[col] = 0.0
            continue

        probs = hist / (total + eps)
        ent = entropy(probs)
        entropies[col] = float(ent)

    vals = list(entropies.values()) or [0.0]
    return {
        'average_entropy': float(np.mean(vals)),
        'total_entropy': float(np.sum(vals)),
        'per_feature': entropies,
        'interpretation': 'higher_is_better'
    }



def calculate_coverage(df_orig, df_aug):
    # Normalizza
    scaler = StandardScaler()
    X_orig = scaler.fit_transform(df_orig)
    X_aug = scaler.transform(df_aug)

    if len(X_aug) == 0 or len(X_orig) == 0:
        return {
            'coverage_ratio': 0.0,
            'average_distance_to_nearest': float('nan'),
            'max_distance_to_nearest': float('nan'),
            'threshold_used': float('nan'),
            'interpretation': 'higher_is_better (max=1.0)'
        }

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X_aug)
    distances, _ = nbrs.kneighbors(X_orig)
    # distances ha shape (n_orig, 1) -> appiattisci
    distances = distances.ravel()

    # Se tutti i valori sono uguali, threshold sarà quel valore
    threshold = float(np.percentile(distances, 90)) if distances.size > 0 else float('nan')
    coverage_ratio = float(np.mean(distances < threshold)) if distances.size > 0 else 0.0

    return {
        'coverage_ratio': float(coverage_ratio),
        'average_distance_to_nearest': float(np.mean(distances)) if distances.size > 0 else float('nan'),
        'max_distance_to_nearest': float(np.max(distances)) if distances.size > 0 else float('nan'),
        'threshold_used': float(threshold),
        'interpretation': 'higher_is_better (max=1.0)'
    }



def calculate_cluster_spread(df_orig, df_aug):
    """
    Analizza come il dataset aumentato si distribuisce nei cluster
    definiti dal dataset originale
    """
    # Normalizza
    scaler = StandardScaler()
    X_orig = scaler.fit_transform(df_orig)
    X_aug = scaler.transform(df_aug)
    
    # Trova cluster ottimali (2-10)
    n_clusters = min(8, len(df_orig) // 10, len(df_aug) // 10)
    n_clusters = max(2, n_clusters)
    
    # K-means sul dataset originale
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    orig_labels = kmeans.fit_predict(X_orig)
    aug_labels = kmeans.predict(X_aug)
    
    # Distribuzione nei cluster
    orig_dist = np.bincount(orig_labels, minlength=n_clusters) / len(orig_labels)
    aug_dist = np.bincount(aug_labels, minlength=n_clusters) / len(aug_labels)
    
    # Jensen-Shannon divergence tra distribuzioni
    from scipy.spatial.distance import jensenshannon
    js_div = jensenshannon(orig_dist, aug_dist)
    
    # Spread uniformità: quanto uniformemente è distribuito
    aug_entropy = entropy(aug_dist + 1e-10)
    max_entropy = np.log(n_clusters)
    uniformity = aug_entropy / max_entropy
    
    return {
        'cluster_distribution_similarity': float(1 - js_div),
        'augmented_uniformity': float(uniformity),
        'n_clusters': n_clusters,
        'original_distribution': orig_dist.tolist(),
        'augmented_distribution': aug_dist.tolist(),
        'interpretation': 'uniformity close to 1.0 means diverse spread'
    }


def calculate_intra_diversity(df):
    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    n_samples = min(500, len(X))
    if n_samples < 2:
        return {
            'mean_pairwise_distance': 0.0,
            'std_pairwise_distance': 0.0,
            'min_pairwise_distance': 0.0,
            'max_pairwise_distance': 0.0,
            'diversity_score': 0.0,
            'interpretation': 'higher_is_better'
        }

    indices = np.random.choice(len(X), n_samples, replace=False)
    X_sample = X[indices]

    from scipy.spatial.distance import pdist
    distances = pdist(X_sample, metric='euclidean')
    if distances.size == 0:
        return {
            'mean_pairwise_distance': 0.0,
            'std_pairwise_distance': 0.0,
            'min_pairwise_distance': 0.0,
            'max_pairwise_distance': 0.0,
            'diversity_score': 0.0,
            'interpretation': 'higher_is_better'
        }

    return {
        'mean_pairwise_distance': float(np.mean(distances)),
        'std_pairwise_distance': float(np.std(distances)),
        'min_pairwise_distance': float(np.min(distances)),
        'max_pairwise_distance': float(np.max(distances)),
        'diversity_score': float(np.mean(distances)),
        'interpretation': 'higher_is_better'
    }



def calculate_range_coverage(df_orig, df_aug):
    coverage = {}

    for col in df_orig.columns:
        orig_min, orig_max = df_orig[col].min(), df_orig[col].max()
        aug_min, aug_max = df_aug[col].min(), df_aug[col].max()

        orig_range = orig_max - orig_min
        if orig_range == 0:
            # se l'originale è costante, consideriamo coverage = 1 se augmented contiene quel valore
            if np.isnan(orig_min) or np.isnan(aug_min):
                coverage[col] = {
                    'coverage_ratio': 0.0,
                    'extension_below': 0.0,
                    'extension_above': 0.0,
                    'total_extension': 0.0
                }
            else:
                contains = (aug_min <= orig_min <= aug_max)
                coverage[col] = {
                    'coverage_ratio': 1.0 if contains else 0.0,
                    'extension_below': 0.0,
                    'extension_above': 0.0,
                    'total_extension': 0.0
                }
            continue

        overlap_min = max(orig_min, aug_min)
        overlap_max = min(orig_max, aug_max)
        overlap = max(0.0, overlap_max - overlap_min)

        coverage_ratio = overlap / orig_range
        extension_below = max(0.0, (orig_min - aug_min) / orig_range)
        extension_above = max(0.0, (aug_max - orig_max) / orig_range)

        coverage[col] = {
            'coverage_ratio': float(coverage_ratio),
            'extension_below': float(extension_below),
            'extension_above': float(extension_above),
            'total_extension': float(extension_below + extension_above)
        }

    per_vals = [c['coverage_ratio'] for c in coverage.values()] or [0.0]
    avg_coverage = float(np.mean(per_vals))
    avg_extension = float(np.mean([c['total_extension'] for c in coverage.values()] or [0.0]))

    return {
        'average_coverage': avg_coverage,
        'average_extension': avg_extension,
        'per_feature': coverage,
        'interpretation': 'coverage close to 1.0 is ideal, extension shows novelty'
    }



def calculate_silhouette_score(df_orig, df_aug):
    """
    Silhouette Score: Misura quanto bene i punti sono clusterizzati
    Range: [-1, 1], più alto = migliore separazione cluster
    """
    from sklearn.metrics import silhouette_score

    # Normalizza
    scaler = StandardScaler()
    X_orig = scaler.fit_transform(df_orig)
    X_aug = scaler.transform(df_aug)

    # Determina numero ottimale di cluster
    n_clusters = min(8, len(df_orig) // 10, len(df_aug) // 10)
    n_clusters = max(2, n_clusters)

    # K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    # Score per dataset originale
    labels_orig = kmeans.fit_predict(X_orig)
    silhouette_orig = silhouette_score(X_orig, labels_orig)

    # Score per dataset aumentato
    labels_aug = kmeans.predict(X_aug)
    silhouette_aug = silhouette_score(X_aug, labels_aug)

    # Differenza (quanto simile è la struttura dei cluster)
    similarity = 1 - abs(silhouette_orig - silhouette_aug)

    return {
        'original_score': float(silhouette_orig),
        'augmented_score': float(silhouette_aug),
        'similarity': float(similarity),
        'n_clusters': n_clusters,
        'interpretation': 'higher_is_better (range: -1 to 1)',
        'quality': 'high' if silhouette_aug > 0.5 else 'medium' if silhouette_aug > 0.25 else 'low'
    }


def calculate_davies_bouldin(df_orig, df_aug):
    """
    Davies-Bouldin Index: Misura la separazione tra cluster
    Range: [0, ∞], più basso = migliore separazione
    """
    from sklearn.metrics import davies_bouldin_score

    # Normalizza
    scaler = StandardScaler()
    X_orig = scaler.fit_transform(df_orig)
    X_aug = scaler.transform(df_aug)

    # Determina numero ottimale di cluster
    n_clusters = min(8, len(df_orig) // 10, len(df_aug) // 10)
    n_clusters = max(2, n_clusters)

    # K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    # Score per dataset originale
    labels_orig = kmeans.fit_predict(X_orig)
    db_orig = davies_bouldin_score(X_orig, labels_orig)

    # Score per dataset aumentato
    labels_aug = kmeans.predict(X_aug)
    db_aug = davies_bouldin_score(X_aug, labels_aug)

    # Similarità (quanto simile è la struttura)
    similarity = 1 / (1 + abs(db_orig - db_aug))

    return {
        'original_score': float(db_orig),
        'augmented_score': float(db_aug),
        'similarity': float(similarity),
        'difference': float(abs(db_orig - db_aug)),
        'n_clusters': n_clusters,
        'interpretation': 'lower_is_better (0 is perfect)',
        'quality': 'high' if db_aug < 1.0 else 'medium' if db_aug < 2.0 else 'low'
    }


def calculate_icc(df_orig, df_aug):
    """
    Intra-Class Compactness (ICC): Misura la compattezza dei cluster
    Più basso = cluster più compatti
    """
    # Normalizza
    scaler = StandardScaler()
    X_orig = scaler.fit_transform(df_orig)
    X_aug = scaler.transform(df_aug)

    # Determina numero ottimale di cluster
    n_clusters = min(8, len(df_orig) // 10, len(df_aug) // 10)
    n_clusters = max(2, n_clusters)

    # K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    # Calcola ICC per originale
    labels_orig = kmeans.fit_predict(X_orig)
    icc_orig = calculate_within_cluster_variance(X_orig, labels_orig)

    # Calcola ICC per aumentato
    labels_aug = kmeans.predict(X_aug)
    icc_aug = calculate_within_cluster_variance(X_aug, labels_aug)

    # Similarità
    similarity = 1 / (1 + abs(icc_orig - icc_aug))

    return {
        'original_icc': float(icc_orig),
        'augmented_icc': float(icc_aug),
        'similarity': float(similarity),
        'difference': float(abs(icc_orig - icc_aug)),
        'n_clusters': n_clusters,
        'interpretation': 'lower_icc_means_more_compact_clusters',
        'quality': 'high' if icc_aug < 0.5 else 'medium' if icc_aug < 1.0 else 'low'
    }


def calculate_within_cluster_variance(X, labels):
    """Helper: calcola la varianza intra-cluster media"""
    variances = []
    for label in np.unique(labels):
        cluster_points = X[labels == label]
        if len(cluster_points) > 1:
            variance = np.var(cluster_points)
            variances.append(variance)

    return np.mean(variances) if variances else 0.0
