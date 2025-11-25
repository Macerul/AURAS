"""
Fidelity Metrics: Misura quanto i dati sintetici somigliano a quelli reali
"""
import numpy as np
import warnings
import pandas as pd
from scipy import stats
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance, ks_2samp


def calculate_fidelity_metrics(df_original, df_augmented):
    """
    Calcola tutte le metriche di fidelity tra dataset originale e aumentato
    
    Args:
        df_original: DataFrame del dataset originale
        df_augmented: DataFrame del dataset aumentato
        
    Returns:
        dict: Dizionario con tutte le metriche di fidelity
    """
    metrics = {}
    
    # Seleziona solo colonne numeriche comuni
    numeric_cols = df_original.select_dtypes(include=[np.number]).columns
    common_cols = numeric_cols.intersection(df_augmented.select_dtypes(include=[np.number]).columns)
    
    if len(common_cols) == 0:
        return {"error": "Nessuna colonna numerica comune trovata"}
    
    df_orig_num = df_original[common_cols].dropna()
    df_aug_num = df_augmented[common_cols].dropna()
    
    # 1. Mean Absolute Distance (distanza media tra distribuzioni)
    metrics['mean_absolute_distance'] = calculate_mean_distance(df_orig_num, df_aug_num)
    
    # 2. Kolmogorov-Smirnov Test (per ogni feature)
    metrics['kolmogorov_smirnov'] = calculate_ks_test(df_orig_num, df_aug_num)
    
    # 3. Maximum Mean Discrepancy (MMD)
    metrics['mmd_score'] = calculate_mmd(df_orig_num, df_aug_num)
    
    # 4. PCA Embedding Comparison
    metrics['pca_comparison'] = calculate_pca_similarity(df_orig_num, df_aug_num)
    
    # 5. Statistical Moments Comparison
    metrics['statistical_moments'] = calculate_statistical_moments(df_orig_num, df_aug_num)

    # 6. KL Divergence
    metrics['kl_divergence'] = calculate_kl_divergence(df_orig_num, df_aug_num)

    # 7. Jensen-Shannon Divergence
    metrics['js_divergence'] = calculate_js_divergence(df_orig_num, df_aug_num)

    # 8. Q-Function Multi-Attributes Similarity
    metrics['q_function'] = calculate_q_function(df_orig_num, df_aug_num)
    
    return metrics


def calculate_q_function(df_orig, df_aug):
    """
    Q-Function Multi-Attributes Similarity:
    Misura la similarità multi-attributo per valutare il rischio di privacy
    considerando combinazioni di attributi
    """
    # Normalizza
    scaler = StandardScaler()
    X_orig = scaler.fit_transform(df_orig)
    X_aug = scaler.transform(df_aug)

    n_features = X_orig.shape[1]
    q_scores = []

    # Calcola Q-function per diverse combinazioni di attributi
    # Consideriamo coppie, triple e quadruple di attributi

    from itertools import combinations

    # Pairwise attributes (coppie)
    for i, j in combinations(range(min(n_features, 5)), 2):
        X_orig_pair = X_orig[:, [i, j]]
        X_aug_pair = X_aug[:, [i, j]]

        # Per ogni punto aumentato, trova il più vicino nell'originale
        nbrs = NearestNeighbors(n_neighbors=1).fit(X_orig_pair)
        distances, _ = nbrs.kneighbors(X_aug_pair)

        # Q-score: media delle distanze inverse (più vicino = più rischioso)
        q_score = 1 / (1 + np.mean(distances))
        q_scores.append(q_score)

    # Triple attributes se ci sono abbastanza feature
    if n_features >= 3:
        for combo in list(combinations(range(min(n_features, 5)), 3))[:5]:
            X_orig_triple = X_orig[:, combo]
            X_aug_triple = X_aug[:, combo]

            nbrs = NearestNeighbors(n_neighbors=1).fit(X_orig_triple)
            distances, _ = nbrs.kneighbors(X_aug_triple)

            q_score = 1 / (1 + np.mean(distances))
            q_scores.append(q_score)

    # Media dei Q-scores
    avg_q_score = np.mean(q_scores)

    privacy_score = avg_q_score
    print(privacy_score)

    return {
        'q_score': float(avg_q_score),
        'privacy_score': float(privacy_score),
        'n_combinations_tested': len(q_scores),
        'min_q_score': float(np.min(q_scores)),
        'max_q_score': float(np.max(q_scores)),
        #'interpretation': 'lower_q_score_means_better_privacy',
        #'risk_level': 'low' if avg_q_score < 0.3 else 'medium' if avg_q_score > 0.6 else 'high',
        'privacy_quality': 'high' if privacy_score > 0.7 else 'medium' if privacy_score > 0.4 else 'low'
    }


def calculate_mean_distance(df_orig, df_aug):
    """Calcola la distanza media tra le distribuzioni"""
    distances = []
    for col in df_orig.columns:
        mean_orig = df_orig[col].mean()
        mean_aug = df_aug[col].mean()
        std_orig = df_orig[col].std()
        std_aug = df_aug[col].std()
        
        # Distanza normalizzata
        dist = abs(mean_orig - mean_aug) / (std_orig + 1e-10)
        distances.append(dist)
    
    return {
        'overall_distance': float(np.mean(distances)),
        'max_distance': float(np.max(distances)),
        'min_distance': float(np.min(distances)),
        'per_feature': {col: float(dist) for col, dist in zip(df_orig.columns, distances)}
    }


def calculate_ks_test(df_orig, df_aug):
    """Kolmogorov-Smirnov test per ogni feature"""
    ks_results = {}

    for col in df_orig.columns:
        statistic, pvalue = stats.ks_2samp(df_orig[col], df_aug[col])
        ks_results[col] = {
            'statistic': float(statistic),
            'pvalue': float(pvalue),
            'similar': bool(pvalue > 0.05)  # <-- CAMBIA QUESTA RIGA
        }

    # Score aggregato (media delle p-values)
    avg_pvalue = np.mean([r['pvalue'] for r in ks_results.values()])

    return {
        'average_pvalue': float(avg_pvalue),
        'similarity_score': float(avg_pvalue),  # Più alto = più simile
        'per_feature': ks_results
    }


def calculate_mmd(df_orig, df_aug, kernel='rbf', gamma=1.0):
    """
    Maximum Mean Discrepancy - misura la distanza tra distribuzioni
    usando kernel methods
    """
    # Campiona se i dataset sono troppo grandi
    n_samples = min(1000, len(df_orig), len(df_aug))
    
    X = df_orig.sample(n=n_samples, random_state=42).values
    Y = df_aug.sample(n=n_samples, random_state=42).values
    
    # Normalizza
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Y = scaler.transform(Y)
    
    # Calcola kernel RBF
    def rbf_kernel(X, Y, gamma=1.0):
        sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)
        return np.exp(-gamma * sq_dists)
    
    XX = rbf_kernel(X, X, gamma)
    YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)
    
    mmd_squared = XX.mean() + YY.mean() - 2 * XY.mean()
    mmd = np.sqrt(max(mmd_squared, 0))
    
    return {
        'mmd_score': float(mmd),
        'interpretation': 'lower_is_better',
        'quality': 'high' if mmd < 0.1 else 'medium' if mmd < 0.3 else 'low'
    }


def calculate_pca_similarity(df_orig, df_aug):
    """Confronta le rappresentazioni PCA dei due dataset"""
    # Normalizza
    scaler = StandardScaler()
    X_orig = scaler.fit_transform(df_orig)
    X_aug = scaler.transform(df_aug)

    # PCA
    n_components = min(5, X_orig.shape[1])
    pca = PCA(n_components=n_components)

    pca_orig = pca.fit_transform(X_orig)
    pca_aug = pca.transform(X_aug)

    # Confronta le distribuzioni delle componenti principali
    similarities = []
    for i in range(n_components):
        # Usa KS test per confrontare le distribuzioni
        statistic, pvalue = stats.ks_2samp(pca_orig[:, i], pca_aug[:, i])
        # Converti p-value in similarity score (più alto = più simile)
        similarity = pvalue
        similarities.append(similarity)

        # In alternativa, confronta momenti statistici
        # mean_diff = abs(np.mean(pca_orig[:, i]) - np.mean(pca_aug[:, i]))
        # std_diff = abs(np.std(pca_orig[:, i]) - np.std(pca_aug[:, i]))
        # similarity = 1 / (1 + mean_diff + std_diff)
        # similarities.append(similarity)

    return {
        'average_component_similarity': float(np.mean(similarities)),
        'explained_variance': pca.explained_variance_ratio_.tolist(),
        'component_similarities': [float(s) for s in similarities],
        'n_components': n_components
    }


def calculate_statistical_moments(
    df_orig,
    df_aug,
    weights=None,
    eps=1e-8,
    include_distribution_metrics=True
):
    """
    Confronta colonne di df_orig vs df_aug e restituisce:
      - metriche raw (mean_diff, std_diff, skew_diff, kurt_diff)
      - metriche normalizzate / standardizzate
      - wasserstein & ks (opzionali)
      - composite_score per feature e aggregati globali

    weights: dict con pesi per 'mean','std','skew','kurt' (sommano idealmente a 1)
    """
    if weights is None:
        weights = {'mean': 0.4, 'std': 0.2, 'skew': 0.2, 'kurt': 0.2}
    # fallback normalization lower bound
    results = {}
    composite_list = []

    for col in df_orig.columns:
        orig_col = df_orig[col].dropna()
        aug_col = df_aug[col].dropna()

        # basic moments with safe checks
        mean_orig = float(orig_col.mean()) if len(orig_col) > 0 else 0.0
        mean_aug = float(aug_col.mean()) if len(aug_col) > 0 else 0.0
        std_orig = float(orig_col.std(ddof=0)) if len(orig_col) > 0 else 0.0
        std_aug = float(aug_col.std(ddof=0)) if len(aug_col) > 0 else 0.0

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            try:
                skew_orig = float(stats.skew(orig_col)) if len(orig_col) > 2 else 0.0
            except:
                skew_orig = 0.0
            try:
                skew_aug = float(stats.skew(aug_col)) if len(aug_col) > 2 else 0.0
            except:
                skew_aug = 0.0
            try:
                kurt_orig = float(stats.kurtosis(orig_col)) if len(orig_col) > 3 else 0.0
            except:
                kurt_orig = 0.0
            try:
                kurt_aug = float(stats.kurtosis(aug_col)) if len(aug_col) > 3 else 0.0
            except:
                kurt_aug = 0.0

        # raw diffs
        mean_diff_raw = abs(mean_orig - mean_aug)
        std_diff_raw = abs(std_orig - std_aug)
        skew_diff_raw = abs(skew_orig - skew_aug)
        kurt_diff_raw = abs(kurt_orig - kurt_aug)

        # pooled std (per Cohen's d)
        pooled_std = np.sqrt((std_orig**2 + std_aug**2) / 2.0)
        pooled_std = max(pooled_std, eps)

        # Standardized / normalized metrics
        cohens_d = abs(mean_orig - mean_aug) / pooled_std  # already scale-free
        std_rel = std_diff_raw / pooled_std

        # normalize skew and kurt to a reasonable scale:
        # divide by max(abs(value), 1.0) to avoid inflating tiny denominators
        skew_scale = max(abs(skew_orig), abs(skew_aug), 1.0)
        kurt_scale = max(abs(kurt_orig), abs(kurt_aug), 1.0)
        skew_norm = skew_diff_raw / skew_scale
        kurt_norm = kurt_diff_raw / kurt_scale

        # distributional distances (optional)
        w_dist = None
        ks_stat = None
        if include_distribution_metrics and len(orig_col) > 0 and len(aug_col) > 0:
            try:
                w_dist = float(wasserstein_distance(orig_col, aug_col))
            except:
                w_dist = None
            try:
                ks_stat = float(ks_2samp(orig_col, aug_col).statistic)
            except:
                ks_stat = None

        # composite score (weighted sum of normalized components)
        composite = (
            weights.get('mean', 0.0) * cohens_d +
            weights.get('std', 0.0) * std_rel +
            weights.get('skew', 0.0) * skew_norm +
            weights.get('kurt', 0.0) * kurt_norm
        )

        results[col] = {
            'raw': {
                'mean_diff': mean_diff_raw,
                'std_diff': std_diff_raw,
                'skew_diff': skew_diff_raw,
                'kurt_diff': kurt_diff_raw
            },
            'normalized': {
                'cohens_d': cohens_d,
                'std_rel': std_rel,
                'skew_norm': skew_norm,
                'kurt_norm': kurt_norm
            },
            'distributional': {
                'wasserstein': w_dist,
                'ks_stat': ks_stat
            },
            'composite_score': float(composite)
        }

        composite_list.append(composite)

    # Aggregati globali sui composite score
    if len(composite_list) == 0:
        global_mean = 0.0
        global_max = 0.0
    else:
        global_mean = float(np.mean(composite_list))
        global_max = float(np.max(composite_list))

    return {
        'per_feature': results,
        'average_moment_difference': global_mean,
        'max_moment_difference': global_max,
        'stat_badge': 'high' if global_mean <= 0.2 else 'low' if global_mean >= 0.5 else 'medium'
    }


def calculate_kl_divergence(df_orig, df_aug):
    from scipy.special import kl_div
    eps = 1e-10
    kl_divs = {}

    for col in df_orig.columns:
        orig = df_orig[col].dropna().values
        aug = df_aug[col].dropna().values

        # Se pochi dati o varianza zero -> salta o assegna 0
        if orig.size < 2 or aug.size < 2 or np.nanstd(orig) == 0 or np.nanstd(aug) == 0:
            kl_divs[col] = 0.0
            continue

        bins = 50
        hist_orig, bin_edges = np.histogram(orig, bins=bins, density=False)
        hist_aug, _ = np.histogram(aug, bins=bin_edges, density=False)

        hist_orig = hist_orig.astype(float) + eps
        hist_aug = hist_aug.astype(float) + eps

        hist_orig /= hist_orig.sum()
        hist_aug /= hist_aug.sum()

        kl = np.sum(kl_div(hist_orig, hist_aug))
        kl_divs[col] = float(kl)

    vals = list(kl_divs.values()) or [0.0]
    avg_kl = float(np.mean(vals))

    return {
        'average_kl_divergence': avg_kl,
        'per_feature': kl_divs,
        'interpretation': 'lower_is_better (0 is identical)',
        'similarity_score': float(1 / (1 + avg_kl)),
        'quality': 'high' if avg_kl < 0.1 else 'medium' if avg_kl < 0.5 else 'low'
    }


def calculate_js_divergence(df_orig, df_aug):
    from scipy.spatial.distance import jensenshannon
    eps = 1e-10
    js_divs = {}

    for col in df_orig.columns:
        orig = df_orig[col].dropna().values
        aug = df_aug[col].dropna().values

        if orig.size < 2 or aug.size < 2 or np.nanstd(orig) == 0 or np.nanstd(aug) == 0:
            js_divs[col] = 0.0
            continue

        bins = 50
        hist_orig, bin_edges = np.histogram(orig, bins=bins, density=False)
        hist_aug, _ = np.histogram(aug, bins=bin_edges, density=False)

        hist_orig = hist_orig.astype(float) + eps
        hist_aug = hist_aug.astype(float) + eps

        hist_orig /= hist_orig.sum()
        hist_aug /= hist_aug.sum()

        js = jensenshannon(hist_orig, hist_aug)
        # jensenshannon può talvolta ritornare nan se input non validi - salvaguarda
        js_divs[col] = float(js) if np.isfinite(js) else 0.0

    vals = list(js_divs.values()) or [0.0]
    avg_js = float(np.mean(vals))

    return {
        'average_js_divergence': avg_js,
        'per_feature': js_divs,
        'interpretation': 'lower_is_better (0 is identical, max is 1)',
        'similarity_score': float(max(0.0, 1 - avg_js)),
        'quality': 'high' if avg_js < 0.1 else 'medium' if avg_js < 0.3 else 'low'
    }

