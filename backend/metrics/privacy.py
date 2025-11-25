"""
Privacy Metrics: Valuta il rischio di re-identificazione
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def calculate_privacy_metrics(df_original, df_augmented):
    """
    Calcola tutte le metriche di privacy tra dataset originale e aumentato
    
    Args:
        df_original: DataFrame del dataset originale
        df_augmented: DataFrame del dataset aumentato
        
    Returns:
        dict: Dizionario con tutte le metriche di privacy
    """
    metrics = {}
    
    # Seleziona solo colonne numeriche comuni
    numeric_cols = df_original.select_dtypes(include=[np.number]).columns
    common_cols = numeric_cols.intersection(df_augmented.select_dtypes(include=[np.number]).columns)
    
    if len(common_cols) == 0:
        return {"error": "Nessuna colonna numerica comune trovata"}
    
    df_orig_num = df_original[common_cols].dropna()
    df_aug_num = df_augmented[common_cols].dropna()
    
    # 1. Nearest Neighbor Distance Ratio (NNDR)
    metrics['nearest_neighbor_risk'] = calculate_nn_disclosure_risk(df_orig_num, df_aug_num)
    
    # 2. Membership Inference Score
    metrics['membership_inference'] = calculate_membership_inference(df_orig_num, df_aug_num)
    
    # 3. Attribute Disclosure Risk
    metrics['attribute_disclosure'] = calculate_attribute_disclosure(df_orig_num, df_aug_num)
    
    # 4. Distance to Closest Record (DCR)
    metrics['distance_to_closest'] = calculate_dcr(df_orig_num, df_aug_num)
    
    # 5. Uniqueness Score
    metrics['uniqueness'] = calculate_uniqueness_score(df_orig_num, df_aug_num)

    # 7. k-anonymity (usa le colonne numeriche comuni come QI)
    metrics['k_anonymity'] = calculate_k_anonymity(df_orig_num, df_aug_num)

    # 8. l-diversity (default: ultimo attributo numerico come sensitivo)
    metrics['l_diversity'] = calculate_l_diversity(df_orig_num, df_aug_num)
    
    return metrics


def calculate_nn_disclosure_risk(df_orig, df_aug):
    """
    Nearest Neighbor Disclosure Risk:
    Misura quanto facilmente un record aumentato può essere collegato
    a un record originale
    """
    # Normalizza
    scaler = StandardScaler()
    X_orig = scaler.fit_transform(df_orig)
    X_aug = scaler.transform(df_aug)
    
    # Per ogni record aumentato, trova i 2 vicini più prossimi nel dataset originale
    k = min(5, len(X_orig))
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X_orig)
    distances, indices = nbrs.kneighbors(X_aug)
    
    # NNDR: rapporto tra prima e seconda distanza (più basso = più rischio)
    nn_ratios = distances[:, 0] / (distances[:, 1] + 1e-10)
    
    # Risk score: percentuale di record con NNDR < threshold
    risk_threshold = 0.5
    high_risk_ratio = np.mean(nn_ratios < risk_threshold)
    
    return {
        'average_nn_ratio': float(np.mean(nn_ratios)),
        'high_risk_ratio': float(high_risk_ratio),
        'min_distance_to_original': float(np.min(distances[:, 0])),
        'average_distance_to_nearest': float(np.mean(distances[:, 0])),
        'risk_level': 'high' if high_risk_ratio > 0.3 else 'medium' if high_risk_ratio > 0.1 else 'low',
        'interpretation': 'lower_ratio_means_higher_risk'
    }


def calculate_membership_inference(df_orig, df_aug):
    """
    Membership Inference Attack:
    Quanto è facile determinare se un record apparteneva al dataset originale
    """
    # Combina i dataset e crea label
    X_orig = df_orig.values
    X_aug = df_aug.values
    
    # Campiona per bilanciare
    n_samples = min(len(X_orig), len(X_aug), 1000)
    
    orig_indices = np.random.choice(len(X_orig), n_samples, replace=False)
    aug_indices = np.random.choice(len(X_aug), n_samples, replace=False)
    
    X = np.vstack([X_orig[orig_indices], X_aug[aug_indices]])
    y = np.hstack([np.ones(n_samples), np.zeros(n_samples)])
    
    # Train classificatore per distinguere originale da aumentato
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    
    accuracy = clf.score(X_test, y_test)
    
    # Privacy score: quanto è difficile distinguere (0.5 = impossibile, 1.0 = facile)
    privacy_score = 1 - (accuracy - 0.5) * 2  # Normalizza a [0, 1]
    
    return {
        'classifier_accuracy': float(accuracy),
        'privacy_score': float(max(0, privacy_score)),
        'distinguishability': float(accuracy),
        'privacy_level': 'high' if privacy_score > 0.7 else 'medium' if privacy_score > 0.4 else 'low',
        'interpretation': 'higher_privacy_score_is_better (0.5=indistinguishable)'
    }


def calculate_attribute_disclosure(df_orig, df_aug):
    """
    Attribute Disclosure Risk:
    Quanto è facile inferire attributi sensibili dal dataset aumentato
    """
    # Normalizza
    scaler = StandardScaler()
    X_orig = scaler.fit_transform(df_orig)
    X_aug = scaler.transform(df_aug)
    
    # Per ogni record aumentato, trova il record originale più vicino
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X_orig)
    distances, indices = nbrs.kneighbors(X_aug)
    
    # Calcola differenze attributo per attributo
    closest_originals = df_orig.iloc[indices.flatten()]
    closest_originals.index = df_aug.index
    
    attribute_risks = {}
    for col in df_aug.columns:
        # Differenza percentuale media
        diffs = np.abs(df_aug[col].values - closest_originals[col].values)
        mean_diff = np.mean(diffs)
        
        # Normalizza per il range
        col_range = df_orig[col].max() - df_orig[col].min()
        if col_range > 0:
            normalized_diff = mean_diff / col_range
        else:
            normalized_diff = 0
        
        attribute_risks[col] = {
            'absolute_difference': float(mean_diff),
            'normalized_difference': float(normalized_diff),
            'risk': 'high' if normalized_diff < 0.1 else 'medium' if normalized_diff < 0.3 else 'low'
        }
    
    avg_risk = np.mean([r['normalized_difference'] for r in attribute_risks.values()])
    
    return {
        'average_attribute_risk': float(avg_risk),
        'per_attribute': attribute_risks,
        'overall_risk': 'high' if avg_risk < 0.15 else 'medium' if avg_risk < 0.35 else 'low',
        'interpretation': 'lower_difference_means_higher_risk'
    }


def calculate_dcr(df_orig, df_aug):
    """
    Distance to Closest Record (DCR):
    Distanza minima tra ogni record aumentato e tutti i record originali
    """
    # Normalizza
    scaler = StandardScaler()
    X_orig = scaler.fit_transform(df_orig)
    X_aug = scaler.transform(df_aug)
    
    # Calcola distanza al record più vicino
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X_orig)
    distances, _ = nbrs.kneighbors(X_aug)
    
    # Statistiche
    dcr_values = distances.flatten()
    
    # Threshold per "troppo vicino" (potenziale privacy leak)
    privacy_threshold = np.percentile(dcr_values, 10)
    risky_records = np.sum(dcr_values < privacy_threshold)
    
    return {
        'mean_dcr': float(np.mean(dcr_values)),
        'median_dcr': float(np.median(dcr_values)),
        'min_dcr': float(np.min(dcr_values)),
        'max_dcr': float(np.max(dcr_values)),
        'risky_records_count': int(risky_records),
        'risky_records_ratio': float(risky_records / len(dcr_values)),
        'privacy_threshold': float(privacy_threshold),
        'interpretation': 'higher_dcr_is_better'
    }


def calculate_uniqueness_score(df_orig, df_aug):
    """
    Uniqueness Score:
    Misura quanto i record aumentati sono unici rispetto agli originali
    """
    # Normalizza
    scaler = StandardScaler()
    X_orig = scaler.fit_transform(df_orig)
    X_aug = scaler.transform(df_aug)
    
    # Per ogni record aumentato, conta quanti record originali sono "vicini"
    radius = 0.5  # Threshold di vicinanza
    nbrs = NearestNeighbors(radius=radius, algorithm='ball_tree').fit(X_orig)
    neighbors = nbrs.radius_neighbors(X_aug, return_distance=False)
    
    # Conta vicini per record
    neighbor_counts = [len(n) for n in neighbors]
    
    # Score: meno vicini = più unico
    uniqueness_scores = 1 / (1 + np.array(neighbor_counts))
    
    return {
        'average_uniqueness': float(np.mean(uniqueness_scores)),
        'median_uniqueness': float(np.median(uniqueness_scores)),
        'highly_unique_ratio': float(np.mean(uniqueness_scores > 0.7)),
        'average_neighbors': float(np.mean(neighbor_counts)),
        'interpretation': 'higher_uniqueness_is_better_for_privacy',
        'privacy_quality': 'good' if np.mean(uniqueness_scores) > 0.6 else 'fair' if np.mean(uniqueness_scores) > 0.4 else 'poor'
    }


def calculate_k_anonymity(df_orig, df_aug, quasi_columns=None, n_bins=10):
    """
    Calcola k-anonymity usando binning quantile sulle colonne quasi-identificative.
    Se quasi_columns is None, usa tutte le colonne presenti.
    Restituisce la k (min group size) sul dataset aumentato e statistiche aggiuntive.
    """
    if quasi_columns is None:
        quasi_columns = list(df_orig.columns)
    if len(quasi_columns) == 0:
        return {"error": "Nessuna colonna per k-anonymity"}

    # Binning quantile su orig per definire gli intervalli (mantenere coerenza)
    binned = {}
    for col in quasi_columns:
        try:
            # qcut può fallire se troppi valori uguali; fallback a cut
            binned[col], bins = pd.qcut(df_orig[col], q=min(n_bins, len(df_orig)), duplicates='drop', retbins=True)
        except Exception:
            binned[col], bins = pd.cut(df_orig[col], bins=min(n_bins, len(df_orig)), retbins=True)
        # applichiamo gli stessi bins a df_aug (per coerenza)
        # store bins
        binned[col] = bins

    def apply_bins(df):
        parts = []
        for col in quasi_columns:
            bins = binned[col]
            # pd.cut ritorna categorie coerenti
            parts.append(pd.cut(df[col], bins=bins, include_lowest=True).astype(str))
        # equivalence class key
        keys = pd.Series(list(zip(*parts)))
        return keys

    try:
        keys_orig = apply_bins(df_orig)
        keys_aug = apply_bins(df_aug)
    except Exception as e:
        return {"error": f"Impossibile applicare il binning: {e}"}

    # calcola dimensione delle equivalence classes
    counts_orig = keys_orig.value_counts()
    counts_aug = keys_aug.value_counts()

    # k-anonymity sul dataset aumentato = minimo gruppo (se 0 -> nessuna classe)
    k_value = int(counts_aug.min()) if len(counts_aug) > 0 else 0

    # statistiche utili
    total_aug = len(df_aug)
    small_k_thresholds = [1, 2, 5, 10]
    small_stats = {f"ratio_groups_lt_{t}": float((counts_aug < t).sum() / len(counts_aug)) if len(counts_aug) > 0 else 0.0
                   for t in small_k_thresholds}
    small_record_stats = {f"ratio_records_in_groups_lt_{t}":
                          float((counts_aug[counts_aug < t].sum()) / total_aug) if total_aug > 0 else 0.0
                          for t in small_k_thresholds}

    return {
        'k_anonymity': k_value,
        'n_equivalence_classes': int(len(counts_aug)),
        'median_equivalence_class_size': float(counts_aug.median()) if len(counts_aug) > 0 else 0.0,
        'mean_equivalence_class_size': float(counts_aug.mean()) if len(counts_aug) > 0 else 0.0,
        'small_class_stats_by_count': small_stats,
        'small_class_stats_by_records': small_record_stats,
        'interpretation': 'k = min class size in augmented dataset (after quantile binning of QIs)'
    }


def calculate_l_diversity(df_orig, df_aug, quasi_columns=None, sensitive_column=None, n_bins=10):
    """
    Calcola diverse misure di l-diversity:
      - distinct count per equivalence class (numero di valori sensibili distinti)
      - entropy per classe (diversity informativa)
    Default: usa tutte le colonne come quasi-identificatori e l'ultima colonna
    come attributo sensibile se non specificato.
    """
    if quasi_columns is None:
        quasi_columns = list(df_orig.columns)
    if len(quasi_columns) == 0:
        return {"error": "Nessuna colonna per l-diversity"}

    if sensitive_column is None:
        sensitive_column = quasi_columns[-1]  # default: ultima colonna numerica

    # assicurati che sensitive_column esista
    if sensitive_column not in df_orig.columns or sensitive_column not in df_aug.columns:
        return {"error": f"Attributo sensibile '{sensitive_column}' non trovato nei dataset"}

    # riusiamo lo stesso binning quantile usato per k-anonymity sulle QI
    binned = {}
    for col in quasi_columns:
        try:
            binned[col], bins = pd.qcut(df_orig[col], q=min(n_bins, len(df_orig)), duplicates='drop', retbins=True)
        except Exception:
            binned[col], bins = pd.cut(df_orig[col], bins=min(n_bins, len(df_orig)), retbins=True)
        binned[col] = bins

    def make_eq_keys(df):
        parts = []
        for col in quasi_columns:
            bins = binned[col]
            parts.append(pd.cut(df[col], bins=bins, include_lowest=True).astype(str))
        keys = pd.Series(list(zip(*parts)), index=df.index)
        return keys

    try:
        keys_aug = make_eq_keys(df_aug)
    except Exception as e:
        return {"error": f"Impossibile creare equivalence classes: {e}"}

    # raggruppa dataset aumentato per equivalence class e calcola metriche sul sensitive_attribute
    eq_groups = df_aug.groupby(keys_aug)

    from scipy.stats import entropy

    distinct_counts = []
    entropies = []
    class_sizes = []

    for key, group in eq_groups:
        class_sizes.append(len(group))
        # per valori numerici, categorizziamo il sensitivo con bins (qcut) per calcolare distinct/entropy
        try:
            sens_vals = pd.qcut(group[sensitive_column], q=min(10, max(2, len(group))), duplicates='drop').astype(str)
        except Exception:
            # fallback a valori originali (cast a string)
            sens_vals = group[sensitive_column].astype(str)

        value_counts = sens_vals.value_counts()
        distinct_counts.append(int(value_counts.size))
        # entropy base e
        probs = value_counts / value_counts.sum()
        entropies.append(float(entropy(probs, base=2)))

    if len(distinct_counts) == 0:
        return {
            'average_distinct_sensitive_values': 0.0,
            'median_distinct_sensitive_values': 0.0,
            'average_entropy': 0.0,
            'median_entropy': 0.0,
            'n_equivalence_classes': 0,
            'interpretation': 'no equivalence classes found'
        }

    # statistiche aggregate
    avg_distinct = float(np.mean(distinct_counts))
    med_distinct = float(np.median(distinct_counts))
    avg_entropy = float(np.mean(entropies))
    med_entropy = float(np.median(entropies))
    n_classes = int(len(distinct_counts))

    # percentuali di classi che soddisfano l >= 2,3,5
    l_thresholds = [2, 3, 5]
    l_stats = {f"ratio_classes_l_{l}": float(np.mean(np.array(distinct_counts) >= l)) for l in l_thresholds}

    return {
        'average_distinct_sensitive_values': avg_distinct,
        'median_distinct_sensitive_values': med_distinct,
        'average_entropy_bits': avg_entropy,
        'median_entropy_bits': med_entropy,
        'n_equivalence_classes': n_classes,
        'l_threshold_stats': l_stats,
        'interpretation': 'higher distinct count and higher entropy = better l-diversity'
    }
