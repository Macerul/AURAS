"""
Utility per caricare e validare i dataset
"""
import pandas as pd
import os


def load_dataset(file_path):
    """
    Carica un dataset da file CSV o Parquet
    
    Args:
        file_path: Path al file
        
    Returns:
        pandas.DataFrame: Dataset caricato
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    try:
        if ext == '.csv':
            df = pd.read_csv(file_path)
        elif ext in ['.parquet', '.pq']:
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Formato file non supportato: {ext}")
        
        return df
    
    except Exception as e:
        raise ValueError(f"Errore nel caricamento del file: {str(e)}")


def validate_datasets(df_original, df_augmented):
    """
    Valida che i due dataset siano compatibili per il confronto
    
    Args:
        df_original: DataFrame originale
        df_augmented: DataFrame aumentato
        
    Returns:
        dict: Risultato della validazione
    """
    errors = []
    warnings = []
    
    # Check: almeno una colonna comune
    common_cols = set(df_original.columns).intersection(set(df_augmented.columns))
    if len(common_cols) == 0:
        errors.append("Nessuna colonna comune tra i due dataset")
    
    # Check: almeno una colonna numerica comune
    numeric_orig = set(df_original.select_dtypes(include=['number']).columns)
    numeric_aug = set(df_augmented.select_dtypes(include=['number']).columns)
    common_numeric = numeric_orig.intersection(numeric_aug)
    
    if len(common_numeric) == 0:
        errors.append("Nessuna colonna numerica comune tra i due dataset")
    
    # Warning: dataset troppo piccoli
    if len(df_original) < 10:
        warnings.append("Dataset originale molto piccolo (< 10 record)")
    if len(df_augmented) < 10:
        warnings.append("Dataset aumentato molto piccolo (< 10 record)")
    
    # Warning: dataset molto sbilanciati
    size_ratio = len(df_augmented) / len(df_original) if len(df_original) > 0 else 0
    if size_ratio < 0.5:
        warnings.append(f"Dataset aumentato significativamente più piccolo dell'originale (ratio: {size_ratio:.2f})")
    elif size_ratio > 10:
        warnings.append(f"Dataset aumentato molto più grande dell'originale (ratio: {size_ratio:.2f})")
    
    # Info
    info = {
        'original_shape': df_original.shape,
        'augmented_shape': df_augmented.shape,
        'common_columns': len(common_cols),
        'common_numeric_columns': len(common_numeric),
        'size_ratio': float(size_ratio)
    }
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'info': info
    }


def get_dataset_summary(df, name="Dataset"):
    """
    Genera un sommario descrittivo del dataset
    
    Args:
        df: DataFrame
        name: Nome del dataset
        
    Returns:
        dict: Sommario del dataset
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    summary = {
        'name': name,
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'n_numeric': len(numeric_cols),
        'n_categorical': len(categorical_cols),
        'columns': df.columns.tolist(),
        'numeric_columns': numeric_cols.tolist(),
        'categorical_columns': categorical_cols.tolist(),
        'missing_values': df.isnull().sum().to_dict(),
        'total_missing': int(df.isnull().sum().sum()),
        'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024)
    }
    
    # Statistiche base per colonne numeriche
    if len(numeric_cols) > 0:
        summary['numeric_stats'] = df[numeric_cols].describe().to_dict()
    
    return summary
