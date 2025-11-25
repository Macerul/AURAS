"""
Demo Script: Generate Sample Datasets for Multi-Comparison Testing
Creates 1 original dataset + 3 synthetic variants with different characteristics
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory
output_dir = Path('./examples')
output_dir.mkdir(exist_ok=True)

print("🎲 Generating demo datasets...")

def create_labels_imbalanced(features, n_samples):
    """Create imbalanced labels based on feature combinations"""
    # Create probability scores based on features
    risk_scores = (
        (features['age'] < 30) * 0.3 +
        (features['credit_score'] < 600) * 0.4 +
        (features['income'] < 30000) * 0.2 +
        (features['num_accounts'] < 2) * 0.1
    )

    # Add some noise
    risk_scores += np.random.normal(0, 0.1, n_samples)

    # Create imbalanced classes (80% class 0, 20% class 1)
    threshold = np.percentile(risk_scores, 80)  # Top 20% become class 1
    labels = (risk_scores > threshold).astype(int)

    return labels

def create_labels_balanced(features, n_samples):
    """Create balanced labels based on feature combinations"""
    # Create probability scores based on features
    risk_scores = (
        (features['age'] < 35) * 0.25 +
        (features['credit_score'] < 620) * 0.35 +
        (features['income'] < 35000) * 0.25 +
        (features['num_accounts'] < 2) * 0.15
    )

    # Add some noise
    risk_scores += np.random.normal(0, 0.15, n_samples)

    # Create balanced classes (50% class 0, 50% class 1)
    threshold = np.median(risk_scores)  # Median split for balanced classes
    labels = (risk_scores > threshold).astype(int)

    return labels

# ========================================
# 1. Original Dataset (IMBALANCED LABELS)
# ========================================
n_samples = 1000
n_samples = n_samples//2

original_data = {
    'age': np.random.normal(45, 15, n_samples).astype(int).clip(18, 90),
    'income': np.random.lognormal(10.5, 0.8, n_samples).clip(20000, 500000),
    'credit_score': np.random.normal(700, 100, n_samples).astype(int).clip(300, 850),
    'num_accounts': np.random.poisson(3, n_samples).clip(0, 10),
    'account_age_years': np.random.exponential(8, n_samples).clip(0, 40),
}

# Add some correlations
original_data['savings'] = (
    original_data['income'] * 0.15 +
    np.random.normal(0, 5000, n_samples)
).clip(0, None)

original_data['loan_amount'] = (
    original_data['income'] * 0.3 +
    (850 - original_data['credit_score']) * 100 +
    np.random.normal(0, 10000, n_samples)
).clip(0, None)

# Add IMBALANCED labels to original dataset
original_data['label'] = create_labels_imbalanced(original_data, n_samples)

df_original = pd.DataFrame(original_data)
df_original.to_csv(output_dir / 'original.csv', index=False)
print(f"✅ Original dataset: {len(df_original)} rows, {len(df_original.columns)} columns")
print(f"   Label distribution: {dict(df_original['label'].value_counts().sort_index())}")

# ========================================
# 2. Synthetic Dataset 1: High Fidelity
# ========================================
# Very similar to original, good distribution matching
n_samples = n_samples*2

synthetic1_data = {
    'age': np.random.normal(45, 15.2, n_samples).astype(int).clip(18, 90),
    'income': np.random.lognormal(10.48, 0.82, n_samples).clip(20000, 500000),
    'credit_score': np.random.normal(698, 102, n_samples).astype(int).clip(300, 850),
    'num_accounts': np.random.poisson(3, n_samples).clip(0, 10),
    'account_age_years': np.random.exponential(8.1, n_samples).clip(0, 40),
}

synthetic1_data['savings'] = (
    synthetic1_data['income'] * 0.14 +
    np.random.normal(0, 5100, n_samples)
).clip(0, None)

synthetic1_data['loan_amount'] = (
    synthetic1_data['income'] * 0.29 +
    (850 - synthetic1_data['credit_score']) * 98 +
    np.random.normal(0, 10200, n_samples)
).clip(0, None)

# Add BALANCED labels to synthetic dataset 1
synthetic1_data['label'] = create_labels_balanced(synthetic1_data, n_samples)

df_synthetic1 = pd.DataFrame(synthetic1_data)
df_synthetic1.to_csv(output_dir / 'synthetic_high_fidelity.csv', index=False)
print(f"✅ Synthetic 1 (High Fidelity): {len(df_synthetic1)} rows")
print(f"   Label distribution: {dict(df_synthetic1['label'].value_counts().sort_index())}")

# ========================================
# 3. Synthetic Dataset 2: High Diversity
# ========================================
# More spread out, covers wider range
synthetic2_data = {
    'age': np.random.normal(45, 20, n_samples).astype(int).clip(18, 90),  # Higher variance
    'income': np.random.lognormal(10.5, 1.2, n_samples).clip(15000, 600000),  # Wider range
    'credit_score': np.random.normal(700, 130, n_samples).astype(int).clip(300, 850),  # More spread
    'num_accounts': np.random.poisson(4, n_samples).clip(0, 12),  # More accounts
    'account_age_years': np.random.exponential(10, n_samples).clip(0, 50),  # Older accounts
}

synthetic2_data['savings'] = (
    synthetic2_data['income'] * np.random.uniform(0.1, 0.25, n_samples) +
    np.random.normal(0, 8000, n_samples)
).clip(0, None)

synthetic2_data['loan_amount'] = (
    synthetic2_data['income'] * np.random.uniform(0.2, 0.4, n_samples) +
    (850 - synthetic2_data['credit_score']) * np.random.uniform(50, 150, n_samples) +
    np.random.normal(0, 15000, n_samples)
).clip(0, None)

# Add BALANCED labels to synthetic dataset 2
synthetic2_data['label'] = create_labels_balanced(synthetic2_data, n_samples)

df_synthetic2 = pd.DataFrame(synthetic2_data)
df_synthetic2.to_csv(output_dir / 'synthetic_high_diversity.csv', index=False)
print(f"✅ Synthetic 2 (High Diversity): {len(df_synthetic2)} rows")
print(f"   Label distribution: {dict(df_synthetic2['label'].value_counts().sort_index())}")

# ========================================
# 4. Synthetic Dataset 3: Balanced
# ========================================
# Balance between fidelity and diversity
synthetic3_data = {
    'age': np.random.normal(46, 17, n_samples).astype(int).clip(18, 90),
    'income': np.random.lognormal(10.52, 0.95, n_samples).clip(18000, 520000),
    'credit_score': np.random.normal(702, 110, n_samples).astype(int).clip(300, 850),
    'num_accounts': np.random.poisson(3, n_samples).clip(0, 11),
    'account_age_years': np.random.exponential(8.5, n_samples).clip(0, 42),
}

synthetic3_data['savings'] = (
    synthetic3_data['income'] * 0.16 +
    np.random.normal(0, 6000, n_samples)
).clip(0, None)

synthetic3_data['loan_amount'] = (
    synthetic3_data['income'] * 0.31 +
    (850 - synthetic3_data['credit_score']) * 105 +
    np.random.normal(0, 11000, n_samples)
).clip(0, None)

# Add BALANCED labels to synthetic dataset 3
synthetic3_data['label'] = create_labels_balanced(synthetic3_data, n_samples)

df_synthetic3 = pd.DataFrame(synthetic3_data)
df_synthetic3.to_csv(output_dir / 'synthetic_balanced.csv', index=False)
print(f"✅ Synthetic 3 (Balanced): {len(df_synthetic3)} rows")
print(f"   Label distribution: {dict(df_synthetic3['label'].value_counts().sort_index())}")

# ========================================
# Summary
# ========================================
print("\n" + "="*50)
print("📊 Demo Datasets Generated Successfully!")
print("="*50)
print(f"\nLocation: {output_dir}")
print("\nFiles created:")
print("  1. original.csv - Original dataset (1000 rows) - IMBALANCED LABELS")
print("  2. synthetic_high_fidelity.csv - High similarity to original - BALANCED LABELS")
print("  3. synthetic_high_diversity.csv - Wide distribution coverage - BALANCED LABELS")
print("  4. synthetic_balanced.csv - Balanced approach - BALANCED LABELS")
print("\nLabel distributions:")
print(f"  Original: {dict(df_original['label'].value_counts().sort_index())}")
print(f"  Synthetic 1: {dict(df_synthetic1['label'].value_counts().sort_index())}")
print(f"  Synthetic 2: {dict(df_synthetic2['label'].value_counts().sort_index())}")
print(f"  Synthetic 3: {dict(df_synthetic3['label'].value_counts().sort_index())}")
print("\nUse these files to test the multi-dataset comparison feature!")
print("\nExpected Results:")
print("  - Synthetic 1: Best fidelity score")
print("  - Synthetic 2: Best diversity score")
print("  - Synthetic 3: Best overall balance")