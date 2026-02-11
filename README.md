# AURAS - Data Augmentation Quality Assessment Tool

AURAS (A Unified tool foR assessing the quality of Synthetic data representation) is a comprehensive tool designed to evaluate the quality of synthetic or augmented datasets. It provides a suite of metrics to assess Fidelity, Diversity, Privacy, and Utility, ensuring that your data augmentation strategies are effective and safe. A video demo is available on Youtube: https://youtu.be/h4l19InIRv4.

## 🚀 Features

AURAS provides a multi-dimensional assessment of your datasets:

*   **Fidelity**: Measures how closely the augmented data resembles the original data.
    *   *Metrics*: Kolmogorov-Smirnov Test, MMD Score, PCA Similarity, KL/JS Divergence, Statistical Moments.
*   **Diversity**: Evaluates the variety and coverage of the augmented data.
    *   *Metrics*: Feature Entropy, Coverage Ratio, Cluster Uniformity, Silhouette Score.
*   **Privacy**: Assesses the risk of re-identification and information leakage.
    *   *Metrics*: Nearest Neighbor Risk, Membership Inference, Attribute Disclosure, k-Anonymity, l-Diversity.
*   **Utility**: Determines how well the augmented data performs in downstream machine learning tasks.
    *   *Metrics*: Classification Accuracy, F1-Score (with Cross-Validation support).
*   **AI-Powered Explanations**: Integrated with **Ollama** to provide natural language explanations of complex metrics and suggestions for improvement.
*   **Interactive Dashboard**: A modern, dark-themed web interface to visualize results with heatmaps, charts, and detailed scorecards.
*   **Synthetic Data Generation**: Generate high-quality synthetic datasets using advanced techniques.
    *   *Techniques*: SMOTE, ADASYN, Gaussian Copula, CTGAN, TVAE, SMOTECDNN, Bayesian Network, Random Sampling, Noise Injection, KDE Sampling, Dummy Generator.
    *   *Interface*: Dedicated page (`synthetic_generator.html`) for configuring parameters and generating data.

## 🔌 API Endpoints

AURAS exposes a RESTful API for integration:

### Core Analysis
*   `POST /api/upload`: Upload original and augmented datasets.
*   `POST /api/upload-multi`: Upload original and multiple augmented datasets for comparison.
*   `POST /api/analyze`: Run quality assessment metrics.
*   `POST /api/utility/analyze`: Run utility metrics (ML models).
*   `POST /api/explain`: Generate AI explanations for metrics using Ollama.
*   `POST /api/suggestions/generate`: Generate improvement suggestions.

### Synthetic Generation
*   `POST /api/generate-synthetic`: Generate synthetic data.
    *   *Params*: `dataset` (file), `techniques` (list), `num_samples`, `technique_params`.
*   `GET /api/download-synthetic`: Download generated synthetic datasets.
    *   *Params*: `file` (path), `technique`.

### Export & History
*   `POST /api/export`: Export results as JSON.
*   `POST /api/export/pdf`: Export report as PDF.
*   `POST /api/export/excel`: Export results as Excel.
*   `GET /api/history`: Get analysis history.

## 📦 Installation

The easiest way to run AURAS is using Docker Compose.

### Prerequisites

*   [Docker](https://www.docker.com/get-started)
*   [Docker Compose](https://docs.docker.com/compose/install/)

### Quick Start

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/AURAS.git
    cd AURAS
    ```

2.  **Start the application**
    ```bash
    make start
    # OR
    docker-compose up -d --build
    ```

3.  **Access the Dashboard**
    Open your browser and navigate to:
    `http://localhost:5088`

4.  **Stop the application**
    ```bash
    make stop
    ```

### Local Development (Without Docker)

If you prefer to run the backend and frontend locally:

1.  **Backend Setup**
    ```bash
    cd backend
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    python app.py
    ```

## 🖥️ Usage

1.  **Upload Datasets**:
    *   Click on "Original Dataset" to upload your source CSV/Parquet file.
    *   Click on "Augmented Dataset" to upload the synthetic version.
    *   *Note*: You can also upload multiple augmented datasets for comparison with the compare button.

2.  **Run Analysis**:
    *   Click **"Analyze Datasets"**. The system will compute metrics across all dimensions.

3.  **Explore Results**:
    *   **Overview**: See the aggregate score and rating (Excellent, Good, Fair, etc.).
    *   **Tabs**: Navigate through Fidelity, Diversity, Privacy, and Utility tabs for detailed metric breakdowns.
    *   **Visualizations**: View PCA plots, correlation heatmaps, and distribution charts.

4.  **Get AI Insights**:
    *   Click **"Explain"** on any metric card to get a natural language explanation of what the score means for your specific data.
    *   Use **"Suggestions"** to get actionable advice on how to improve your augmentation strategy.

5.  **Generate Synthetic data**:
    *   Click **"Generate"** button to view the synthetic generation interface.
    *   Select the desired models for generation, configure the parameters and click **"Generate Synthetic Data"** to start generating samples.

## ➕ Extending AURAS

AURAS is designed to be easily extensible. You can add new metrics or additional machine learning models for utility analysis.

### Adding a New Metric

Metrics are organized by category in `backend/metrics/` (`fidelity.py`, `diversity.py`, `privacy.py`, `utility.py`).

**Step-by-Step Guide:**

1.  **Locate the Category**: Open the relevant file in `backend/metrics/` (e.g., `fidelity.py` for new fidelity metrics).
2.  **Define the Function**: Create a new function that accepts the original and augmented DataFrames.
    ```python
    def calculate_my_custom_metric(df_orig, df_aug):
        # Implement your logic
        score = ... 
        return {
            'score': float(score),
            'interpretation': 'higher_is_better'
        }
    ```
3.  **Register the Metric**: Add your function call to the main calculation function in that file (e.g., `calculate_fidelity_metrics`).
    ```python
    # In calculate_fidelity_metrics(...)
    metrics['my_custom_metric'] = calculate_my_custom_metric(df_orig_num, df_aug_num)
    ```
4.  **Frontend Display**: The new metric will appear in the raw JSON output. To visualize it in the dashboard, update `frontend/script.js` to parse the new key and render a card or chart.

### Adding a New Utility Model

You can add more scikit-learn compatible models to the utility analysis.

**Step-by-Step Guide:**

1.  **Open Utility Config**: Edit `backend/metrics/utility.py`.
2.  **Import Model**: Import your model class at the top of the file.
    ```python
    from sklearn.linear_model import MyNewClassifier
    ```
3.  **Register Model**: In the `get_model` function, add your model to the `models` dictionary.
    ```python
    models = {
        # ... existing models
        'my_new_model': MyNewClassifier
    }
    ```
4.  **Default Parameters**: (Optional) Add default parameters in the `get_model` function if needed.
    ```python
    if model_type == 'my_new_model':
        params_copy.setdefault('param_name', default_value)
    ```
5.  **Usage**: You can now request this model via the API by passing `{"type": "my_new_model"}` in the `model_configs` list.

### Adding a New Synthetic Data Generator

You can add new synthetic data generation techniques to the system.

**Step-by-Step Guide:**

1.  **Define the Generator Function**: In `backend/app.py`, define a function that takes the original DataFrame and parameters.
    ```python
    def generate_my_technique(df_original, num_samples, params, random_seed):
        # Implement your generation logic here
        # ...
        return df_synthetic
    ```
2.  **Register in Dispatcher**: Update the `generate_with_technique` function in `backend/app.py` to include your new technique.
    ```python
    def generate_with_technique(df_original, technique_id, ...):
        # ... existing techniques
        elif technique_id == 'my_new_technique':
            return generate_my_technique(df_original, num_samples, params, random_seed)
    ```
3.  **Frontend Update**: To make it accessible in the UI, update `frontend/synthetic_generator.html` to include your new technique in the selection dropdown.
    ```html
    <select id="technique-selector">
        <!-- ... existing options -->
        <option value="my_new_technique">My New Technique</option>
    </select>
    ```


