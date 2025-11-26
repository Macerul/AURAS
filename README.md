# AURAS - Data Augmentation Quality Assessment Tool

AURAS (A Unified tool foR assessing the quality of Synthetic data representation) is a comprehensive tool designed to evaluate the quality of synthetic or augmented datasets. It provides a suite of metrics to assess Fidelity, Diversity, Privacy, and Utility, ensuring that your data augmentation strategies are effective and safe.

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


