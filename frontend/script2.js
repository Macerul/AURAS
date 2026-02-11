// API Configuration
//const API_URL = 'http://localhost:50088/api';
const API_URL = (() => {
    // Se siamo in sviluppo locale (non in container)
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
        return 'http://localhost:5088/api';
    }
    // Se siamo in produzione/container, usa percorso relativo
    return '/api';
})();

let uploadedFiles = { original: null, augmented: [] };
let uploadedPaths = { original: null, augmented: [] };
let currentAugmentedIndex = 0; // Per gestire quale augmented dataset visualizzare
let userWeights = {
    fidelity: 0.33,
    diversity: 0.33,
    privacy: 0.34
};

let analysisResults = null;
let charts = {};
let selectedModel = '';
let explanationLevel = '';
let selectedModels = [];
let metricDescriptions = {};
let currentDbType = 'postgresql';

// Onboarding
document.addEventListener('DOMContentLoaded', () => {
    initializeTheme();
    initializeOnboarding();
    initializeEventListeners();
    loadMetricDescriptions();
});

function initializeTheme() {
    const savedTheme = localStorage.getItem('heroesTheme') || 'dark';
    document.body.dataset.theme = savedTheme;
    updateThemeIcon(savedTheme);
}

function updateThemeIcon(theme) {
    const icon = document.getElementById('themeIcon');
    icon.textContent = theme === 'dark' ? '🌙' : '☀️';
}

async function loadMetricDescriptions() {
    try {
        const response = await fetch('metric_descriptions.json');
        metricDescriptions = await response.json();
    } catch (error) {
        console.error('Failed to load metric descriptions:', error);
    }
}

function initializeOnboarding() {
    const hasSeenOnboarding = localStorage.getItem('heroesOnboardingComplete');
    if (!hasSeenOnboarding) {
        document.getElementById('onboardingOverlay').classList.remove('hidden');
    }

    let currentStep = 1;
    const totalSteps = 5;

    document.querySelectorAll('.onboarding-next').forEach(btn => {
        btn.addEventListener('click', () => {
            if (currentStep < totalSteps) goToStep(currentStep + 1);
        });
    });

    document.querySelectorAll('.onboarding-prev').forEach(btn => {
        btn.addEventListener('click', () => {
            if (currentStep > 1) goToStep(currentStep - 1);
        });
    });

    document.getElementById('skipOnboarding').addEventListener('click', completeOnboarding);
    document.getElementById('startOnboarding').addEventListener('click', completeOnboarding);

    document.querySelectorAll('.onboarding-dots .dot').forEach(dot => {
        dot.addEventListener('click', () => goToStep(parseInt(dot.dataset.step)));
    });

    function goToStep(step) {
        document.querySelectorAll('.onboarding-step').forEach(s => s.classList.remove('active'));
        document.querySelectorAll('.onboarding-dots .dot').forEach(d => d.classList.remove('active'));

        document.querySelector(`.onboarding-step[data-step="${step}"]`).classList.add('active');
        document.querySelector(`.onboarding-dots .dot[data-step="${step}"]`).classList.add('active');

        currentStep = step;
    }

    function completeOnboarding() {
        localStorage.setItem('heroesOnboardingComplete', 'true');
        document.getElementById('onboardingOverlay').classList.add('hidden');
    }
}

// ==================== EVENT LISTENERS ====================

function initializeEventListeners() {
    // Theme toggle
    document.getElementById('themeToggle').addEventListener('click', toggleTheme);

    // Help and Settings
    document.getElementById('helpBtn').addEventListener('click', () => {
        document.getElementById('onboardingOverlay').classList.remove('hidden');
    });
    document.getElementById('settingsBtn').addEventListener('click', () => {
        document.getElementById('settingsModal').style.display = 'block';
        loadOllamaModels();
    });
    document.getElementById('closeSettings').addEventListener('click', () => {
        document.getElementById('settingsModal').style.display = 'none';
    });

    // Flying hero
    document.getElementById('heroIcon').addEventListener('click', function () {
        this.classList.add('fly');
        setTimeout(() => this.classList.remove('fly'), 2000);
    });

    // File uploads
    document.getElementById('originalFile').addEventListener('change', handleOriginalFile);
    document.getElementById('augmentedFiles').addEventListener('change', handleAugmentedFiles);

    // Database
    document.getElementById('loadFromDbBtn').addEventListener('click', () => {
        document.getElementById('dbModal').style.display = 'block';
    });
    document.getElementById('closeDbModal').addEventListener('click', () => {
        document.getElementById('dbModal').style.display = 'none';
    });
    document.querySelectorAll('.db-type-option').forEach(opt => {
        opt.addEventListener('click', function () {
            document.querySelectorAll('.db-type-option').forEach(o => o.classList.remove('selected'));
            this.classList.add('selected');
            currentDbType = this.dataset.type;
            document.getElementById('dbPort').value = currentDbType === 'mysql' ? '3306' : '5432';
        });
    });
    document.getElementById('testDbConnection').addEventListener('click', testDatabaseConnection);
    document.getElementById('loadFromDb').addEventListener('click', loadDatasetFromDatabase);

    // Analysis
    document.getElementById('analyzeBtn').addEventListener('click', analyzeDatasets);

    // Weight sliders
    ['Fidelity', 'Diversity', 'Privacy'].forEach(category => {
        const slider = document.getElementById(`weight${category}`);
        const valueSpan = document.getElementById(`weight${category}Value`);

        slider.addEventListener('input', (e) => {
            valueSpan.textContent = e.target.value + '%';
            updateWeightsTotal();
        });
    });


    document.getElementById('resetWeights').addEventListener('click', resetWeights);

    // Priority Buttons
    const btnPrioritizeFidelity = document.getElementById('btnPrioritizeFidelity');
    if (btnPrioritizeFidelity) btnPrioritizeFidelity.addEventListener('click', () => setPriorityWeight('fidelity'));

    const btnPrioritizeDiversity = document.getElementById('btnPrioritizeDiversity');
    if (btnPrioritizeDiversity) btnPrioritizeDiversity.addEventListener('click', () => setPriorityWeight('diversity'));

    const btnPrioritizePrivacy = document.getElementById('btnPrioritizePrivacy');
    if (btnPrioritizePrivacy) btnPrioritizePrivacy.addEventListener('click', () => setPriorityWeight('privacy'));

    // Utility
    document.querySelectorAll('.model-option').forEach(opt => {
        opt.addEventListener('click', function () {
            this.classList.toggle('selected');
            updateSelectedModels();
        });
    });

    const useCVEl = document.getElementById('useCV');
    const cvContainer = document.getElementById('utilityCvControls');
    if (useCVEl && cvContainer) {
        useCVEl.addEventListener('change', () => {
            if (useCVEl.checked) cvContainer.classList.add('use-cv-enabled');
            else cvContainer.classList.remove('use-cv-enabled');
        });
        // set initial state
        if (useCVEl.checked) cvContainer.classList.add('use-cv-enabled');
    }

    document.getElementById('trainModelsBtn').addEventListener('click', trainModels);

    // Suggestions
    document.getElementById('generateSuggestions').addEventListener('click', generateSuggestions);

    // Explainability
    document.getElementById('explainFidelity').addEventListener('click', () => explainMetrics('fidelity'));
    document.getElementById('explainDiversity').addEventListener('click', () => explainMetrics('diversity'));
    document.getElementById('explainPrivacy').addEventListener('click', () => explainMetrics('privacy'));
    document.getElementById('explainUtility').addEventListener('click', () => explainMetrics('utility'));
    document.getElementById('explainCustom').addEventListener('click', explainCustom);
    document.getElementById('closeExplanation').addEventListener('click', () => {
        document.getElementById('explanationResult').style.display = 'none';
    });

    // Exports
    document.getElementById('exportBtn').addEventListener('click', exportJSON);
    document.getElementById('exportPdfBtn').addEventListener('click', exportPDF);
    document.getElementById('exportExcelBtn').addEventListener('click', exportExcel);
    document.getElementById('screenshotBtn').addEventListener('click', captureScreenshot);
    document.getElementById('newAnalysisBtn').addEventListener('click', () => location.reload());

    // Tabs
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.dataset.tab;
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(tabName + 'Tab').classList.add('active');
        });
    });

    // Settings
    document.getElementById('refreshModels').addEventListener('click', loadOllamaModels);
    document.getElementById('explanationLevel').addEventListener('change', (e) => {
        explanationLevel = e.target.value;
    });

    // History
    document.getElementById('viewHistory').addEventListener('click', (e) => {
        e.preventDefault();
        viewAnalysisHistory();
    });
}

// ==================== THEME ====================

function toggleTheme() {
    const current = document.body.dataset.theme;
    const newTheme = current === 'dark' ? 'light' : 'dark';
    document.body.dataset.theme = newTheme;
    localStorage.setItem('heroesTheme', newTheme);
    updateThemeIcon(newTheme);

    // Update charts if they exist
    if (Object.keys(charts).length > 0) {
        Object.values(charts).forEach(chart => {
            if (chart) {
                chart.options.plugins.title.color = newTheme === 'dark' ? '#f1f5f9' : '#0f172a';
                chart.update();
            }
        });
    }
}

// ==================== FILE HANDLING ====================

function handleOriginalFile(e) {
    const file = e.target.files[0];
    if (file) {
        uploadedFiles.original = file;
        document.getElementById('originalFileName').textContent = file.name;
        checkFilesReady();
    }
}

function handleAugmentedFile(e) {
    const file = e.target.files[0];
    if (file) {
        uploadedFiles.augmented = file;
        document.getElementById('augmentedFileName').textContent = file.name;
        checkFilesReady();
    }
}


// ==================== DATABASE ====================

async function testDatabaseConnection() {
    const dbConfig = getDbConfig();

    try {
        const response = await fetch(`${API_URL}/db/test`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(dbConfig)
        });

        const result = await response.json();
        const statusEl = document.getElementById('dbStatus');
        statusEl.style.display = 'block';

        if (result.success) {
            statusEl.textContent = '✅ Connection successful!';
            statusEl.className = 'db-status success';
        } else {
            statusEl.textContent = `❌ Connection failed: ${result.message}`;
            statusEl.className = 'db-status error';
        }
    } catch (error) {
        const statusEl = document.getElementById('dbStatus');
        statusEl.style.display = 'block';
        statusEl.textContent = `❌ Error: ${error.message}`;
        statusEl.className = 'db-status error';
    }
}

async function loadDatasetFromDatabase() {
    const dbConfig = getDbConfig();
    const datasetType = document.getElementById('dbDatasetType').value;

    try {
        const statusEl = document.getElementById('dbStatus');
        statusEl.style.display = 'block';
        statusEl.textContent = '⏳ Loading dataset from database...';
        statusEl.className = 'db-status info';

        const response = await fetch(`${API_URL}/db/load`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                type: datasetType,
                db_config: dbConfig
            })
        });

        const result = await response.json();

        if (result.success) {
            statusEl.textContent = `✅ Loaded ${result.rows} rows, ${result.columns} columns`;
            statusEl.className = 'db-status success';

            // Store path
            uploadedPaths[datasetType] = result.path;

            // Update UI
            const fileNameEl = document.getElementById(`${datasetType}FileName`);
            fileNameEl.textContent = `DB: ${result.rows} rows (${result.columns} cols)`;

            checkFilesReady();

            setTimeout(() => {
                document.getElementById('dbModal').style.display = 'none';
            }, 2000);
        } else {
            statusEl.textContent = `❌ Failed: ${result.error}`;
            statusEl.className = 'db-status error';
        }
    } catch (error) {
        const statusEl = document.getElementById('dbStatus');
        statusEl.textContent = `❌ Error: ${error.message}`;
        statusEl.className = 'db-status error';
    }
}

function getDbConfig() {
    return {
        db_type: currentDbType,
        host: document.getElementById('dbHost').value,
        port: parseInt(document.getElementById('dbPort').value),
        database: document.getElementById('dbDatabase').value,
        user: document.getElementById('dbUser').value,
        password: document.getElementById('dbPassword').value,
        query: document.getElementById('dbQuery').value
    };
}

// ==================== ANALYSIS ====================

async function analyzeDatasets() {
    try {
        document.getElementById('loading').style.display = 'block';
        document.getElementById('analyzeBtn').disabled = true;

        let origPath;
        const augPaths = [];

        // Upload files if not loaded from DB
        if (uploadedFiles.original && uploadedFiles.augmented.length > 0) {
            const formData = new FormData();
            formData.append('original', uploadedFiles.original);

            // Aggiungi tutti i file augmented
            uploadedFiles.augmented.forEach((file, index) => {
                formData.append('augmented', file);
            });

            const uploadResponse = await fetch(`${API_URL}/upload-multi`, {
                method: 'POST',
                body: formData
            });

            if (!uploadResponse.ok) throw new Error('Upload failed');

            const uploadData = await uploadResponse.json();
            origPath = uploadData.original_path;
            augPaths.push(...uploadData.augmented_paths);
        } else {
            origPath = uploadedPaths.original;
            augPaths.push(...uploadedPaths.augmented);

        }
        // SALVA I PATH NELLA VARIABILE GLOBALE - QUESTA È LA CORREZIONE
        uploadedPaths.original = origPath;
        uploadedPaths.augmented = augPaths; // Questo è l'array di paths
        console.log("[Analize Dataset]")
        console.log(origPath)
        console.log(augPaths)

        // Analyze tutti i dataset
        const analyzePromises = augPaths.map(augPath =>
            fetch(`${API_URL}/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    original_path: origPath,
                    augmented_path: augPath,
                    weights: userWeights
                })
            }).then(response => {
                if (!response.ok) throw new Error('Analysis failed');
                return response.json();
            })
        );

        const analyzeResults = await Promise.all(analyzePromises);
        console.log("analyzeResults: ")
        console.log(analyzeResults)

        // Combina i risultati
        const combinedResults = {
            analyses: analyzeResults.map(result => result.results),
            augmented_count: analyzeResults.length
        };
        console.log(combinedResults)

        analysisResults = combinedResults;

        // Display results
        displayResults(analysisResults);

        // Show results section
        document.getElementById('uploadSection').style.display = 'none';
        document.getElementById('resultsSection').style.display = 'block';

    } catch (error) {
        alert(`Error: ${error.message}`);
        console.error(error);
    } finally {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('analyzeBtn').disabled = false;
    }
}

// Display Results
function displayResults(results) {
    console.log("SONO displayResults:")
    console.log(results)

    if (!results.analyses || results.analyses.length === 0) return;

    // Per ora mostra solo il primo risultato, ma puoi implementare un selettore
    const firstResult = results.analyses[0];

    console.log("firstResult")
    console.log(firstResult)

    // Overall Score
    const aggregate = firstResult.aggregate_score;
    document.getElementById('overallScore').textContent = (aggregate.overall * 100).toFixed(0);
    document.getElementById('overallRating').textContent = aggregate.rating;

    if (aggregate.weights) {
        const weightsInfo = `
        <div class="weights-used">
            <small>Weights used:
                Fidelity ${(aggregate.weights.fidelity * 100).toFixed(0)}%,
                Diversity ${(aggregate.weights.diversity * 100).toFixed(0)}%,
                Privacy ${(aggregate.weights.privacy * 100).toFixed(0)}%
            </small>
        </div>
    `;
    }

    // Category Scores
    if (aggregate.scores.fidelity !== undefined) {
        const fid = (aggregate.scores.fidelity * 100).toFixed(0);
        document.getElementById('fidelityScore').textContent = fid;
        document.getElementById('fidelityBar').style.width = `${fid}%`;
    }
    else {
        console.log("fidelity undefined")

    }

    if (aggregate.scores.diversity !== undefined) {
        const div = (aggregate.scores.diversity * 100).toFixed(0);
        document.getElementById('diversityScore').textContent = div;
        document.getElementById('diversityBar').style.width = `${div}%`;
    }
    else {
        console.log("diversity undefined")

    }

    if (aggregate.scores.privacy !== undefined) {
        const priv = (aggregate.scores.privacy * 100).toFixed(0);
        document.getElementById('privacyScore').textContent = priv;
        document.getElementById('privacyBar').style.width = `${priv}%`;
    }
    else {
        console.log("privacy undefined")
    }

    console.log("firstResult.fidelity")
    console.log(firstResult.fidelity)
    console.log("firstResult.diversity")
    console.log(firstResult.diversity)
    console.log("firstResult.privacy")
    console.log(firstResult.privacy)

    // Display metrics per il primo dataset
    if (firstResult.fidelity) displayFidelityMetrics(firstResult.fidelity);
    if (firstResult.diversity) displayDiversityMetrics(firstResult.diversity);
    if (firstResult.privacy) displayPrivacyMetrics(firstResult.privacy);

    if (firstResult.summaries && (typeof firstResult.summaries === 'object')) {
        displaySummary(firstResult.summaries);
    } else {
        const container = document.getElementById('summaryContent');
        if (container) {
            container.innerHTML = `<p class="metric-info">No dataset summary returned by the server.</p>`;
        }
    }
    // Aggiungi selettore per dataset augmented
    addDatasetSelector(results.analyses.length);

    // Create heatmaps
    console.log("firstResult.fidelity");
    console.log(firstResult.fidelity);
    console.log("firstResult.privacy");
    console.log(firstResult.privacy);

    createInfoLossHeatmap(firstResult.fidelity);
    createPrivacyRiskHeatmap(firstResult.privacy);
}

// Display Fidelity Metrics
function displayFidelityMetrics(fidelity) {
    const container = document.getElementById('fidelityMetrics');
    container.innerHTML = '';

    if (fidelity.error) {
        container.innerHTML = `<p class="metric-info">Errore: ${fidelity.error}</p>`;
        return;
    }

    // KS Test
    if (fidelity.kolmogorov_smirnov) {
        const ks = fidelity.kolmogorov_smirnov;
        container.appendChild(createMetricCard(
            'KS Similarity',
            //(ks.similarity_score * 100).toFixed(1) + '%',
            (ks.similarity_score).toFixed(3),
            'ks_similarity',
            getBadgeClass(ks.similarity_score, 0.5, 0.3)
        ));
    }


    // Q-Function
    if (fidelity.q_function) {
        const qf = fidelity.q_function;
        container.appendChild(createMetricCard(
            'Q-Function Score',
            qf.q_score.toFixed(3),
            'q_function',
            qf.privacy_quality
        ));
        //console.log(qf.privacy_quality);
        //console.log(qf.q_score.toFixed(3))
    }

    //Statistical Moments
    if (fidelity.statistical_moments) {
        const sm = fidelity.statistical_moments;
        container.appendChild(createMetricCard(
            'Statistical Moments',
            sm.average_moment_difference.toFixed(3),
            'statistical_moments',
            sm.stat_badge
        ));

    }


    // MMD
    if (fidelity.mmd_score) {
        const mmd = fidelity.mmd_score;
        container.appendChild(createMetricCard(
            'MMD Score',
            mmd.mmd_score.toFixed(3),
            'mmd_score',
            mmd.quality
        ));
    }

    // PCA Similarity
    if (fidelity.pca_comparison) {
        const pca = fidelity.pca_comparison;
        container.appendChild(createMetricCard(
            'PCA Similarity',
            //(pca.average_component_similarity * 100).toFixed(1) + '%',
            (pca.average_component_similarity).toFixed(3),
            'pca_similarity',
            getBadgeClass(pca.average_component_similarity, 0.7, 0.5)
        ));
    }

    // KL Divergence
    if (fidelity.kl_divergence) {
        const kl = fidelity.kl_divergence;
        container.appendChild(createMetricCard(
            'KL Divergence',
            kl.average_kl_divergence.toFixed(3),
            'kl_divergence',
            kl.quality
        ));
    }

    // JS Divergence
    if (fidelity.js_divergence) {
        const js = fidelity.js_divergence;
        container.appendChild(createMetricCard(
            'JS Divergence',
            js.average_js_divergence.toFixed(3),
            'js_divergence',
            js.quality
        ));
    }

    // Mean Distance
    if (fidelity.mean_absolute_distance) {
        const dist = fidelity.mean_absolute_distance;
        container.appendChild(createMetricCard(
            'Mean Absolute Distance',
            dist.overall_distance.toFixed(3),
            'mean_absolute_distance',
            getBadgeClass(1 - dist.overall_distance, 0.7, 0.5)
        ));
    }

    // Create chart
    createFidelityChart(fidelity);
}

// Display Diversity Metrics
function displayDiversityMetrics(diversity) {
    const container = document.getElementById('diversityMetrics');
    container.innerHTML = '';

    if (diversity.error) {
        container.innerHTML = `<p class="metric-info">Errore: ${diversity.error}</p>`;
        return;
    }

    // Feature Entropy
    if (diversity.feature_entropy) {
        const ent = diversity.feature_entropy;
        container.appendChild(createMetricCard(
            'Feature Entropy',
            ent.average_entropy.toFixed(2),
            'feature_entropy',
            getBadgeClass(ent.average_entropy / 4, 0.6, 0.4)
        ));
    }

    // Coverage
    if (diversity.coverage) {
        const cov = diversity.coverage;
        container.appendChild(createMetricCard(
            'Coverage Ratio',
            (cov.coverage_ratio * 100).toFixed(1) + '%',
            'coverage_ratio',
            getBadgeClass(cov.coverage_ratio, 0.7, 0.5)
        ));
    }

    // Silhouette Score
    if (diversity.silhouette_score) {
        const sil = diversity.silhouette_score;
        container.appendChild(createMetricCard(
            'Silhouette Score',
            sil.augmented_score.toFixed(3),
            'silhouette_score',
            sil.quality
        ));
    }

    // Davies-Bouldin Index
    if (diversity.davies_bouldin_index) {
        const db = diversity.davies_bouldin_index;
        container.appendChild(createMetricCard(
            'Davies-Bouldin Index',
            db.augmented_score.toFixed(3),
            'davies_bouldin_index',
            db.quality
        ));
    }

    // Intra-Class Compactness
    if (diversity.intra_class_compactness) {
        const icc = diversity.intra_class_compactness;
        container.appendChild(createMetricCard(
            'Intra-Class Compactness',
            icc.augmented_icc.toFixed(3),
            'intra_class_compactness',
            icc.quality
        ));
    }

    // Cluster Spread
    if (diversity.cluster_spread) {
        const spread = diversity.cluster_spread;
        container.appendChild(createMetricCard(
            'Cluster Uniformity',
            (spread.augmented_uniformity * 100).toFixed(1) + '%',
            'cluster_spread',
            getBadgeClass(spread.augmented_uniformity, 0.7, 0.5)
        ));
    }

    // Intra-diversity
    if (diversity.intra_diversity) {
        const intra = diversity.intra_diversity;
        container.appendChild(createMetricCard(
            'Diversity Score',
            intra.diversity_score.toFixed(2),
            'intra_diversity',
            getBadgeClass(intra.diversity_score / 10, 0.5, 0.3)
        ));
    }

    // Range Coverage
    if (diversity.range_coverage) {
        const range = diversity.range_coverage;
        container.appendChild(createMetricCard(
            'Range Coverage',
            (range.average_coverage * 100).toFixed(1) + '%',
            'range_coverage',
            getBadgeClass(range.average_coverage, 0.8, 0.6)
        ));
    }

    createDiversityChart(diversity);
}

// Display Privacy Metrics
function displayPrivacyMetrics(privacy) {
    const container = document.getElementById('privacyMetrics');
    container.innerHTML = '';

    if (privacy.error) {
        container.innerHTML = `<p class="metric-info">Errore: ${privacy.error}</p>`;
        return;
    }

    if (privacy.attribute_disclosure) {
        const ad = privacy.attribute_disclosure;
        container.appendChild(createMetricCard(
            'Attribute Disclosure',
            ad.average_attribute_risk.toFixed(3),
            'attribute_disclosure',
            ad.overall_risk
        ));
    }

    // Membership Inference
    if (privacy.membership_inference) {
        const mi = privacy.membership_inference;
        container.appendChild(createMetricCard(
            'Privacy Score',
            (mi.privacy_score * 100).toFixed(1) + '%',
            'membership_inference',
            mi.privacy_level
        ));
    }


    // Nearest Neighbor Risk
    if (privacy.nearest_neighbor_risk) {
        const nn = privacy.nearest_neighbor_risk;
        container.appendChild(createMetricCard(
            'NN Disclosure Risk',
            (nn.high_risk_ratio * 100).toFixed(1) + '%',
            'nearest_neighbor_risk',
            nn.risk_level === 'low' ? 'high' : nn.risk_level === 'medium' ? 'medium' : 'low'
        ));
    }

    // Distance to Closest
    if (privacy.distance_to_closest) {
        const dcr = privacy.distance_to_closest;
        container.appendChild(createMetricCard(
            'Mean DCR',
            dcr.mean_dcr.toFixed(3),
            'distance_to_closest',
            getBadgeClass(dcr.mean_dcr / 5, 0.6, 0.4)
        ));
    }

    // Uniqueness
    if (privacy.uniqueness) {
        const uniq = privacy.uniqueness;
        container.appendChild(createMetricCard(
            'Uniqueness Score',
            (uniq.average_uniqueness * 100).toFixed(1) + '%',
            'uniqueness',
            uniq.privacy_quality === 'good' ? 'high' : uniq.privacy_quality === 'fair' ? 'medium' : 'low'
        ));
    }

    // k-anonymity
    if (privacy.k_anonymity) {
        const k = privacy.k_anonymity;
        const kDisplay = k.k_anonymity !== undefined ? k.k_anonymity : '—';
        const eqClasses = k.n_equivalence_classes || 0;
        const medianSize = k.median_equivalence_class_size !== undefined ? k.median_equivalence_class_size.toFixed(1) : '—';

        container.appendChild(createMetricCard(
            'k-Anonymity',
            kDisplay,
            'k_anonymity',
            k.k_anonymity >= 10 ? 'high' : k.k_anonymity >= 3 ? 'medium' : 'low'
        ));

    }

    // l-diversity
    if (privacy.l_diversity) {
        const ld = privacy.l_diversity;
        const avgDistinct = ld.average_distinct_sensitive_values !== undefined ? Number(ld.average_distinct_sensitive_values).toFixed(2) : '—';
        const avgEntropy = ld.average_entropy_bits !== undefined ? Number(ld.average_entropy_bits).toFixed(2) : '—';
        const nClasses = ld.n_equivalence_classes || 0;

        container.appendChild(createMetricCard(
            'l-Diversity',
            avgDistinct,
            'l_diversity',
            avgDistinct >= 5 ? 'high' : avgDistinct >= 2.5 ? 'medium' : 'low'
        ));

    }

    createPrivacyChart(privacy);
}

// Display Summary
function displaySummary(summaries) {
    const container = document.getElementById('summaryContent');
    container.innerHTML = '';

    if (!summaries || typeof summaries !== 'object') {
        container.innerHTML = `<p class="metric-info">Summary not available for this analysis.</p>`;
        return;
    }

    const safeNumber = (v) => (typeof v === 'number' ? v.toLocaleString() : '—');
    const safeFixed = (v, dp = 2) => (typeof v === 'number' ? v.toFixed(dp) : '—');

    console.log("Summaries");
    console.log(summaries);

    ['original', 'augmented'].forEach(type => {
        const summary = summaries[type];
        const section = document.createElement('div');
        section.className = 'summary-section';

        if (!summary || typeof summary !== 'object') {
            section.innerHTML = `
                <h4>${type.charAt(0).toUpperCase() + type.slice(1)} Dataset</h4>
                <p class="metric-info">No summary returned for ${type}.</p>
            `;
            container.appendChild(section);
            return;
        }

        // Basic info table
        section.innerHTML = `
            <h4>${summary.name || (type.charAt(0).toUpperCase() + type.slice(1))} Dataset</h4>
            <table class="summary-table">
                <tr><td>Rows</td><td>${safeNumber(summary.n_rows)}</td></tr>
                <tr><td>Columns</td><td>${safeNumber(summary.n_columns)}</td></tr>
                <tr><td>Numeric Columns</td><td>${safeNumber(summary.n_numeric)}</td></tr>
                <tr><td>Categorical Columns</td><td>${safeNumber(summary.n_categorical)}</td></tr>
                <tr><td>Missing Values</td><td>${safeNumber(summary.total_missing)}</td></tr>
                <tr><td>Memory Usage</td><td>${summary.memory_usage_mb ? safeFixed(summary.memory_usage_mb, 2) + ' MB' : '—'}</td></tr>
            </table>
        `;

        // Column Types Visualization
        if (summary.columns && summary.columns.length > 0) {
            const columnTypesSection = document.createElement('div');
            columnTypesSection.className = 'summary-plot-section';
            columnTypesSection.innerHTML = `
                <h5>📊 Column Types Distribution</h5>
                <div id="columnTypesChart-${type}" class="plot-container"></div>
            `;
            section.appendChild(columnTypesSection);

            // Create column types chart
            setTimeout(() => {
                createColumnTypesChart(summary, type);
            }, 100);
        }

        // Missing Values Visualization
        if (summary.missing_values && Object.keys(summary.missing_values).length > 0) {
            const missingValuesSection = document.createElement('div');
            missingValuesSection.className = 'summary-plot-section';
            missingValuesSection.innerHTML = `
                <h5>🔍 Missing Values by Column</h5>
                <div id="missingValuesChart-${type}" class="plot-container"></div>
            `;
            section.appendChild(missingValuesSection);

            // Create missing values chart
            setTimeout(() => {
                createMissingValuesChart(summary, type);
            }, 150);
        }

        // Numeric Statistics Visualization
        if (summary.numeric_stats && Object.keys(summary.numeric_stats).length > 0) {
            const numericStatsSection = document.createElement('div');
            numericStatsSection.className = 'summary-plot-section';
            numericStatsSection.innerHTML = `
                <h5>📈 Numeric Columns Statistics</h5>
                <div id="numericStatsChart-${type}" class="plot-container"></div>
            `;
            section.appendChild(numericStatsSection);

            // Create numeric stats chart
            setTimeout(() => {
                createNumericStatsChart(summary, type);
            }, 200);
        }

        // Distribution Comparison (if both original and augmented available)
        if (type === 'augmented' && summaries.original && summaries.original.numeric_stats) {
            const distributionSection = document.createElement('div');
            distributionSection.className = 'summary-plot-section';
            distributionSection.innerHTML = `
                <h5>🔄 Distribution Comparison (Original vs Augmented)</h5>
                <div id="distributionComparisonChart" class="plot-container"></div>
            `;
            section.appendChild(distributionSection);

            // Create distribution comparison chart
            setTimeout(() => {
                createDistributionComparisonChart(summaries.original, summary);
            }, 250);
        }

        container.appendChild(section);
    });
}

// Helper function to create column types chart
function createColumnTypesChart(summary, type) {
    const numericCount = summary.n_numeric || 0;
    const categoricalCount = summary.n_categorical || 0;
    const totalColumns = summary.n_columns || 0;
    const otherCount = totalColumns - numericCount - categoricalCount;

    const data = [{
        values: [numericCount, categoricalCount, otherCount],
        labels: ['Numeric', 'Categorical', 'Other'],
        type: 'pie',
        hole: 0.4,
        marker: {
            colors: ['#4f46e5', '#10b981', '#6b7280']
        },
        textinfo: 'label+percent',
        hoverinfo: 'label+value+percent'
    }];

    const layout = {
        //title: 'Column Types',
        showlegend: false,
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: document.body.dataset.theme === 'dark' ? '#f1f5f9' : '#0f172a' },
        height: 400
    };

    Plotly.newPlot(`columnTypesChart-${type}`, data, layout, { responsive: true });
}

// Helper function to create missing values chart
function createMissingValuesChart(summary, type) {
    const columns = Object.keys(summary.missing_values || {});
    const missingCounts = Object.values(summary.missing_values || {});
    const totalRows = summary.n_rows || 1;

    // Calculate missing percentages
    const missingPercentages = missingCounts.map(count => (count / totalRows) * 100);

    const data = [{
        x: columns,
        y: missingPercentages,
        type: 'bar',
        marker: {
            color: missingPercentages.map(p => p > 0 ? '#ef4444' : '#10b981'),
            line: {
                color: document.body.dataset.theme === 'dark' ? '#334155' : '#cbd5e1',
                width: 1
            }
        },
        hovertemplate: 'Column: %{x}<br>Missing: %{customdata} values (%{y:.1f}%)<extra></extra>',
        customdata: missingCounts
    }];

    const layout = {
        title: 'Missing Values Percentage by Column',
        xaxis: {
            title: 'Columns',
            tickangle: -45
        },
        yaxis: {
            title: 'Missing Values (%)',
            range: [0, Math.max(...missingPercentages) * 1.1]
        },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: document.body.dataset.theme === 'dark' ? '#f1f5f9' : '#0f172a' },
        height: 350
    };

    Plotly.newPlot(`missingValuesChart-${type}`, data, layout, { responsive: true });
}

// Helper function to create numeric statistics chart
function createNumericStatsChart(summary, type) {
    const numericStats = summary.numeric_stats || {};
    const columns = Object.keys(numericStats);

    if (columns.length === 0) return;

    // Prepare data for box plot
    const data = columns.map(column => {
        const stats = numericStats[column];
        return {
            y: [
                stats.min,
                stats['25%'],
                stats['50%'],
                stats['75%'],
                stats.max
            ],
            name: column,
            type: 'box',
            boxpoints: false,
            hoverinfo: 'y+name',
            marker: {
                color: '#4f46e5'
            },
            line: {
                color: document.body.dataset.theme === 'dark' ? '#f1f5f9' : '#0f172a'
            }
        };
    });

    const layout = {
        title: 'Distribution of Numeric Columns',
        xaxis: {
            title: 'Columns',
            tickangle: -45
        },
        yaxis: { title: 'Values' },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: document.body.dataset.theme === 'dark' ? '#f1f5f9' : '#0f172a' },
        height: 400,
        showlegend: false
    };

    Plotly.newPlot(`numericStatsChart-${type}`, data, layout, { responsive: true });
}

// Helper function to create distribution comparison chart
function createDistributionComparisonChart(originalSummary, augmentedSummary) {
    const originalStats = originalSummary.numeric_stats || {};
    const augmentedStats = augmentedSummary.numeric_stats || {};
    const commonColumns = Object.keys(originalStats).filter(col => augmentedStats[col]);

    if (commonColumns.length === 0) return;

    const data = commonColumns.map(column => {
        const origMean = originalStats[column].mean;
        const augMean = augmentedStats[column].mean;
        const origStd = originalStats[column].std;
        const augStd = augmentedStats[column].std;

        return {
            type: 'scatter',
            mode: 'markers',
            x: [origMean],
            y: [augMean],
            error_x: {
                type: 'data',
                array: [origStd],
                color: '#ef4444'
            },
            error_y: {
                type: 'data',
                array: [augStd],
                color: '#10b981'
            },
            name: column,
            marker: {
                size: 12,
                opacity: 0.7
            },
            hovertemplate: `Column: ${column}<br>Original: ${origMean.toFixed(2)} ± ${origStd.toFixed(2)}<br>Augmented: ${augMean.toFixed(2)} ± ${augStd.toFixed(2)}<extra></extra>`
        };
    });

    // Add perfect correlation line
    const maxValue = Math.max(
        ...Object.values(originalStats).map(s => s.mean + s.std),
        ...Object.values(augmentedStats).map(s => s.mean + s.std)
    );

    data.push({
        x: [0, maxValue],
        y: [0, maxValue],
        type: 'scatter',
        mode: 'lines',
        line: {
            color: '#6b7280',
            dash: 'dash',
            width: 1
        },
        name: 'Perfect Correlation',
        showlegend: false
    });

    const layout = {
        title: 'Mean Values Comparison (Original vs Augmented)',
        xaxis: {
            title: 'Original Dataset Mean ± Std',
            range: [0, maxValue * 1.1]
        },
        yaxis: {
            title: 'Augmented Dataset Mean ± Std',
            range: [0, maxValue * 1.1]
        },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: document.body.dataset.theme === 'dark' ? '#f1f5f9' : '#0f172a' },
        height: 500,
        showlegend: true,
        legend: {
            orientation: 'h',
            y: -0.2
        }
    };

    Plotly.newPlot('distributionComparisonChart', data, layout, { responsive: true });
}


// ==================== METRIC CARDS WITH TOOLTIPS ====================

function createMetricCard(title, value, metricKey, badgeClass) {
    const card = document.createElement('div');
    card.className = 'metric-card';

    const tooltip = createTooltip(metricKey);
    const badge = badgeClass ? `<span class="metric-badge badge-${badgeClass}">${badgeClass.toUpperCase()}</span>` : '';

    card.innerHTML = `
        <div class="metric-title">
            ${title}
            ${tooltip}
        </div>
        <div class="metric-value">${value}</div>
        ${badge}
    `;

    return card;
}

function createTooltip(metricKey) {
    // Find metric description
    let desc = null;
    for (const category in metricDescriptions) {
        if (metricDescriptions[category][metricKey]) {
            desc = metricDescriptions[category][metricKey];
            break;
        }
    }

    if (!desc) return '';

    return `
        <div class="tooltip-container">
            <span class="tooltip-icon">ℹ️</span>
            <div class="tooltip-content">
                <div class="tooltip-title">${desc.icon || ''} ${desc.title}</div>
                <div class="tooltip-desc">${desc.description}</div>
                <div class="tooltip-interp">${desc.interpretation}</div>
            </div>
        </div>
    `;
}


// Helper: Get Badge Class
function getBadgeClass(value, highThreshold, mediumThreshold) {
    if (value >= highThreshold) return 'high';
    if (value >= mediumThreshold) return 'medium';
    return 'low';
}

// ==================== CHARTS ====================

// Create Charts
function createFidelityChart(fidelity) {
    const ctx = document.getElementById('fidelityChart');
    if (charts.fidelity) charts.fidelity.destroy();

    const theme = document.body.dataset.theme;
    const textColor = theme === 'dark' ? '#f1f5f9' : '#0f172a';

    const data = {
        labels: ['KS Similarity', 'PCA Similarity', 'MDD', 'Mean Absolute Dist', 'KL Div', 'JS Div', "Q-Function", "Statistical Moments"],
        datasets: [{
            label: 'Fidelity Scores',
            data: [
                fidelity.kolmogorov_smirnov?.similarity_score * 100 || 0,
                fidelity.pca_comparison?.average_component_similarity * 100 || 0,
                fidelity.mmd_score?.mmd_score * 100 || 0,
                (1 - (fidelity.mean_absolute_distance?.overall_distance || 0)) * 100,
                fidelity.kl_divergence?.similarity_score * 100 || 0,
                fidelity.js_divergence?.similarity_score * 100 || 0,
                fidelity.q_function?.privacy_score * 100 || 0,
                fidelity.statistical_moments?.average_moment_difference * 100 || 0
            ],
            backgroundColor: ['#4f46e5', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#ef4444', '#3b82f6'],
            borderWidth: 0
        }]
    };

    charts.fidelity = new Chart(ctx, {
        type: 'bar',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false },
                title: { display: true, text: 'Fidelity Components', color: '#f1f5f9' }
            },
            scales: {
                y: { beginAtZero: true, max: 100, ticks: { color: '#94a3b8' }, grid: { color: '#334155' } },
                x: { ticks: { color: '#94a3b8' }, grid: { display: false } }
            }
        }
    });
}

function createDiversityChart(diversity) {
    const ctx = document.getElementById('diversityChart');
    if (charts.diversity) charts.diversity.destroy();

    const theme = document.body.dataset.theme;
    const textColor = theme === 'dark' ? '#f1f5f9' : '#0f172a';

    const data = {
        labels: ['Entropy', 'Coverage', 'Silhouette', 'DB Index', 'ICC', 'Uniformity', 'Intra Diversity', 'Feature Range Coverage'],
        datasets: [{
            label: 'Diversity Scores',
            data: [
                (diversity.feature_entropy?.average_entropy / 4) * 100 || 0,
                diversity.coverage?.coverage_ratio * 100 || 0,
                ((diversity.silhouette_score?.augmented_score + 1) / 2) * 100 || 0,
                (1 / (1 + (diversity.davies_bouldin_index?.augmented_score || 0))) * 100,
                (1 / (1 + (diversity.intra_class_compactness?.augmented_icc || 0))) * 100,
                diversity.cluster_spread?.augmented_uniformity * 100 || 0,
                (diversity.intra_diversity?.diversity_score / 10) * 100 || 0,
                diversity.range_coverage?.average_coverage * 100 || 0,
            ],
            backgroundColor: '#10b981',
            borderColor: '#059669',
            borderWidth: 2
        }]
    };

    charts.diversity = new Chart(ctx, {
        type: 'radar',
        data: data,
        options: {
            responsive: true,
            plugins: {
                legend: { display: false },
                title: { display: true, text: 'Diversity Components', color: '#f1f5f9' }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { color: '#94a3b8', backdropColor: 'transparent' },
                    grid: { color: '#334155' },
                    pointLabels: { color: '#f1f5f9' }
                }
            }
        }
    });
}

function createPrivacyChart(privacy) {
    const ctx = document.getElementById('privacyChart');
    if (charts.privacy) charts.privacy.destroy();
    const theme = document.body.dataset.theme;
    const textColor = theme === 'dark' ? '#f1f5f9' : '#0f172a';

    // Gather values with safe fallbacks
    const membership = (privacy.membership_inference?.privacy_score || 0) * 100;
    const nnr = (privacy.nearest_neighbor_risk?.high_risk_ratio || 0) * 100;
    const dsk = (privacy.attribute_disclosure?.average_attribute_risk || 0) * 100;

    const uniqueness = (privacy.uniqueness?.average_uniqueness || 0) * 100;
    const dcrNorm = ((privacy.distance_to_closest?.mean_dcr || 0) / 5) * 100; // same normalization as before

    // k-anonymity and l-diversity values (scaled to chart-friendly numbers)
    const kVal = privacy.k_anonymity?.k_anonymity !== undefined ? Number(privacy.k_anonymity.k_anonymity) : 0;
    // scale k into 0-100 (clamp): kScaled = 100 if k>=20 ; otherwise (k/20)*100
    const kScaled = Math.min(100, (kVal / 20) * 100);

    const lAvgDistinct = privacy.l_diversity?.average_distinct_sensitive_values !== undefined ? Number(privacy.l_diversity.average_distinct_sensitive_values) : 0;
    const lScaled = Math.min(100, (lAvgDistinct / 10) * 100); // treat 10 distinct values as "100"

    const labels = ['Privacy Score', 'Uniqueness', 'DCR (norm)', 'k-anonymity (scaled)', 'l-diversity (scaled)', 'NNR', 'Attribute Disclosure'];
    const values = [membership, uniqueness, dcrNorm, kScaled, lScaled, nnr, dsk];

    const data = {
        labels: labels,
        datasets: [{
            label: 'Privacy Components',
            data: values,
            backgroundColor: ['#4f46e5', '#ec4899', '#f59e0b', '#06b6d4', '#a78bfa', '#10b981', '#e11d48'],
            borderWidth: 0
        }]
    };

    charts.privacy = new Chart(ctx, {
        type: 'doughnut',
        data: data,
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'bottom', labels: { color: textColor } },
                title: { display: true, text: 'Privacy Components', color: textColor },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            const label = context.label || '';
                            const v = context.raw;
                            // Provide readable tooltip for k and l
                            if (label.includes('k-anonymity')) {
                                const kRaw = kVal !== undefined ? kVal : '—';
                                return `${label}: ${v.toFixed(1)} (k = ${kRaw})`;
                            }
                            if (label.includes('l-diversity')) {
                                const lRaw = lAvgDistinct !== undefined ? lAvgDistinct.toFixed(2) : '—';
                                return `${label}: ${v.toFixed(1)} (avg distinct = ${lRaw})`;
                            }
                            return `${label}: ${v.toFixed(1)}`;
                        }
                    }
                }
            }
        }
    });
}


function createUtilityChart(utility) {
    const ctx = document.getElementById('utilityChart');
    if (charts.utility) charts.utility.destroy();

    const theme = document.body.dataset.theme;
    const textColor = theme === 'dark' ? '#f1f5f9' : '#0f172a';

    const models = utility.model_results.map(m => m.model_type);
    const origAcc = utility.model_results.map(m => m.original_only.accuracy * 100);
    const augAcc = utility.model_results.map(m => m.augmented_only.accuracy * 100);
    const combAcc = utility.model_results.map(m => m.combined.accuracy * 100);

    const data = {
        labels: models,
        datasets: [
            {
                label: 'Original Only',
                data: origAcc,
                backgroundColor: '#ef4444'
            },
            {
                label: 'Augmented Only',
                data: augAcc,
                backgroundColor: '#f59e0b'
            },
            {
                label: 'Combined',
                data: combAcc,
                backgroundColor: '#10b981'
            }
        ]
    };

    charts.utility = new Chart(ctx, {
        type: 'bar',
        data: data,
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Model Performance Comparison',
                    color: textColor
                },
                legend: {
                    labels: { color: textColor }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Accuracy (%)',
                        color: textColor
                    },
                    ticks: { color: textColor },
                    grid: { color: theme === 'dark' ? '#334155' : '#e2e8f0' }
                },
                x: {
                    ticks: { color: textColor },
                    grid: { display: false }
                }
            }
        }
    });
}

// Heroes v3.0 - Frontend JavaScript (Part 3)
// Heatmaps, Utility Analysis, Suggestions, and Exports

// ==================== HEATMAPS ====================

function createInfoLossHeatmap(fidelity) {
    if (!fidelity.kl_divergence || !fidelity.kl_divergence.per_feature) return;

    const perFeature = fidelity.kl_divergence.per_feature;
    const features = Object.keys(perFeature);
    const values = Object.values(perFeature);

    // Create matrix for heatmap (single row)
    const z = [values];

    const data = [{
        z: z,
        x: features,
        y: ['KL Divergence'],
        type: 'heatmap',
        colorscale: [
            [0, '#10b981'],
            [0.5, '#f59e0b'],
            [1, '#ef4444']
        ],
        hovertemplate: 'Feature: %{x}<br>KL Div: %{z:.4f}<extra></extra>'
    }];

    const layout = {
        title: 'Information Loss by Feature',
        xaxis: { title: 'Features' },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: document.body.dataset.theme === 'dark' ? '#f1f5f9' : '#0f172a' }
    };

    Plotly.newPlot('infoLossHeatmap', data, layout, { responsive: true });
}

function createPrivacyRiskHeatmap(privacy) {
    if (!privacy.attribute_disclosure || !privacy.attribute_disclosure.per_attribute) return;

    const perAttr = privacy.attribute_disclosure.per_attribute;
    const features = Object.keys(perAttr);
    const values = features.map(f => perAttr[f].normalized_difference);

    // Create matrix (single row)
    const z = [values];

    const data = [{
        z: z,
        x: features,
        y: ['Privacy Risk'],
        type: 'heatmap',
        colorscale: [
            [0, '#ef4444'],  // Low risk (close to original) = red
            [0.5, '#f59e0b'],
            [1, '#10b981']   // High risk (far from original) = green
        ],
        hovertemplate: 'Feature: %{x}<br>Risk: %{z:.4f}<extra></extra>'
    }];

    const layout = {
        title: 'Privacy Risk by Feature (Lower = More Privacy)',
        xaxis: { title: 'Features' },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: document.body.dataset.theme === 'dark' ? '#f1f5f9' : '#0f172a' }
    };

    Plotly.newPlot('privacyRiskHeatmap', data, layout, { responsive: true });
}

// ==================== UTILITY ANALYSIS ====================

// Aggiorna updateSelectedModels() per includere i nuovi modelli
function updateSelectedModels() {
    selectedModels = [];
    document.querySelectorAll('.model-option.selected').forEach(opt => {
        selectedModels.push(opt.dataset.model);
    });

    // Abilita/disabilita il pulsante di training
    //document.getElementById('trainModelsBtn').disabled = selectedModels.length === 0;
}

// Aggiungi questa funzione per gestire i parametri dei modelli
// Aggiorna la funzione getModelParams() per includere i nuovi modelli
// Aggiorna la funzione getModelParams() per includere i nuovi modelli
function getModelParams(modelType) {
    const defaultParams = {
        'logistic_regression': { C: 1.0, max_iter: 1000 },
        'random_forest': { n_estimators: 100, max_depth: 10 },
        'svm': { C: 1.0, kernel: 'rbf', gamma: 'scale' },
        'knn': { n_neighbors: 5 },
        'decision_tree': { max_depth: 10 },
        'gradient_boosting': { n_estimators: 100, learning_rate: 0.1 },
        'xgboost': { n_estimators: 100, max_depth: 6, learning_rate: 0.1 },
        'lightgbm': { n_estimators: 100, max_depth: 6, learning_rate: 0.1 },
        'mlp': { hidden_layer_sizes: [100, 50], max_iter: 1000, random_state: 42 },
        'extra_trees': { n_estimators: 100, max_depth: 10 },
        'linear_svc': { C: 1.0, max_iter: 1000 },
        'naive_bayes': {},
        'ada_boost': { n_estimators: 50, learning_rate: 1.0 },
        'hist_gradient_boosting': { max_iter: 100, learning_rate: 0.1 },
        'ridge_classifier': { alpha: 1.0 },
        'sgd_classifier': { max_iter: 1000, learning_rate: 'optimal' },
        'catboost': { iterations: 100, depth: 6, learning_rate: 0.1, verbose: false },
        'lda': {},
        'qda': {},
        'passive_aggressive': { max_iter: 1000, random_state: 42 },
        'perceptron': { max_iter: 1000, random_state: 42 },
        'bagging': { n_estimators: 10, random_state: 42 },
        'calibrated_cv': { cv: 3, method: 'sigmoid' },
        'dummy': { strategy: 'stratified' },
        'gaussian_process': { random_state: 42 },
        'isolation_forest': { n_estimators: 100, contamination: 'auto', random_state: 42 }
    };

    return defaultParams[modelType] || {};
}

async function trainModels() {
    const targetColumn = document.getElementById('targetColumn').value.trim();

    if (!targetColumn) {
        alert('Please specify target column');
        return;
    }

    if (selectedModels.length === 0) {
        alert('Please select at least one model');
        return;
    }

    try {
        document.getElementById('utilityLoading').style.display = 'block';
        document.getElementById('trainModelsBtn').disabled = true;

        // Build model configs
        const modelConfigs = selectedModels.map(model => {
            return {
                type: model,
                params: getModelParams(model)
            };
        });

        console.log("[uploadedPaths.augmented]")
        console.log(uploadedPaths)
        console.log(uploadedPaths.augmented)

        // Usa il path corretto per il dataset augmented corrente
        let origPath = uploadedPaths.original || null;
        let augPath = uploadedPaths.augmented || null;

        // Determina il path corretto per il dataset augmented corrente
        if (uploadedPaths.augmented && uploadedPaths.augmented.length > 0) {
            // Se abbiamo paths dal database, usa quello corrente
            augPath = uploadedPaths.augmented[currentAugmentedIndex];
        } else if (uploadedFiles.augmented && uploadedFiles.augmented.length > 0) {
            // Se abbiamo file uploadati, usa il file corrente
            augPath = uploadedPaths.augmented ? uploadedPaths.augmented[currentAugmentedIndex] : null;
        }

        console.log("Current augmented dataset:", currentAugmentedIndex);
        console.log("augPath:", augPath);

        // Se non abbiamo il path per il file augmented corrente, carica i file
        if ((!origPath && uploadedFiles.original) || (!augPath && uploadedFiles.augmented && uploadedFiles.augmented.length > 0)) {
            const formData = new FormData();
            if (uploadedFiles.original) formData.append('original', uploadedFiles.original);

            // Aggiungi solo il file augmented corrente, non tutti
            if (uploadedFiles.augmented && uploadedFiles.augmented.length > currentAugmentedIndex) {
                formData.append('augmented', uploadedFiles.augmented[currentAugmentedIndex]);
            }

            const uploadResponse = await fetch(`${API_URL}/upload-multi`, { method: 'POST', body: formData });

            if (!uploadResponse.ok) {
                const text = await uploadResponse.text();
                throw new Error(`Upload failed: ${uploadResponse.status} ${text}`);
            }

            const uploadData = await uploadResponse.json();
            // expect backend to return original_path / augmented_path
            origPath = uploadData.original_path || origPath;
            augPath = uploadData.augmented_path || augPath;


            // save them for later use
            if (origPath) uploadedPaths.original = origPath;

            // Aggiorna il path per il dataset augmented corrente
            if (augPath) {
                if (!uploadedPaths.augmented) {
                    uploadedPaths.augmented = [];
                }
                uploadedPaths.augmented[currentAugmentedIndex] = augPath;
            }
        }

        if (!origPath || !augPath) {
            throw new Error('Missing dataset path(s). Upload files first or load from DB.');
        }

        // Send request for utility analysis
        console.log('UTILITY REQUEST', {
            original_path: origPath,
            augmented_path: augPath,
            target_column: targetColumn,
            model_configs: modelConfigs
        });

        const useCV = document.getElementById('useCV') ? document.getElementById('useCV').checked : false;
        const cvFolds = document.getElementById('cvFolds') ? parseInt(document.getElementById('cvFolds').value, 10) : 5;
        const cvMetric = document.getElementById('cvMetric') ? document.getElementById('cvMetric').value : 'accuracy';

        const payload = {
            original_path: origPath,
            augmented_path: augPath,
            target_column: targetColumn,
            model_configs: modelConfigs,
            use_cv: useCV,
            cv_folds: cvFolds,
            cv_metric: cvMetric
        };

        const response = await fetch(`${API_URL}/utility/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        // If server error: try to read error details
        if (!response.ok) {
            const txt = await response.text();
            throw new Error(`Utility analysis failed: ${response.status} ${txt}`);
        }

        const data = await response.json();

        if (data.success) {
            displayUtilityResults(data.results);

            const utilityScore = (data.results.overall_utility_score * 100).toFixed(0);
            document.getElementById('utilityScore').textContent = utilityScore;
            document.getElementById('utilityBar').style.width = `${utilityScore}%`;
            document.getElementById('utilityScoreItem').style.display = 'grid';
            document.getElementById('explainUtility').style.display = 'inline-block';

            // Salva i risultati utility per il dataset corrente
            if (!analysisResults.analyses[currentAugmentedIndex].utility) {
                analysisResults.analyses[currentAugmentedIndex].utility = {};
            }
            analysisResults.analyses[currentAugmentedIndex].utility = data.results;
        } else {
            throw new Error(`Utility analysis failed: ${data.error || JSON.stringify(data)}`);
        }

    } catch (error) {
        alert(`Error: ${error.message}`);
        console.error('trainModels error:', error);
    } finally {
        document.getElementById('utilityLoading').style.display = 'none';
        document.getElementById('trainModelsBtn').disabled = false;
    }
}

function displayUtilityResults(utility) {
    const container = document.getElementById('utilityResultsContent');

    let html = `
      <div class="utility-summary">
          <h4>Overall Utility Score: ${(utility.overall_utility_score * 100).toFixed(1)}%</h4>
          <p>${utility.recommendation.icon} ${utility.recommendation.verdict}: ${utility.recommendation.message}</p>
          ${utility.use_cv ? `<p>Cross-Validation: ${utility.cv_folds} folds, metric: ${utility.cv_metric}</p>` : ''}
      </div>

      <table class="utility-table">
          <thead>
              <tr>
                  <th>Model</th>
                  <th>Original Only</th>
                  <th>Augmented Only</th>
                  <th>Combined</th>
                  <th>Improvement</th>
                  <th>CV Mean</th>
                  <th>CV Std</th>
                  <th>Training Time</th>
              </tr>
          </thead>
          <tbody>
    `;

    utility.model_results.forEach(model => {
        const improvement = model.improvement.accuracy;
        const improvementClass = improvement > 0 ? 'improvement-positive' : 'improvement-negative';

        // cv results: may be stored per scenario as model.original_only.cv or aggregated
        const cvMean = (model.combined.cv_mean !== undefined) ? (model.combined.cv_mean * 100).toFixed(2) + '%' : 'N/A';
        const cvStd = (model.combined.cv_std !== undefined) ? (model.combined.cv_std * 100).toFixed(2) + '%' : 'N/A';

        html += `
          <tr>
              <td><strong>${model.model_type.replace('_', ' ')}</strong></td>
              <td>${(model.original_only.accuracy * 100).toFixed(2)}%</td>
              <td>${(model.augmented_only.accuracy * 100).toFixed(2)}%</td>
              <td>${(model.combined.accuracy * 100).toFixed(2)}%</td>
              <td class="${improvementClass}">${(improvement * 100).toFixed(2)}%</td>
              <td>${cvMean}</td>
              <td>${cvStd}</td>
              <td>${model.original_only.training_time_seconds.toFixed(2)}s</td>
          </tr>
        `;
    });
    html += '</tbody></table>';

    container.innerHTML = html;
    document.getElementById('utilityResults').style.display = 'block';

    // Create utility chart
    createUtilityChart(utility);
}

// ==================== SUGGESTIONS ====================

async function generateSuggestions() {
    if (!analysisResults) {
        alert('Please run analysis first');
        return;
    }

    try {
        document.getElementById('suggestionsLoading').style.display = 'block';
        document.getElementById('generateSuggestions').disabled = true;

        const response = await fetch(`${API_URL}/suggestions/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: selectedModel,
                metrics: analysisResults,
                analysis_id: analysisResults.analysis_id
            })
        });

        const data = await response.json();

        if (data.success) {
            displaySuggestions(data.suggestions);
        } else {
            alert(`Failed to generate suggestions: ${data.error}`);
        }

    } catch (error) {
        alert(`Error: ${error.message}`);
        console.error(error);
    } finally {
        document.getElementById('suggestionsLoading').style.display = 'none';
        document.getElementById('generateSuggestions').disabled = false;
    }
}

function displaySuggestions(suggestionsText) {
    const container = document.getElementById('suggestionsContent');

    // Parse suggestions from markdown format
    const lines = suggestionsText.split('\n');
    let html = '';

    lines.forEach(line => {
        line = line.trim();
        if (!line || line.length < 10) return;

        // Try to extract priority and category
        let priority = 'medium';
        let category = 'general';
        let text = line;

        if (line.toUpperCase().includes('[PRIORITY:')) {
            if (line.toUpperCase().includes('HIGH')) priority = 'high';
            else if (line.toUpperCase().includes('LOW')) priority = 'low';
        }

        if (line.toUpperCase().includes('[CATEGORY:')) {
            if (line.toUpperCase().includes('FIDELITY')) category = 'fidelity';
            else if (line.toUpperCase().includes('DIVERSITY')) category = 'diversity';
            else if (line.toUpperCase().includes('PRIVACY')) category = 'privacy';
            else if (line.toUpperCase().includes('UTILITY')) category = 'utility';
        }

        // Clean text
        text = text.replace(/\[PRIORITY:.*?\]/gi, '')
            .replace(/\[CATEGORY:.*?\]/gi, '')
            .replace(/HIGH\]/gi, '')
            .replace(/MEDIUM\]/gi, '')
            .replace(/LOW\]/gi, '')
            .replace(/FIDELITY\]/gi, '')
            .replace(/DIVERSITY\]/gi, '')
            .replace(/PRIVACY\]/gi, '')
            .replace(/UTILITY\]/gi, '')
            .trim();

        if (text) {
            html += `
                <div class="suggestion-card priority-${priority}">
                    <div class="suggestion-header">
                        <span class="suggestion-type">${category}</span>
                        <span class="suggestion-priority ${priority}">${priority}</span>
                    </div>
                    <div class="suggestion-text">${text}</div>
                </div>
            `;
        }
    });

    container.innerHTML = html;
    document.getElementById('suggestionsContainer').style.display = 'block';
}

// ==================== EXPLAINABILITY ====================
// Load Ollama Models
async function loadOllamaModels() {
    try {
        const response = await fetch(`${API_URL}/ollama/models`);
        const data = await response.json();

        console.log('OLLAMA MODELS RESPONSE:', response.status, data);

        const select = document.getElementById('ollamaModel');
        const statusEl = document.getElementById('ollamaStatus');

        if (data.available && data.models && data.models.length > 0) {
            select.innerHTML = data.models.map(model =>
                `<option value="${model}" ${model === selectedModel ? 'selected' : ''}>${model}</option>`
            ).join('');

            statusEl.textContent = `✅ Ollama is available with ${data.models.length} model(s)`;
            statusEl.style.background = 'rgba(16, 185, 129, 0.1)';
            statusEl.style.borderColor = 'var(--secondary)';

            select.addEventListener('change', (e) => { selectedModel = e.target.value; });
            // set default if empty
            if (!selectedModel) selectedModel = data.models[0];
        } else {
            select.innerHTML = '<option value="">No models available</option>';
            statusEl.textContent = '⚠️ Ollama not available. Make sure it\'s running on http://localhost:11434';
            statusEl.style.background = 'rgba(239, 68, 68, 0.1)';
            statusEl.style.borderColor = 'var(--danger)';
            console.warn('Ollama fetch result:', data);
        }
    } catch (error) {
        console.error('Error loading Ollama models', error);
        const statusEl = document.getElementById('ollamaStatus');
        statusEl.textContent = '❌ Error connecting to Ollama';
        statusEl.style.background = 'rgba(239, 68, 68, 0.1)';
    }
}

// Explanation level
document.getElementById('explanationLevel').addEventListener('change', (e) => {
    explanationLevel = e.target.value;
});

document.getElementById('refreshModels').addEventListener('click', loadOllamaModels);

// Explainability Functions
document.getElementById('explainFidelity').addEventListener('click', () => {
    explainMetrics('fidelity');
});

document.getElementById('explainDiversity').addEventListener('click', () => {
    explainMetrics('diversity');
});

document.getElementById('explainPrivacy').addEventListener('click', () => {
    explainMetrics('privacy');
});

document.getElementById('explainCustom').addEventListener('click', () => {
    const customPrompt = document.getElementById('customPrompt').value;
    if (customPrompt.trim()) {
        explainWithCustomPrompt(customPrompt);
    } else {
        alert('Please enter a question');
    }
});

document.getElementById('closeExplanation').addEventListener('click', () => {
    document.getElementById('explanationResult').style.display = 'none';
});

async function explainMetrics(category) {
    if (!analysisResults) {
        alert('Please run analysis first');
        return;
    }

    const idx = parseInt(document.getElementById("augmentedDatasetSelect").value, 10);
    const metrics = analysisResults.analyses?.[idx]?.[category];
    const prompt = buildPrompt(category, metrics);

    await sendExplanationRequest(prompt);
}

async function explainCustom() {
    const customPrompt = document.getElementById('customPrompt').value;
    if (!customPrompt.trim()) {
        alert('Please enter a question');
        return;
    }

    const fullPrompt = `
Context - Analysis Results:
Fidelity: ${JSON.stringify(analysisResults.fidelity, null, 2)}
Diversity: ${JSON.stringify(analysisResults.diversity, null, 2)}
Privacy: ${JSON.stringify(analysisResults.privacy, null, 2)}
${analysisResults.utility ? `Utility: ${JSON.stringify(analysisResults.utility, null, 2)}` : ''}

User Question: ${customPrompt}
`;

    await sendExplanationRequest(fullPrompt);
}

function buildPrompt(category, metrics) {
    const levelSuffix = {
        low: '\n\nProvide a brief, high-level summary (2-3 sentences max).',
        medium: '\n\nProvide a balanced explanation with key insights and recommendations.',
        detailed: '\n\nProvide comprehensive analysis with technical details, implications, and specific recommendations.'
    };

    const basePrompts = {
        fidelity: `Analyze fidelity metrics: ${JSON.stringify(metrics, null, 2)}\n\nExplain distribution similarity, reconstruction quality, and issues.`,
        diversity: `Analyze diversity metrics: ${JSON.stringify(metrics, null, 2)}\n\nExplain variety, coverage, and cluster quality.`,
        privacy: `Analyze privacy metrics: ${JSON.stringify(metrics, null, 2)}\n\nExplain re-identification risks and privacy protection.`,
        utility: `Analyze utility metrics: ${JSON.stringify(metrics, null, 2)}\n\nExplain ML performance, improvements, and recommendations.`
    };

    return basePrompts[category] + levelSuffix[explanationLevel];
}

async function sendExplanationRequest(prompt) {
    const loading = document.getElementById('explainLoading');
    const resultDiv = document.getElementById('explanationResult');
    const contentDiv = document.getElementById('explanationContent');
    const modelSpan = document.getElementById('explanationModel');

    try {
        loading.style.display = 'block';
        resultDiv.style.display = 'none';

        const response = await fetch(`${API_URL}/explain`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: selectedModel,
                prompt: prompt,
                metrics: analysisResults
            })
        });

        const data = await response.json();

        if (data.success) {
            // Render markdown
            contentDiv.innerHTML = marked.parse(data.explanation);
            modelSpan.textContent = `Model: ${data.model} | Detail: ${explanationLevel}`;
            resultDiv.style.display = 'block';
        } else {
            alert(`Error: ${data.error}\n${data.details || ''}`);
        }

    } catch (error) {
        alert(`Failed to get explanation: ${error.message}`);
    } finally {
        loading.style.display = 'none';
    }
}

// ==================== EXPORTS ====================

async function exportJSON() {
    if (!analysisResults) {
        alert('No results to export');
        return;
    }

    const dataStr = JSON.stringify(analysisResults, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `auras_analysis_${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
}

async function exportPDF() {
    if (!analysisResults) {
        alert('No results to export');
        return;
    }

    try {
        const chartImages = await captureCharts();

        const response = await fetch(`${API_URL}/export/pdf`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                results: analysisResults,
                charts: chartImages
            })
        });

        if (response.ok) {
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `auras_report_${Date.now()}.pdf`;
            link.click();
            URL.revokeObjectURL(url);
        } else {
            alert('PDF export failed');
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

async function exportExcel() {
    if (!analysisResults) {
        alert('No results to export');
        return;
    }

    try {
        const response = await fetch(`${API_URL}/export/excel`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ results: analysisResults })
        });

        if (response.ok) {
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `auras_report_${Date.now()}.xlsx`;
            link.click();
            URL.revokeObjectURL(url);
        } else {
            alert('Excel export failed');
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

async function captureScreenshot() {
    try {
        const node = document.getElementById('resultsSection');

        const dataUrl = await htmlToImage.toPng(node, {
            cacheBust: true,
            backgroundColor: document.body.dataset.theme === 'dark'
                ? '#0f172a'
                : '#f8fafc'
        });

        const link = document.createElement('a');
        link.download = `auras_screenshot_${Date.now()}.png`;
        link.href = dataUrl;
        link.click();
    } catch (error) {
        alert(`Failed: ${error.message}`);
    }
}

async function captureCharts() {
    const chartImages = {};

    try {
        const chartIds = ['fidelityChart', 'diversityChart', 'privacyChart', 'utilityChart'];

        for (const chartId of chartIds) {
            const canvas = document.getElementById(chartId);
            if (canvas && canvas.offsetParent !== null) {
                chartImages[chartId.replace('Chart', '')] = canvas.toDataURL('image/png');
            }
        }
    } catch (error) {
        console.error('Error capturing charts:', error);
    }

    return chartImages;
}

// ==================== HISTORY ====================

async function viewAnalysisHistory() {
    try {
        const response = await fetch(`${API_URL}/history?limit=10`);
        const data = await response.json();

        if (data.success && data.history.length > 0) {
            let html = '<h3>Recent Analyses</h3><ul>';
            data.history.forEach(item => {
                html += `<li>${new Date(item.timestamp).toLocaleString()} - ${item.dataset_name} (Score: ${item.overall_score.toFixed(2)})</li>`;
            });
            html += '</ul>';

            alert(html);  // In production, use a proper modal
        } else {
            alert('No analysis history found');
        }
    } catch (error) {
        alert(`Error loading history: ${error.message}`);
    }
}



// Gestione file augmented multipli
function handleAugmentedFiles(e) {
    const files = Array.from(e.target.files);
    uploadedFiles.augmented = files;

    const filesList = document.getElementById('augmentedFilesList');
    filesList.innerHTML = '';

    files.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'augmented-file-item';
        fileItem.innerHTML = `
            <span>${file.name}</span>
            <button class="remove-file-btn" data-index="${index}">×</button>
        `;
        filesList.appendChild(fileItem);
    });

    document.getElementById('augmentedFilesName').textContent = `${files.length} file(s) selected`;
    checkFilesReady();

    // Aggiungi event listener per rimuovere file
    document.querySelectorAll('.remove-file-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const index = parseInt(e.target.dataset.index);
            removeAugmentedFile(index);
        });
    });
}

function removeAugmentedFile(index) {
    uploadedFiles.augmented.splice(index, 1);
    // Re-renderizza la lista
    const files = document.getElementById('augmentedFiles').files;
    const fileList = new DataTransfer();

    uploadedFiles.augmented.forEach(file => {
        fileList.items.add(file);
    });

    document.getElementById('augmentedFiles').files = fileList.files;
    handleAugmentedFiles({ target: document.getElementById('augmentedFiles') });
}

// Aggiorna checkFilesReady
function checkFilesReady() {
    const ready = (uploadedFiles.original && uploadedFiles.augmented) ||
        (uploadedPaths.original && uploadedPaths.augmented);
    document.getElementById('analyzeBtn').disabled = !ready;
}

function addDatasetSelector(count) {
    // Rimuovi selettore esistente
    const existingSelector = document.getElementById('datasetSelector');
    if (existingSelector) existingSelector.remove();

    if (count > 1) {
        const selector = document.createElement('div');
        selector.id = 'datasetSelector';
        selector.className = 'dataset-selector';

        // Crea opzioni con nomi dei file
        const options = Array.from({ length: count }, (_, i) => {
            const fileName = uploadedFiles.augmented[i] ? uploadedFiles.augmented[i].name : `Augmented Dataset ${i + 1}`;
            return `<option value="${i}">${fileName}</option>`;
        }).join('');

        selector.innerHTML = `
            <label for="augmentedDatasetSelect">Select Augmented Dataset:</label>
            <select id="augmentedDatasetSelect">
                ${options}
            </select>
        `;

        // Inserisci dopo il score-card
        const scoreCard = document.querySelector('.score-card');
        scoreCard.parentNode.insertBefore(selector, scoreCard.nextSibling);

        // Aggiungi event listener
        document.getElementById('augmentedDatasetSelect').addEventListener('change', (e) => {
            currentAugmentedIndex = parseInt(e.target.value);
            updateDisplayedDataset(currentAugmentedIndex);
        });

        console.log(`🎯 Selettore dataset creato con ${count} opzioni`);
    }
}

function updateWeightsTotal() {
    const fidelity = parseInt(document.getElementById('weightFidelity').value);
    const diversity = parseInt(document.getElementById('weightDiversity').value);
    const privacy = parseInt(document.getElementById('weightPrivacy').value);

    const total = fidelity + diversity + privacy;
    const totalSpan = document.getElementById('weightsTotal');
    const warning = document.getElementById('weightsWarning');
    const analyzeBtn = document.getElementById('analyzeBtn');

    totalSpan.textContent = total + '%';

    if (total === 100) {
        totalSpan.classList.add('valid');
        totalSpan.classList.remove('invalid');
        warning.style.display = 'none';

        // Update global weights
        userWeights = {
            fidelity: fidelity / 100,
            diversity: diversity / 100,
            privacy: privacy / 100
        };

        // Enable analyze button if files are ready
        checkFilesReady();
    } else {
        totalSpan.classList.remove('valid');
        totalSpan.classList.add('invalid');
        warning.style.display = 'block';
        analyzeBtn.disabled = true;
    }
}

function resetWeights() {
    document.getElementById('weightFidelity').value = 33;
    document.getElementById('weightDiversity').value = 33;
    document.getElementById('weightPrivacy').value = 34;

    document.getElementById('weightFidelityValue').textContent = '33%';
    document.getElementById('weightDiversityValue').textContent = '33%';
    document.getElementById('weightPrivacyValue').textContent = '34%';

    updateWeightsTotal();
}

function setPriorityWeight(priority) {
    const weights = {
        fidelity: 12,
        diversity: 13,
        privacy: 12
    };

    if (priority === 'fidelity') {
        weights.fidelity = 75;
        weights.diversity = 12;
        weights.privacy = 13;
    } else if (priority === 'diversity') {
        weights.fidelity = 13;
        weights.diversity = 75;
        weights.privacy = 12;
    } else if (priority === 'privacy') {
        weights.fidelity = 12;
        weights.diversity = 13;
        weights.privacy = 75;
    }

    document.getElementById('weightFidelity').value = weights.fidelity;
    document.getElementById('weightDiversity').value = weights.diversity;
    document.getElementById('weightPrivacy').value = weights.privacy;

    document.getElementById('weightFidelityValue').textContent = weights.fidelity + '%';
    document.getElementById('weightDiversityValue').textContent = weights.diversity + '%';
    document.getElementById('weightPrivacyValue').textContent = weights.privacy + '%';

    updateWeightsTotal();
}

function updateDisplayedDataset(index) {
    if (!analysisResults.analyses || !analysisResults.analyses[index]) {
        console.error('Dataset non trovato:', index);
        return;
    }

    const result = analysisResults.analyses[index];
    console.log(`🔄 Aggiornamento visualizzazione per dataset ${index + 1}`, result);

    // Aggiorna overall score
    const aggregate = result.aggregate_score;
    document.getElementById('overallScore').textContent = (aggregate.overall * 100).toFixed(0);
    document.getElementById('overallRating').textContent = aggregate.rating;

    // Aggiorna category scores
    if (aggregate.scores.fidelity !== undefined) {
        const fid = (aggregate.scores.fidelity * 100).toFixed(0);
        document.getElementById('fidelityScore').textContent = fid;
        document.getElementById('fidelityBar').style.width = `${fid}%`;
    }

    if (aggregate.scores.diversity !== undefined) {
        const div = (aggregate.scores.diversity * 100).toFixed(0);
        document.getElementById('diversityScore').textContent = div;
        document.getElementById('diversityBar').style.width = `${div}%`;
    }

    if (aggregate.scores.privacy !== undefined) {
        const priv = (aggregate.scores.privacy * 100).toFixed(0);
        document.getElementById('privacyScore').textContent = priv;
        document.getElementById('privacyBar').style.width = `${priv}%`;
    }

    // Aggiorna metriche fidelity
    if (result.fidelity) {
        displayFidelityMetrics(result.fidelity);
        createFidelityChart(result.fidelity);
    } else {
        document.getElementById('fidelityMetrics').innerHTML = '<p class="metric-info">Dati fidelity non disponibili</p>';
        if (charts.fidelity) charts.fidelity.destroy();
    }

    // Aggiorna metriche diversity
    if (result.diversity) {
        displayDiversityMetrics(result.diversity);
        createDiversityChart(result.diversity);
    } else {
        document.getElementById('diversityMetrics').innerHTML = '<p class="metric-info">Dati diversity non disponibili</p>';
        if (charts.diversity) charts.diversity.destroy();
    }

    // Aggiorna metriche privacy
    if (result.privacy) {
        displayPrivacyMetrics(result.privacy);
        createPrivacyChart(result.privacy);
    } else {
        document.getElementById('privacyMetrics').innerHTML = '<p class="metric-info">Dati privacy non disponibili</p>';
        if (charts.privacy) charts.privacy.destroy();
    }

    // Aggiorna summary
    if (result.summaries && (typeof result.summaries === 'object')) {
        displaySummary(result.summaries);
    } else {
        document.getElementById('summaryContent').innerHTML = '<p class="metric-info">Summary non disponibile</p>';
    }

    // Aggiorna heatmaps
    if (result.fidelity) {
        createInfoLossHeatmap(result.fidelity);
    } else {
        document.getElementById('infoLossHeatmap').innerHTML = '<p class="metric-info">Dati fidelity non disponibili per heatmap</p>';
    }

    if (result.privacy) {
        createPrivacyRiskHeatmap(result.privacy);
    } else {
        document.getElementById('privacyRiskHeatmap').innerHTML = '<p class="metric-info">Dati privacy non disponibili per heatmap</p>';
    }

    // Aggiorna utility se presente
    // AGGIORNA UTILITY - QUESTA È LA PARTE CRITICA
    if (result.utility) {
        // Se ci sono risultati utility per questo dataset, mostrali
        displayUtilityResults(result.utility);
        createUtilityChart(result.utility);

        const utilityScore = (result.utility.overall_utility_score * 100).toFixed(0);
        document.getElementById('utilityScore').textContent = utilityScore;
        document.getElementById('utilityBar').style.width = `${utilityScore}%`;
        document.getElementById('utilityScoreItem').style.display = 'true';
        document.getElementById('explainUtility').style.display = 'inline-block';

        console.log(`✅ Utility aggiornata per dataset ${index + 1}: ${utilityScore}%`);
    } else {
        // Se non ci sono risultati utility per questo dataset, nascondi tutto
        document.getElementById('utilityScoreItem').style.display = 'none';
        document.getElementById('explainUtility').style.display = 'none';
        document.getElementById('utilityResults').style.display = 'none';

        // Reset dei contenuti utility
        document.getElementById('utilityResultsContent').innerHTML = '';

        if (charts.utility) {
            charts.utility.destroy();
            charts.utility = null;
        }

        console.log(`ℹ️ Nessun risultato utility per dataset ${index + 1}`);
    }

    console.log(`✅ Visualizzazione aggiornata per dataset augmented ${index + 1}`);
}