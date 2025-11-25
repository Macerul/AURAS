"""
AURAS - Data Augmentation Quality Assessment Tool
Flask Backend API
"""
from flask_cors import CORS
import os
import json
import numpy as np
from werkzeug.utils import secure_filename
import traceback
from metrics.fidelity import calculate_fidelity_metrics
from metrics.diversity import calculate_diversity_metrics
from metrics.privacy import calculate_privacy_metrics
from metrics.utility import calculate_utility_metrics
from utils.data_loader import load_dataset, validate_datasets, get_dataset_summary
import pandas as pd
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE, ADASYN
from scipy import stats
from sklearn.neighbors import KernelDensity
from flask import Flask, request, jsonify, send_from_directory, send_file
from db_manager import DatabaseManager, load_from_external_db, test_db_connection
import traceback

class NumpyEncoder(json.JSONEncoder):
    """Encoder personalizzato per gestire tipi NumPy"""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app = Flask(__name__, static_folder='../frontend', static_url_path='')
app.json_encoder = NumpyEncoder
CORS(app)

# Configurazione
UPLOAD_FOLDER = '/tmp/heroes_uploads'
ALLOWED_EXTENSIONS = {'csv', 'parquet', 'pq'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Initialize database
db_manager = DatabaseManager()

def allowed_file(filename):
    """Verifica se il file ha un'estensione consentita"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Serve la pagina principale"""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'Heroes API'})


@app.route('/api/upload', methods=['POST'])
def upload_files():
    """
    Upload dei due dataset (originale e aumentato)
    """
    try:
        # Verifica presenza dei file
        if 'original' not in request.files or 'augmented' not in request.files:
            return jsonify({
                'error': 'Entrambi i file (original e augmented) sono richiesti'
            }), 400
        
        file_original = request.files['original']
        file_augmented = request.files['augmented']
        
        # Verifica nomi file
        if file_original.filename == '' or file_augmented.filename == '':
            return jsonify({'error': 'Nome file non valido'}), 400
        
        # Verifica estensioni
        if not (allowed_file(file_original.filename) and allowed_file(file_augmented.filename)):
            return jsonify({
                'error': f'Estensioni consentite: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Salva i file
        orig_filename = secure_filename(file_original.filename)
        aug_filename = secure_filename(file_augmented.filename)
        
        orig_path = os.path.join(app.config['UPLOAD_FOLDER'], f'original_{orig_filename}')
        aug_path = os.path.join(app.config['UPLOAD_FOLDER'], f'augmented_{aug_filename}')
        
        file_original.save(orig_path)
        file_augmented.save(aug_path)
        
        # Carica i dataset
        df_original = load_dataset(orig_path)
        df_augmented = load_dataset(aug_path)
        
        # Valida i dataset
        validation = validate_datasets(df_original, df_augmented)
        
        if not validation['valid']:
            return jsonify({
                'error': 'Validazione fallita',
                'details': validation
            }), 400
        
        # Genera sommari
        summary_original = get_dataset_summary(df_original, "Original")
        summary_augmented = get_dataset_summary(df_augmented, "Augmented")

        
        return jsonify({
            'success': True,
            'message': 'File caricati con successo',
            'original_path': orig_path,
            'augmented_path': aug_path,
            'validation': validation,
            'summaries': {
                'original': summary_original,
                'augmented': summary_augmented
            }
        })
    
    except Exception as e:
        return jsonify({
            'error': 'Errore durante l\'upload',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/upload-multi', methods=['POST'])
def upload_files_multi():
    """
    Upload di dataset multipli (originale + N augmented)
    Nuovo endpoint per la funzionalità multi-dataset comparison
    """
    try:
        # Verifica presenza file originale
        if 'original' not in request.files:
            return jsonify({
                'error': 'File originale richiesto'
            }), 400

        print("[MULTI] Upload request received:", request.files)
        file_original = request.files['original']
        augmented_files = request.files.getlist('augmented')

        if len(augmented_files) == 0 or augmented_files[0].filename == '':
            return jsonify({'error': 'Almeno un file aumentato richiesto'}), 400

        # Verifica nomi file
        if file_original.filename == '':
            return jsonify({'error': 'Nome file originale non valido'}), 400

        # Verifica estensioni
        if not allowed_file(file_original.filename):
            return jsonify({
                'error': f'Estensione file originale non consentita: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400

        for file in augmented_files:
            if not allowed_file(file.filename):
                return jsonify({
                    'error': f'Estensione file aumentato non consentita: {", ".join(ALLOWED_EXTENSIONS)}'
                }), 400

        # Salva file originale
        orig_filename = secure_filename(file_original.filename)
        orig_path = os.path.join(app.config['UPLOAD_FOLDER'], f'original_{orig_filename}')
        file_original.save(orig_path)

        # Salva file augmented
        aug_paths = []
        for i, file in enumerate(augmented_files):
            aug_filename = secure_filename(file.filename)
            aug_path = os.path.join(app.config['UPLOAD_FOLDER'], f'augmented_{i}_{aug_filename}')
            file.save(aug_path)
            aug_paths.append(aug_path)

        # Carica e valida dataset
        df_original = load_dataset(orig_path)
        validation_results = []

        for aug_path in aug_paths:
            df_augmented = load_dataset(aug_path)
            validation = validate_datasets(df_original, df_augmented)
            validation_results.append(validation)

        # Genera sommari
        summary_original = get_dataset_summary(df_original, "Original")
        summary_augmented = [
            get_dataset_summary(load_dataset(path), f"Augmented {i + 1}")
            for i, path in enumerate(aug_paths)
        ]

        return jsonify({
            'success': True,
            'message': f'File caricati con successo: {len(aug_paths)} dataset aumentati',
            'original_path': orig_path,
            'augmented_paths': aug_paths,
            'validation': validation_results,
            'summaries': {
                'original': summary_original,
                'augmented': summary_augmented
            }
        })

    except Exception as e:
        return jsonify({
            'error': 'Errore durante l\'upload',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Analizza i dataset e calcola tutte le metriche
    """
    try:
        data = request.json
        orig_path = data.get('original_path')
        aug_path = data.get('augmented_path')
        
        if not orig_path or not aug_path:
            return jsonify({'error': 'Path dei file non forniti'}), 400
        
        # Verifica esistenza file
        if not os.path.exists(orig_path) or not os.path.exists(aug_path):
            return jsonify({'error': 'File non trovati'}), 404
        
        # Carica i dataset
        df_original = load_dataset(orig_path)
        df_augmented = load_dataset(aug_path)
        
        # Calcola tutte le metriche
        results = {}
        
        # Fidelity metrics
        try:
            results['fidelity'] = calculate_fidelity_metrics(df_original, df_augmented)
        except Exception as e:
            results['fidelity'] = {'error': str(e)}
        
        # Diversity metrics
        try:
            results['diversity'] = calculate_diversity_metrics(df_original, df_augmented)
        except Exception as e:
            results['diversity'] = {'error': str(e)}
        
        # Privacy metrics
        try:
            results['privacy'] = calculate_privacy_metrics(df_original, df_augmented)
        except Exception as e:
            results['privacy'] = {'error': str(e)}

        try:
            summary_original = get_dataset_summary(df_original, "Original")
        except Exception as e:
            summary_original = {'error': f'Error generating summary for original: {str(e)}'}

        try:
            summary_augmented = get_dataset_summary(df_augmented, "Augmented")
        except Exception as e:
            summary_augmented = {'error': f'Error generating summary for augmented: {str(e)}'}

        results['summaries'] = {
            'original': summary_original,
            'augmented': summary_augmented
        }
        
        # Calcola score aggregato
        results['aggregate_score'] = calculate_aggregate_score(results)

        # Save to database
        try:
            dataset_name = os.path.basename(orig_path)
            analysis_id = db_manager.save_analysis(
                dataset_name=dataset_name,
                original_rows=len(df_original),
                augmented_rows=len(df_augmented),
                fidelity_score=results['aggregate_score']['scores'].get('fidelity', 0),
                diversity_score=results['aggregate_score']['scores'].get('diversity', 0),
                privacy_score=results['aggregate_score']['scores'].get('privacy', 0),
                overall_score=results['aggregate_score']['overall'],
                rating=results['aggregate_score']['rating'],
                metrics_dict=results
            )
            results['analysis_id'] = analysis_id
        except Exception as db_error:
            print(f"Database save error: {db_error}")
            # Continue even if DB save fails

        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'error': 'Errore durante l\'analisi',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500


def calculate_aggregate_score(results):
    """
    Calcola uno score aggregato basato su tutte le metriche,
    includendo k-anonymity e l-diversity nelle componenti privacy.
    """
    scores = {}

    # Fidelity score (più alto = migliore somiglianza)
    if 'error' not in results.get('fidelity', {}):
        fid = results['fidelity']
        fid_score = 0
        if 'kolmogorov_smirnov' in fid:
            fid_score += fid['kolmogorov_smirnov'].get('similarity_score', 0) * 0.3
        if 'mmd_score' in fid:
            mmd = fid['mmd_score'].get('mmd_score', 0)
            fid_score += max(0, 1 - mmd) * 0.3
        if 'pca_comparison' in fid:
            fid_score += fid['pca_comparison'].get('average_component_similarity', 0) * 0.3
        if 'mean_absolute_distance' in fid:
            fid_score += fid['mean_absolute_distance'].get('overall_distance', 0) * 0.3
        if 'kl_divergence' in fid:
            fid_score += fid['kl_divergence'].get('average_kl_divergence', 0) * 0.3
        if 'js_divergence' in fid:
            fid_score += fid['js_divergence'].get('average_js_divergence', 0) * 0.3
        scores['fidelity'] = min(1.0, max(0.0, fid_score))

    # Diversity score (più alto = più diversità)
    if 'error' not in results.get('diversity', {}):
        div = results['diversity']
        div_score = 0
        if 'feature_entropy' in div:
            div_score += min(1.0, div['feature_entropy'].get('average_entropy', 0) / 4) * 0.3
        if 'coverage' in div:
            div_score += div['coverage'].get('coverage_ratio', 0) * 0.3
        if 'cluster_spread' in div:
            div_score += div['cluster_spread'].get('augmented_uniformity', 0) * 0.2
        if 'intra_diversity' in div:
            div_score += min(1.0, div['intra_diversity'].get('diversity_score', 0) / 10) * 0.2
        if 'range_coverage' in div:
            div_score += div['range_coverage'].get('average_coverage', 0) * 0.2
        if 'silhouette_score' in div:
            div_score += div['silhouette_score'].get('similarity', 0) * 0.2
        if 'davies_bouldin_index' in div:
            div_score += div['davies_bouldin_index'].get('similarity', 0) * 0.2
        if 'intra_class_compactness' in div:
            div_score += div['intra_class_compactness'].get('similarity', 0) * 0.2
        scores['diversity'] = min(1.0, max(0.0, div_score))

    # Privacy score (più alto = migliore privacy)
    if 'error' not in results.get('privacy', {}):
        priv = results['privacy']
        priv_score = 0.0

        if 'nearest_neighbor_risk' in priv:
            priv_score += priv['nearest_neighbor_risk'].get('high_risk_ratio', 0) * 0.35
        # membership_inference: original weight ~0.4 -> reduce a 0.35
        if 'membership_inference' in priv:
            priv_score += priv['membership_inference'].get('privacy_score', 0) * 0.35

        if 'attribute_disclosure' in priv:
            priv_score += priv['attribute_disclosure'].get('average_attribute_risk', 0) * 0.35

        if 'q_function' in priv:
            priv_score += priv['q_function'].get('privacy_score', 0) * 0.35

        # uniqueness: keep contribution but slightly reduced
        if 'uniqueness' in priv:
            priv_score += priv['uniqueness'].get('average_uniqueness', 0) * 0.25

        # distance to closest (mean_dcr) normalized on a reasonable scale (e.g., 0..5)
        if 'distance_to_closest' in priv:
            dcr = priv['distance_to_closest'].get('mean_dcr', 0)
            priv_score += min(1.0, dcr / 5) * 0.15

        # k-anonymity: normalize k (treat k >= 20 as "good")
        if 'k_anonymity' in priv and isinstance(priv['k_anonymity'], dict):
            try:
                k_val = float(priv['k_anonymity'].get('k_anonymity', 0))
                k_norm = min(20.0, max(0.0, k_val)) / 20.0
                priv_score += k_norm * 0.15
            except Exception:
                pass

        # l-diversity: use average distinct sensitive values (treat 10 distinct as excellent)
        if 'l_diversity' in priv and isinstance(priv['l_diversity'], dict):
            try:
                l_avg = float(priv['l_diversity'].get('average_distinct_sensitive_values', 0))
                l_norm = min(10.0, max(0.0, l_avg)) / 10.0
                priv_score += l_norm * 0.10
            except Exception:
                pass

        # clamp
        scores['privacy'] = min(1.0, max(0.0, priv_score))

    # Score complessivo (media semplice delle categorie disponibili)
    if len(scores) > 0:
        overall = sum(scores.values()) / len(scores)
    else:
        overall = 0.0

    return {
        'scores': scores,
        'overall': float(overall),
        'rating': get_rating(overall)
    }



def get_rating(score):
    """Converte lo score in rating testuale"""
    if score >= 0.8:
        return 'Excellent'
    elif score >= 0.6:
        return 'Good'
    elif score >= 0.4:
        return 'Fair'
    elif score >= 0.2:
        return 'Adequate'
    else:
        return 'Poor'


@app.route('/api/export', methods=['POST'])
def export_results():
    """
    Esporta i risultati in formato JSON
    """
    try:
        data = request.json
        results = data.get('results')
        
        if not results:
            return jsonify({'error': 'Results not provided'}), 400
        
        return jsonify({
            'success': True,
            'data': results
        })
    
    except Exception as e:
        return jsonify({
            'error': 'Export error',
            'details': str(e)
        }), 500


def get_ollama_url():
    """
    Restituisce l'URL di Ollama in base all'ambiente
    - In Docker: usa il nome del servizio 'ollama'
    - In sviluppo locale: usa 'localhost'
    """
    # Controlla se siamo in un container Docker
    in_docker = os.path.exists('/.dockerenv')

    if in_docker:
        # Nel container Docker, usa il nome del servizio
        return 'http://ollama:11434'
    else:
        # In sviluppo locale, usa localhost
        return 'http://localhost:11434'

@app.route('/api/explain', methods=['POST'])
def explain_with_ollama():
    """
    Spiega le metriche usando Ollama
    """
    try:
        data = request.json
        model = data.get('model', 'llama3')
        prompt = data.get('prompt', '')
        metrics = data.get('metrics', {})

        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400

        # Invia richiesta a Ollama
        import requests as req

        #ollama_url = 'http://localhost:11434/api/generate'
        # USA L'URL DINAMICO
        ollama_base_url = get_ollama_url()
        ollama_url = f'{ollama_base_url}/api/generate'

        payload = {
            'model': model,
            'prompt': prompt,
            'stream': False
        }

        response = req.post(ollama_url, json=payload, timeout=60)

        if response.status_code != 200:
            return jsonify({
                'error': 'Ollama request failed',
                'details': f'Status code: {response.status_code}'
            }), 500

        ollama_response = response.json()
        explanation = ollama_response.get('response', '')

        return jsonify({
            'success': True,
            'explanation': explanation,
            'model': model
        })

    except req.exceptions.ConnectionError:
        return jsonify({
            'error': 'Cannot connect to Ollama',
            'details': 'Make sure Ollama is running on http://localhost:11434'
        }), 503

    except req.exceptions.Timeout:
        return jsonify({
            'error': 'Ollama request timeout',
            'details': 'The model took too long to respond'
        }), 504

    except Exception as e:
        return jsonify({
            'error': 'Explanation error',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/ollama/models', methods=['GET'])
def get_ollama_models():
    """
    Ottiene la lista dei modelli disponibili in Ollama (gestisce più formati)
    """
    try:
        import requests as req

        # Provo prima /api/tags poi /api/models come fallback
        '''
        candidates = [
            'http://localhost:11434/api/tags',
            'http://localhost:11434/api/models',
            'http://localhost:11434/api/list'
        ]
        '''
        # USA L'URL DINAMICO
        ollama_base_url = get_ollama_url()

        # Provo prima /api/tags poi /api/models come fallback
        candidates = [
            f'{ollama_base_url}/api/tags',
            f'{ollama_base_url}/api/models',
            f'{ollama_base_url}/api/list'
        ]

        models = []
        last_resp = None

        for url in candidates:
            try:
                resp = req.get(url, timeout=3)
                last_resp = resp
                if resp.status_code != 200:
                    continue
                body = resp.json()
                print("Ollama Request")
                print(body)

                # Possibili formati:
                # 1) {"models": [{"name": "llama2"}, ...]}
                if isinstance(body, dict) and 'models' in body:
                    for m in body.get('models', []):
                        # supportare sia dict con name sia stringhe
                        if isinstance(m, dict) and 'name' in m:
                            models.append(m['name'])
                        elif isinstance(m, str):
                            models.append(m)
                # 2) lista semplice: ["llama2", "llama3"]
                elif isinstance(body, list):
                    for m in body:
                        if isinstance(m, dict) and 'name' in m:
                            models.append(m['name'])
                        elif isinstance(m, str):
                            models.append(m)
                # 3) fallback: dict di tags -> keys
                elif isinstance(body, dict):
                    # tenta estrarre nomi noti
                    for v in body.values():
                        if isinstance(v, (str,)):
                            models.append(v)
                # se abbiamo almeno uno, usiamo questo risultato
                if models:
                    break
            except Exception:
                continue

        if not models:
            return jsonify({
                'success': False,
                'available': False,
                'models': [],
                'error': f'No models found. Last response status: {getattr(last_resp, "status_code", None)}'
            }), 200

        return jsonify({
            'success': True,
            'available': True,
            'models': models
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'available': False,
            'models': [],
            'error': str(e)
        }), 500


@app.route('/api/db/test', methods=['POST'])
def test_db():
    """Test external database connection"""
    try:
        db_config = request.json
        result = test_db_connection(db_config)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/db/load', methods=['POST'])
def load_from_db():
    """Load dataset from external database"""
    try:
        data = request.json
        db_config = data.get('db_config')
        dataset_type = data.get('type', 'original')  # 'original' or 'augmented'

        df = load_from_external_db(db_config)

        # Save to temporary file
        temp_filename = f"{dataset_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        df.to_csv(temp_path, index=False)

        summary = get_dataset_summary(df, f"{dataset_type.title()} (from DB)")

        return jsonify({
            'success': True,
            'path': temp_path,
            'summary': summary,
            'rows': len(df),
            'columns': len(df.columns)
        })

    except Exception as e:
        return jsonify({
            'error': 'Database load failed',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/utility/analyze', methods=['POST'])
def analyze_utility():
    """
    Analyze utility by training ML models
    """
    try:
        data = request.json
        orig_path = data.get('original_path')
        aug_path = data.get('augmented_path')
        target_column = data.get('target_column')
        model_configs = data.get('model_configs', [])

        # new CV fields (optional)
        use_cv = bool(data.get('use_cv', False))
        cv_folds = int(data.get('cv_folds', 5))
        cv_metric = data.get('cv_metric', 'accuracy')

        if not orig_path or not aug_path or not target_column:
            return jsonify({'error': 'Missing required parameters'}), 400

        # Load datasets
        df_original = load_dataset(orig_path)
        df_augmented = load_dataset(aug_path)

        # Calculate utility metrics
        # Make sure calculate_utility_metrics signature supports use_cv, cv_folds, cv_metric
        utility_results = calculate_utility_metrics(
            df_original,
            df_augmented,
            target_column,
            model_configs,
            use_cv=use_cv,
            cv_folds=cv_folds,
            cv_metric=cv_metric
        )

        # Attach CV meta so front-end can show it
        if isinstance(utility_results, dict):
            utility_results['use_cv'] = use_cv
            utility_results['cv_folds'] = cv_folds
            utility_results['cv_metric'] = cv_metric

        return jsonify({
            'success': True,
            'results': utility_results
        })

    except Exception as e:
        return jsonify({
            'error': 'Utility analysis failed',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500



@app.route('/api/suggestions/generate', methods=['POST'])
def generate_suggestions():
    """
    Generate improvement suggestions using Ollama
    """
    try:
        import requests as req

        data = request.json
        model = data.get('model', 'llama3')
        metrics = data.get('metrics', {})
        analysis_id = data.get('analysis_id')

        # Build comprehensive prompt
        prompt = build_improvement_prompt(metrics)

        #ollama_url = 'http://localhost:11434/api/generate'
        # USA L'URL DINAMICO
        ollama_base_url = get_ollama_url()
        ollama_url = f'{ollama_base_url}/api/generate'

        payload = {
            'model': model,
            'prompt': prompt,
            'stream': False
        }

        response = req.post(ollama_url, json=payload, timeout=90)

        if response.status_code != 200:
            return jsonify({
                'error': 'Ollama request failed',
                'details': f'Status code: {response.status_code}'
            }), 500

        ollama_response = response.json()
        suggestions_text = ollama_response.get('response', '')

        # Parse suggestions and save to database
        if analysis_id:
            suggestions = parse_suggestions(suggestions_text)
            for sugg in suggestions:
                db_manager.save_suggestion(
                    analysis_id=analysis_id,
                    suggestion_type=sugg.get('type', 'general'),
                    text=sugg.get('text', ''),
                    priority=sugg.get('priority', 'medium')
                )

        return jsonify({
            'success': True,
            'suggestions': suggestions_text,
            'model': model
        })

    except Exception as e:
        return jsonify({
            'error': 'Suggestion generation failed',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get analysis history"""
    try:
        limit = request.args.get('limit', 10, type=int)
        history = db_manager.get_analysis_history(limit)

        return jsonify({
            'success': True,
            'history': history
        })
    except Exception as e:
        return jsonify({
            'error': 'Failed to fetch history',
            'details': str(e)
        }), 500


@app.route('/api/suggestions/history', methods=['GET'])
def get_suggestions_history():
    """Get suggestion history"""
    try:
        analysis_id = request.args.get('analysis_id', type=int)
        limit = request.args.get('limit', 10, type=int)

        suggestions = db_manager.get_suggestions(analysis_id, limit)

        return jsonify({
            'success': True,
            'suggestions': suggestions
        })
    except Exception as e:
        return jsonify({
            'error': 'Failed to fetch suggestions',
            'details': str(e)
        }), 500


def build_improvement_prompt(metrics):
    """Build prompt for improvement suggestions"""
    fidelity = metrics.get('fidelity', {})
    diversity = metrics.get('diversity', {})
    privacy = metrics.get('privacy', {})
    utility = metrics.get('utility', {})

    # Extract k and l if present
    k_val = privacy.get('k_anonymity', {}).get('k_anonymity') if isinstance(privacy.get('k_anonymity'), dict) else None
    l_avg = privacy.get('l_diversity', {}).get('average_distinct_sensitive_values') if isinstance(privacy.get('l_diversity'), dict) else None
    l_entropy = privacy.get('l_diversity', {}).get('average_entropy_bits') if isinstance(privacy.get('l_diversity'), dict) else None

    prompt = f"""You are an expert in synthetic data generation and data augmentation.
Analyze these quality metrics and provide specific, actionable improvement suggestions.

FIDELITY METRICS:
- KS Similarity: {fidelity.get('kolmogorov_smirnov', {}).get('similarity_score', 'N/A')}
- KL Divergence: {fidelity.get('kl_divergence', {}).get('average_kl_divergence', 'N/A')}
- JS Divergence: {fidelity.get('js_divergence', {}).get('average_js_divergence', 'N/A')}

DIVERSITY METRICS:
- Feature Entropy: {diversity.get('feature_entropy', {}).get('average_entropy', 'N/A')}
- Coverage: {diversity.get('coverage', {}).get('coverage_ratio', 'N/A')}
- Silhouette Score: {diversity.get('silhouette_score', {}).get('augmented_score', 'N/A')}

PRIVACY METRICS:
- Membership-Inference Privacy Score: {privacy.get('membership_inference', {}).get('privacy_score', 'N/A')}
- Q-Function Score: {privacy.get('q_function', {}).get('q_score', 'N/A')}
- Uniqueness (avg): {privacy.get('uniqueness', {}).get('average_uniqueness', 'N/A')}
- Mean DCR: {privacy.get('distance_to_closest', {}).get('mean_dcr', 'N/A')}
- NN Risk Ratio: {privacy.get('nearest_neighbor_risk', {}).get('high_risk_ratio', 'N/A')}
"""

    # add k/l details if present
    if k_val is not None:
        prompt += f"- k-Anonymity (k): {k_val} (interpretation: minimum equivalence class size after QI binning)\n"
    else:
        prompt += "- k-Anonymity (k): N/A\n"

    if l_avg is not None or l_entropy is not None:
        prompt += f"- l-Diversity (avg distinct sensitive values): {l_avg if l_avg is not None else 'N/A'}\n"
        prompt += f"- l-Diversity (avg entropy bits): {l_entropy if l_entropy is not None else 'N/A'}\n"
    else:
        prompt += "- l-Diversity: N/A\n"

    if utility:
        prompt += f"""UTILITY METRICS:
- Overall Utility Score: {utility.get('overall_utility_score', 'N/A')}
- Average Improvement: {utility.get('average_improvement', {}).get('accuracy', 'N/A')}
"""

    prompt += """
Provide 5-7 specific, prioritized suggestions for improving the synthetic data quality.
Format each suggestion as:
[PRIORITY: HIGH/MEDIUM/LOW] [CATEGORY: Fidelity/Diversity/Privacy/Utility] Suggestion text

Pay particular attention to privacy remediation when k-anonymity < 5 or avg distinct sensitive values < 2.
Offer concrete steps such as:
- generalization / wider binning for quasi-identifiers,
- suppression of rare records,
- increasing noise or training regularization,
- post-processing (micro-aggregation, record suppression),
and discuss trade-offs with fidelity/diversity.
"""

    return prompt



def parse_suggestions(text):
    """Parse AI suggestions into structured format"""
    suggestions = []
    lines = text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line or len(line) < 10:
            continue

        # Try to parse structured suggestions
        priority = 'medium'
        category = 'general'

        if '[PRIORITY:' in line.upper():
            if 'HIGH' in line.upper():
                priority = 'high'
            elif 'LOW' in line.upper():
                priority = 'low'

        if '[CATEGORY:' in line.upper():
            if 'FIDELITY' in line.upper():
                category = 'fidelity'
            elif 'DIVERSITY' in line.upper():
                category = 'diversity'
            elif 'PRIVACY' in line.upper():
                category = 'privacy'
            elif 'UTILITY' in line.upper():
                category = 'utility'

        # Extract main text
        text_content = line
        for tag in ['[PRIORITY:', '[CATEGORY:', 'HIGH]', 'MEDIUM]', 'LOW]',
                    'FIDELITY]', 'DIVERSITY]', 'PRIVACY]', 'UTILITY]']:
            text_content = text_content.replace(tag, '').replace(tag.lower(), '')

        text_content = text_content.strip()

        if text_content:
            suggestions.append({
                'priority': priority,
                'type': category,
                'text': text_content
            })

    return suggestions


@app.route('/api/export/pdf', methods=['POST'])
def export_pdf():
    """
    Esporta il report in formato PDF
    Supporta sia la versione singola che la versione compare con più datasets
    """
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from io import BytesIO
        import base64

        data = request.json
        results = data.get('results', {})
        charts = data.get('charts', {})

        # Detect if this is compare mode (multiple datasets) or single mode
        is_compare_mode = 'analyses' in results and isinstance(results.get('analyses'), list)

        # Create PDF buffer
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#4f46e5'),
            spaceAfter=30,
            alignment=TA_CENTER
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#1e293b'),
            spaceAfter=12,
            spaceBefore=12
        )

        subheading_style = ParagraphStyle(
            'CustomSubheading',
            parent=styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#475569'),
            spaceAfter=10,
            spaceBefore=10
        )

        # Title
        if is_compare_mode:
            story.append(Paragraph("🦸 AURAS - Multi-Dataset Quality Comparison Report", title_style))
            story.append(
                Paragraph(f"Comparing {results.get('augmented_count', len(results['analyses']))} Augmented Datasets",
                          subheading_style))
        else:
            story.append(Paragraph("🦸 AURAS - Data Augmentation Quality Report", title_style))

        story.append(Spacer(1, 0.3 * inch))

        # Process datasets
        datasets = results['analyses'] if is_compare_mode else [results]

        for idx, dataset_results in enumerate(datasets):
            if is_compare_mode:
                story.append(Paragraph(f"📊 Dataset {idx + 1}", heading_style))
                story.append(Spacer(1, 0.2 * inch))

            # Overall Score
            aggregate = dataset_results.get('aggregate_score', {})
            story.append(Paragraph("Overall Assessment", heading_style if not is_compare_mode else subheading_style))

            score_data = [
                ['Metric', 'Score', 'Rating'],
                ['Overall', f"{aggregate.get('overall', 0) * 100:.1f}%", aggregate.get('rating', 'N/A')],
                ['Fidelity', f"{aggregate.get('scores', {}).get('fidelity', 0) * 100:.1f}%", 'See details below'],
                ['Diversity', f"{aggregate.get('scores', {}).get('diversity', 0) * 100:.1f}%", 'See details below'],
                ['Privacy', f"{aggregate.get('scores', {}).get('privacy', 0) * 100:.1f}%", 'See details below']
            ]

            score_table = Table(score_data, colWidths=[2 * inch, 1.5 * inch, 2 * inch])
            score_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4f46e5')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            story.append(score_table)
            story.append(Spacer(1, 0.4 * inch))

            # Fidelity Metrics
            story.append(Paragraph("✅ Fidelity Metrics", subheading_style))
            fidelity = dataset_results.get('fidelity', {})

            fidelity_text = f"""
            <b>KS Test Similarity:</b> {fidelity.get('kolmogorov_smirnov', {}).get('similarity_score', 0):.3f}<br/>
            <b>KL Divergence:</b> {fidelity.get('kl_divergence', {}).get('average_kl_divergence', 0):.3f}<br/>
            <b>JS Divergence:</b> {fidelity.get('js_divergence', {}).get('average_js_divergence', 0):.3f}<br/>
            <b>MMD Score:</b> {fidelity.get('mmd_score', {}).get('mmd_score', 0):.3f}<br/>
            <b>MAD Score:</b> {fidelity.get('mean_absolute_distance', {}).get('overall_distance', 0):.3f}<br/>
            <b>Statistical Moments Score:</b> {fidelity.get('statistical_moments', {}).get('average_moment_difference', 0):.3f}<br/>
            <b>Q-Function Score:</b> {fidelity.get('q_function', {}).get('q_score', 0):.3f}<br/>
            <b>PCA Similarity:</b> {fidelity.get('pca_comparison', {}).get('average_component_similarity', 0):.3f}
            """
            story.append(Paragraph(fidelity_text, styles['Normal']))
            story.append(Spacer(1, 0.3 * inch))

            # Diversity Metrics
            story.append(Paragraph("🌈 Diversity Metrics", subheading_style))
            diversity = dataset_results.get('diversity', {})

            diversity_text = f"""
            <b>Feature Entropy:</b> {diversity.get('feature_entropy', {}).get('average_entropy', 0):.2f}<br/>
            <b>Coverage Ratio:</b> {diversity.get('coverage', {}).get('coverage_ratio', 0) * 100:.1f}%<br/>
            <b>Silhouette Score:</b> {diversity.get('silhouette_score', {}).get('augmented_score', 0):.3f}<br/>
            <b>Davies-Bouldin Index:</b> {diversity.get('davies_bouldin_index', {}).get('augmented_score', 0):.3f}<br/>        
            <b>Cluster Uniformity:</b> {diversity.get('cluster_spread', {}).get('augmented_uniformity', 0) * 100:.1f}%<br/>
            <b>Intra-Cluster Diversity:</b> {diversity.get('intra_diversity', {}).get('diversity_score', 0):.2f}<br/>
            <b>Range Coverage:</b> {diversity.get('range_coverage', {}).get('average_coverage', 0) * 100:.1f}%<br/>        
            <b>ICC:</b> {diversity.get('intra_class_compactness', {}).get('augmented_icc', 0):.3f}
            """
            story.append(Paragraph(diversity_text, styles['Normal']))
            story.append(Spacer(1, 0.3 * inch))

            # Privacy Metrics
            story.append(Paragraph("🔒 Privacy Metrics", subheading_style))
            privacy = dataset_results.get('privacy', {})

            privacy_text = f"""
            <b>Privacy Score:</b> {privacy.get('membership_inference', {}).get('privacy_score', 0):.3f}<br/>
            <b>Uniqueness (avg):</b> {privacy.get('uniqueness', {}).get('average_uniqueness', 0):.3f}<br/>
            <b>Mean DCR:</b> {privacy.get('distance_to_closest', {}).get('mean_dcr', 0):.3f}<br/>
            <b>NN Risk (high ratio):</b> {privacy.get('nearest_neighbor_risk', {}).get('high_risk_ratio', 0):.3f}<br/>
            <b>Attribute Disclosure:</b> {privacy.get('attribute_disclosure', {}).get('average_attribute_risk', 0):.3f}<br/>
            """

            # k-anonymity and l-diversity
            k_val = None
            if isinstance(privacy.get('k_anonymity'), dict):
                k_val = privacy['k_anonymity'].get('k_anonymity', None)
            l_avg = None
            if isinstance(privacy.get('l_diversity'), dict):
                l_avg = privacy['l_diversity'].get('average_distinct_sensitive_values', None)

            if k_val is not None:
                privacy_text += f"<b>k-Anonymity (k):</b> {k_val}<br/>"
            else:
                privacy_text += "<b>k-Anonymity (k):</b> N/A<br/>"

            if l_avg is not None:
                privacy_text += f"<b>l-Diversity (avg distinct):</b> {l_avg}<br/>"
            else:
                privacy_text += "<b>l-Diversity:</b> N/A<br/>"

            story.append(Paragraph(privacy_text, styles['Normal']))
            story.append(Spacer(1, 0.3 * inch))

            # Utility Metrics (CV-aware)
            utility = dataset_results.get('utility', {})
            if utility:
                story.append(Paragraph("🧰 Utility Metrics (Model performance & Cross-Validation)", subheading_style))

                overall_util = utility.get('overall_utility_score', None)
                avg_imp = utility.get('average_improvement', {}).get('accuracy', None)

                if overall_util is not None:
                    story.append(Paragraph(f"<b>Overall Utility Score:</b> {overall_util:.3f}", styles['Normal']))
                if avg_imp is not None:
                    story.append(Paragraph(f"<b>Average improvement (accuracy):</b> {avg_imp:.3f}", styles['Normal']))

                story.append(Spacer(1, 0.2 * inch))

                # Per-model details
                model_results = utility.get('model_results', [])
                for m in model_results:
                    m_title = f"{m.get('model_type', 'model')} — {m.get('parameters', {})}"
                    story.append(Paragraph(m_title, styles.get('Heading4', styles['Heading3'])))

                    # Combined CV info if available
                    comb = m.get('combined', {})
                    comb_cv = comb.get('cv', {})

                    if isinstance(comb_cv, dict) and 'n_folds' in comb_cv:
                        cv_text = f"<b>Combined CV ({comb_cv.get('n_folds')} folds):</b> acc {comb_cv.get('accuracy_mean', 0):.3f} ± {comb_cv.get('accuracy_std', 0):.3f}, f1 {comb_cv.get('f1_mean', 0):.3f} ± {comb_cv.get('f1_std', 0):.3f}"
                        story.append(Paragraph(cv_text, styles['Normal']))

                    # Test set metrics for combined
                    if comb:
                        story.append(Paragraph(
                            f"<b>Combined test accuracy:</b> {comb.get('accuracy', 0):.3f}, <b>f1:</b> {comb.get('f1_weighted', 0):.3f}",
                            styles['Normal']))

                    story.append(Spacer(1, 0.15 * inch))

                story.append(Spacer(1, 0.15 * inch))

            # Add page break between datasets in compare mode
            if is_compare_mode and idx < len(datasets) - 1:
                story.append(PageBreak())
            elif not is_compare_mode:
                story.append(PageBreak())

        # Build PDF
        doc.build(story)
        buffer.seek(0)

        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'auras_report_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        )

    except Exception as e:
        return jsonify({
            'error': 'PDF export failed',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/export/excel', methods=['POST'])
def export_excel():
    """
    Esporta i risultati in formato Excel
    Supporta sia la versione singola che la versione compare con più datasets
    """
    try:
        import xlsxwriter
        from io import BytesIO
        import json
        import pandas as pd
        from flask import send_file, jsonify
        import traceback

        data = request.json
        results = data.get('results', {})

        # Detect if this is compare mode (multiple datasets) or single mode
        is_compare_mode = 'analyses' in results and isinstance(results.get('analyses'), list)
        datasets = results['analyses'] if is_compare_mode else [results]

        # Create Excel buffer
        buffer = BytesIO()
        workbook = xlsxwriter.Workbook(buffer, {'in_memory': True})

        # Add formats
        header_format = workbook.add_format({
            'bold': True,
            'font_size': 14,
            'bg_color': '#4f46e5',
            'font_color': 'white',
            'align': 'center',
            'valign': 'vcenter'
        })

        subheader_format = workbook.add_format({
            'bold': True,
            'font_size': 12,
            'bg_color': '#e0e7ff',
            'align': 'left'
        })

        metric_format = workbook.add_format({
            'align': 'left',
            'valign': 'vcenter'
        })

        value_format = workbook.add_format({
            'align': 'right',
            'valign': 'vcenter',
            'num_format': '0.000'
        })

        integer_format = workbook.add_format({
            'align': 'right',
            'valign': 'vcenter',
            'num_format': '0'
        })

        def write_value(sheet, r, c, value):
            """Helper: choose an appropriate format for values"""
            if isinstance(value, bool):
                sheet.write(r, c, str(value), metric_format)
            elif isinstance(value, int):
                sheet.write(r, c, value, integer_format)
            elif isinstance(value, float):
                sheet.write(r, c, value, value_format)
            elif value is None:
                sheet.write(r, c, 'N/A', metric_format)
            else:
                sheet.write(r, c, str(value), metric_format)

        # --- Summary Sheet (Comparison Overview) ---
        summary_sheet = workbook.add_worksheet('Summary')
        summary_sheet.set_column('A:A', 40)
        summary_sheet.set_column('B:Z', 20)

        row = 0
        if is_compare_mode:
            summary_sheet.merge_range(row, 0, row, len(datasets),
                                      'AURAS - Multi-Dataset Comparison Report', header_format)
        else:
            summary_sheet.merge_range(row, 0, row, 2,
                                      'AURAS - Data Augmentation Quality Report', header_format)
        row += 2

        # Create comparison table
        summary_sheet.write(row, 0, 'Dataset', subheader_format)
        summary_sheet.write(row, 1, 'Overall Score', subheader_format)
        summary_sheet.write(row, 2, 'Rating', subheader_format)
        summary_sheet.write(row, 3, 'Fidelity', subheader_format)
        summary_sheet.write(row, 4, 'Diversity', subheader_format)
        summary_sheet.write(row, 5, 'Privacy', subheader_format)
        summary_sheet.write(row, 6, 'Utility', subheader_format)
        row += 1

        for idx, dataset_results in enumerate(datasets):
            aggregate = dataset_results.get('aggregate_score', {})
            utility = dataset_results.get('utility', {})

            dataset_name = f"Dataset {idx + 1}" if is_compare_mode else "Main Dataset"
            summary_sheet.write(row, 0, dataset_name, metric_format)
            write_value(summary_sheet, row, 1, aggregate.get('overall', 0))
            summary_sheet.write(row, 2, aggregate.get('rating', 'N/A'), metric_format)
            write_value(summary_sheet, row, 3, aggregate.get('scores', {}).get('fidelity', 0))
            write_value(summary_sheet, row, 4, aggregate.get('scores', {}).get('diversity', 0))
            write_value(summary_sheet, row, 5, aggregate.get('scores', {}).get('privacy', 0))
            write_value(summary_sheet, row, 6, utility.get('overall_utility_score', 0) if utility else 0)
            row += 1

        # --- Detailed sheets for each dataset ---
        for idx, dataset_results in enumerate(datasets):
            sheet_suffix = f"_D{idx + 1}" if is_compare_mode else ""

            # --- Fidelity Sheet ---
            fidelity_sheet = workbook.add_worksheet(f'Fidelity{sheet_suffix}')
            fidelity_sheet.set_column('A:A', 50)
            fidelity_sheet.set_column('B:B', 25)

            row = 0
            fidelity_sheet.merge_range(row, 0, row, 1,
                                       f'Fidelity Metrics - Dataset {idx + 1}' if is_compare_mode else 'Fidelity Metrics',
                                       header_format)
            row += 2

            fidelity = dataset_results.get('fidelity', {})

            # Write all fidelity metrics
            ks = fidelity.get('kolmogorov_smirnov', {})
            fidelity_sheet.write(row, 0, 'KS Test - Similarity Score', subheader_format)
            write_value(fidelity_sheet, row, 1, ks.get('similarity_score', 'N/A'))
            row += 1

            kl = fidelity.get('kl_divergence', {})
            fidelity_sheet.write(row, 0, 'KL Divergence - Average KL', subheader_format)
            write_value(fidelity_sheet, row, 1, kl.get('average_kl_divergence', 'N/A'))
            row += 1

            js = fidelity.get('js_divergence', {})
            fidelity_sheet.write(row, 0, 'JS Divergence - Average JS', subheader_format)
            write_value(fidelity_sheet, row, 1, js.get('average_js_divergence', 'N/A'))
            row += 1

            mmd = fidelity.get('mmd_score', {})
            fidelity_sheet.write(row, 0, 'MMD Score', subheader_format)
            write_value(fidelity_sheet, row, 1, mmd.get('mmd_score', 'N/A'))
            row += 1

            mad = fidelity.get('mean_absolute_distance', {})
            fidelity_sheet.write(row, 0, 'MAD - Overall Distance', subheader_format)
            write_value(fidelity_sheet, row, 1, mad.get('overall_distance', 'N/A'))
            row += 1

            sm = fidelity.get('statistical_moments', {})
            fidelity_sheet.write(row, 0, 'Statistical Moments - Avg Moment Difference', subheader_format)
            write_value(fidelity_sheet, row, 1, sm.get('average_moment_difference', 'N/A'))
            row += 1

            qf = fidelity.get('q_function', {})
            fidelity_sheet.write(row, 0, 'Q-Function Score', subheader_format)
            write_value(fidelity_sheet, row, 1, qf.get('q_score', 'N/A'))
            row += 1

            pca = fidelity.get('pca_comparison', {})
            fidelity_sheet.write(row, 0, 'PCA - Avg Component Similarity', subheader_format)
            write_value(fidelity_sheet, row, 1, pca.get('average_component_similarity', 'N/A'))
            row += 2

            # --- Diversity Sheet ---
            diversity_sheet = workbook.add_worksheet(f'Diversity{sheet_suffix}')
            diversity_sheet.set_column('A:A', 50)
            diversity_sheet.set_column('B:B', 25)

            row = 0
            diversity_sheet.merge_range(row, 0, row, 1,
                                        f'Diversity Metrics - Dataset {idx + 1}' if is_compare_mode else 'Diversity Metrics',
                                        header_format)
            row += 2

            diversity = dataset_results.get('diversity', {})

            diversity_sheet.write(row, 0, 'Feature Entropy - Avg', subheader_format)
            write_value(diversity_sheet, row, 1, diversity.get('feature_entropy', {}).get('average_entropy', 'N/A'))
            row += 1

            diversity_sheet.write(row, 0, 'Coverage Ratio', metric_format)
            write_value(diversity_sheet, row, 1, diversity.get('coverage', {}).get('coverage_ratio', 'N/A'))
            row += 1

            diversity_sheet.write(row, 0, 'Silhouette Score (augmented)', metric_format)
            write_value(diversity_sheet, row, 1, diversity.get('silhouette_score', {}).get('augmented_score', 'N/A'))
            row += 1

            diversity_sheet.write(row, 0, 'Davies-Bouldin Index (augmented)', metric_format)
            write_value(diversity_sheet, row, 1,
                        diversity.get('davies_bouldin_index', {}).get('augmented_score', 'N/A'))
            row += 1

            diversity_sheet.write(row, 0, 'Cluster Uniformity', metric_format)
            write_value(diversity_sheet, row, 1, diversity.get('cluster_spread', {}).get('augmented_uniformity', 'N/A'))
            row += 1

            diversity_sheet.write(row, 0, 'Intra-Cluster Diversity', metric_format)
            write_value(diversity_sheet, row, 1, diversity.get('intra_diversity', {}).get('diversity_score', 'N/A'))
            row += 1

            diversity_sheet.write(row, 0, 'Range Coverage - Avg', metric_format)
            write_value(diversity_sheet, row, 1, diversity.get('range_coverage', {}).get('average_coverage', 'N/A'))
            row += 1

            diversity_sheet.write(row, 0, 'ICC (augmented_icc)', metric_format)
            write_value(diversity_sheet, row, 1,
                        diversity.get('intra_class_compactness', {}).get('augmented_icc', 'N/A'))
            row += 2

            # --- Privacy Sheet ---
            privacy_sheet = workbook.add_worksheet(f'Privacy{sheet_suffix}')
            privacy_sheet.set_column('A:A', 50)
            privacy_sheet.set_column('B:B', 30)

            row = 0
            privacy_sheet.merge_range(row, 0, row, 1,
                                      f'Privacy Metrics - Dataset {idx + 1}' if is_compare_mode else 'Privacy Metrics',
                                      header_format)
            row += 2

            privacy = dataset_results.get('privacy', {})

            membership = privacy.get('membership_inference', {})
            privacy_sheet.write(row, 0, 'Membership Inference - Privacy Score', subheader_format)
            write_value(privacy_sheet, row, 1, membership.get('privacy_score', 'N/A'))
            row += 1

            uniqueness = privacy.get('uniqueness', {})
            privacy_sheet.write(row, 0, 'Uniqueness - Average', subheader_format)
            write_value(privacy_sheet, row, 1, uniqueness.get('average_uniqueness', 'N/A'))
            row += 1

            dtc = privacy.get('distance_to_closest', {})
            privacy_sheet.write(row, 0, 'Mean DCR (distance to closest)', subheader_format)
            write_value(privacy_sheet, row, 1, dtc.get('mean_dcr', 'N/A'))
            row += 1

            nnr = privacy.get('nearest_neighbor_risk', {})
            privacy_sheet.write(row, 0, 'NN Risk - High Risk Ratio', subheader_format)
            write_value(privacy_sheet, row, 1, nnr.get('high_risk_ratio', 'N/A'))
            row += 1

            ad = privacy.get('attribute_disclosure', {})
            privacy_sheet.write(row, 0, 'Attribute Disclosure - Avg Attribute Risk', subheader_format)
            write_value(privacy_sheet, row, 1, ad.get('average_attribute_risk', 'N/A'))
            row += 1

            if isinstance(privacy.get('k_anonymity'), dict):
                kval = privacy['k_anonymity'].get('k_anonymity', 'N/A')
                privacy_sheet.write(row, 0, 'k-Anonymity (k)', subheader_format)
                write_value(privacy_sheet, row, 1, kval)
                row += 1
            else:
                privacy_sheet.write(row, 0, 'k-Anonymity (k)', subheader_format)
                write_value(privacy_sheet, row, 1, 'N/A')
                row += 1

            if isinstance(privacy.get('l_diversity'), dict):
                lavg = privacy['l_diversity'].get('average_distinct_sensitive_values', 'N/A')
                privacy_sheet.write(row, 0, 'l-Diversity (avg distinct)', subheader_format)
                write_value(privacy_sheet, row, 1, lavg)
                row += 1
            else:
                privacy_sheet.write(row, 0, 'l-Diversity', subheader_format)
                write_value(privacy_sheet, row, 1, 'N/A')
                row += 1

            # --- Utility Sheet ---
            utility_sheet = workbook.add_worksheet(f'Utility{sheet_suffix}')
            utility_sheet.set_column('A:A', 60)
            utility_sheet.set_column('B:B', 30)

            row = 0
            utility_sheet.merge_range(row, 0, row, 1,
                                      f'Utility Metrics - Dataset {idx + 1}' if is_compare_mode else 'Utility Metrics',
                                      header_format)
            row += 2

            utility = dataset_results.get('utility', {})
            if utility:
                overall_util = utility.get('overall_utility_score', None)
                avg_imp = utility.get('average_improvement', {}).get('accuracy', None)

                utility_sheet.write(row, 0, 'Overall Utility Score', subheader_format)
                write_value(utility_sheet, row, 1, overall_util)
                row += 1

                utility_sheet.write(row, 0, 'Average Improvement (accuracy)', subheader_format)
                write_value(utility_sheet, row, 1, avg_imp)
                row += 2

                model_results = utility.get('model_results', [])
                for m in model_results:
                    utility_sheet.write(row, 0, f"Model: {m.get('model_type', 'N/A')}", subheader_format)
                    row += 1

                    params = m.get('parameters', {})
                    utility_sheet.write(row, 0, 'Parameters (json)', metric_format)
                    utility_sheet.write(row, 1, json.dumps(params, ensure_ascii=False), metric_format)
                    row += 1

                    comb = m.get('combined', {})
                    comb_cv = comb.get('cv', {})

                    if isinstance(comb_cv, dict) and comb_cv:
                        utility_sheet.write(row, 0, 'Combined CV - n_folds', metric_format)
                        write_value(utility_sheet, row, 1, comb_cv.get('n_folds', 'N/A'))
                        row += 1

                        utility_sheet.write(row, 0, 'Combined CV - accuracy mean', metric_format)
                        write_value(utility_sheet, row, 1, comb_cv.get('accuracy_mean', 'N/A'))
                        row += 1

                        utility_sheet.write(row, 0, 'Combined CV - accuracy std', metric_format)
                        write_value(utility_sheet, row, 1, comb_cv.get('accuracy_std', 'N/A'))
                        row += 1

                        utility_sheet.write(row, 0, 'Combined CV - f1 mean', metric_format)
                        write_value(utility_sheet, row, 1, comb_cv.get('f1_mean', 'N/A'))
                        row += 1

                        utility_sheet.write(row, 0, 'Combined CV - f1 std', metric_format)
                        write_value(utility_sheet, row, 1, comb_cv.get('f1_std', 'N/A'))
                        row += 1

                    if comb:
                        utility_sheet.write(row, 0, 'Combined test accuracy', metric_format)
                        write_value(utility_sheet, row, 1, comb.get('accuracy', 'N/A'))
                        row += 1

                        utility_sheet.write(row, 0, 'Combined test f1 (weighted)', metric_format)
                        write_value(utility_sheet, row, 1, comb.get('f1_weighted', 'N/A'))
                        row += 1

                    row += 1
            else:
                utility_sheet.write(row, 0, 'No utility results available', metric_format)
                row += 1

        # Close workbook and send
        workbook.close()
        buffer.seek(0)

        return send_file(
            buffer,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'auras_report_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        )

    except Exception as e:
        return jsonify({
            'error': 'Excel export failed',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/compare')
def index_compare():
    """Serve la pagina principale"""
    return send_from_directory(app.static_folder, 'index2.html')


@app.route('/synthetic')
def synthetic_generator():
    """Serve la pagina del generatore sintetico"""
    return send_from_directory(app.static_folder, 'synthetic_generator.html')


@app.route('/api/generate-synthetic', methods=['POST'])
def generate_synthetic():
    """
    Genera dati sintetici utilizzando multiple tecniche
    """
    try:
        # Verifica presenza del file
        if 'dataset' not in request.files:
            return jsonify({'error': 'File dataset richiesto'}), 400

        dataset_file = request.files['dataset']
        if dataset_file.filename == '':
            return jsonify({'error': 'Nome file non valido'}), 400

        # Carica parametri
        techniques = json.loads(request.form.get('techniques', '[]'))
        num_samples = int(request.form.get('num_samples', 1000))
        random_seed = request.form.get('random_seed')
        technique_params = json.loads(request.form.get('technique_params', '{}'))

        if random_seed:
            random_seed = int(random_seed)
            np.random.seed(random_seed)

        # Salva e carica il dataset
        filename = secure_filename(dataset_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f'original_{filename}')
        dataset_file.save(file_path)

        df_original = load_dataset(file_path)

        # Genera dati sintetici per ogni tecnica
        results = []
        for technique_id in techniques:
            start_time = time.time()

            try:
                # Ottieni parametri specifici per la tecnica
                params = technique_params.get(technique_id, {})

                # Genera dati sintetici
                df_synthetic = generate_with_technique(
                    df_original,
                    technique_id,
                    num_samples,
                    params,
                    random_seed
                )

                generation_time = time.time() - start_time

                # Salva il dataset sintetico - FIX: usa os.path.join correttamente
                synthetic_filename = f'synthetic_{technique_id}_{int(time.time())}.csv'
                synthetic_path = os.path.join(app.config['UPLOAD_FOLDER'], synthetic_filename)

                # Assicurati che la directory esista
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

                df_synthetic.to_csv(synthetic_path, index=False)

                # Verifica che il file esista
                if not os.path.exists(synthetic_path):
                    raise Exception(f"File non salvato correttamente: {synthetic_path}")

                print(f"✅ File salvato: {synthetic_path}")

                # Calcola quality score (opzionale - versione semplificata)
                #quality_score = calculate_simple_quality(df_original, df_synthetic)

                results.append({
                    'technique': technique_id,
                    'file_path': synthetic_path,  # Path completo per il backend
                    'filename': synthetic_filename,
                    'rows': len(df_synthetic),
                    'columns': len(df_synthetic.columns),
                    'generation_time': generation_time
                    #'quality_score': quality_score
                })

            except Exception as e:
                print(f"❌ Error generating with {technique_id}: {str(e)}")
                traceback.print_exc()
                results.append({
                    'technique': technique_id,
                    'error': str(e),
                    'generation_time': time.time() - start_time
                })

        return jsonify({
            'success': True,
            'message': f'Generati {len([r for r in results if "error" not in r])} dataset sintetici',
            'generated_datasets': results
        })

    except Exception as e:
        print(f"❌ Error in generate_synthetic: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': 'Errore durante la generazione',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500


def generate_with_technique(df_original, technique_id, num_samples, params, random_seed):
    """
    Genera dati sintetici con la tecnica specificata
    """

    if technique_id == 'smote':
        return generate_smote(df_original, num_samples, random_seed)

    elif technique_id == 'adasyn':
        return generate_adasyn(df_original, num_samples, random_seed)

    elif technique_id == 'gaussian_copula':
        return generate_gaussian_copula(df_original, num_samples, random_seed)

    elif technique_id == 'ctgan':
        return generate_ctgan(df_original, num_samples, params, random_seed)

    elif technique_id == 'tvae':
        return generate_tvae(df_original, num_samples, params, random_seed)

    elif technique_id == 'smotecdnn':
        return generate_smotecdnn(df_original, num_samples, params, random_seed)

    elif technique_id == 'bayesian_network':
        return generate_bayesian_network(df_original, num_samples, random_seed)

    elif technique_id == 'random_sampling':
        return generate_random_sampling(df_original, num_samples, random_seed)

    elif technique_id == 'noise_injection':
        noise_level = float(params.get('noise_level', 0.1))
        return generate_noise_injection(df_original, num_samples, noise_level, random_seed)

    elif technique_id == 'kde':
        return generate_kde_sampling(df_original, num_samples, random_seed)

    elif technique_id == 'dummy':
        return generate_dummy(df_original, num_samples, random_seed)

    else:
        raise ValueError(f"Tecnica non supportata: {technique_id}")


def generate_smotecdnn(df, num_samples, params, random_seed):
    """
    Genera dati sintetici usando SMOTE-CDNN (SMOTE + Edited Condensed Nearest Neighbors)

    Parametri:
    - df: DataFrame originale
    - num_samples: numero di campioni sintetici da generare
    - params: dizionario con parametri opzionali
        - 'n_neighbors': numero di vicini (default: 5)
        - 'kind_sel': tipo di selezione per CDNN ('cd' o 'all', default: 'cd')
        - 'smote_strategy': strategia per SMOTE (default: 'auto')
        - 'cdnn_strategy': strategia per CDNN (default: 'all')
    - random_seed: seed per riproducibilità
    """
    try:
        from backend.SMOTECDNN.SMOTECDNN import SMOTECDNN, EditedCDNN
        from sklearn.preprocessing import OneHotEncoder
        import numpy as np
        import pandas as pd

        if num_samples <= 0:
            return pd.DataFrame(columns=df.columns)

        np.random.seed(random_seed)
        df = df.copy().reset_index(drop=True)

        # Estrai parametri
        n_neighbors = int(params.get('n_neighbors', 5))
        kind_sel = params.get('kind_sel', 'cd')
        smote_strategy = params.get('smote_strategy', 'auto')
        cdnn_strategy = params.get('cdnn_strategy', 'all')

        # Split features / target (target = ultima colonna)
        feature_cols = list(df.columns[:-1])
        target_col = df.columns[-1]
        X_df = df[feature_cols]
        y = df[target_col].values

        # Identifica colonne categoriche e numeriche
        categorical_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
        numeric_cols = [c for c in feature_cols if c not in categorical_cols]

        # Encoding delle colonne categoriche
        if len(categorical_cols) > 0:
            ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
            cat_values = X_df[categorical_cols].astype(object).values
            ohe.fit(cat_values)
            cat_encoded = ohe.transform(cat_values)
        else:
            ohe = None
            cat_encoded = np.zeros((len(X_df), 0))

        # Matrice numerica
        if len(numeric_cols) > 0:
            X_numeric = X_df[numeric_cols].astype(float).values
        else:
            X_numeric = np.zeros((len(X_df), 0))

        # Matrice finale per SMOTE-CDNN
        X_encoded = np.hstack([X_numeric, cat_encoded]).astype(float)

        # Verifica se ci sono abbastanza campioni per applicare SMOTE-CDNN
        unique_labels, counts = np.unique(y, return_counts=True)
        min_samples = min(counts)

        if min_samples < 2:
            print("SMOTE-CDNN: non abbastanza campioni, fallback a SMOTE standard")
            return generate_smote(df, num_samples, random_seed)

        # Aggiusta n_neighbors se necessario
        k_neighbors = min(n_neighbors, min_samples - 1)
        if k_neighbors < 1:
            k_neighbors = 1

        # Calcola sampling strategy per ottenere circa num_samples
        # Dato che CDNN ridurrà i campioni, oversampling inizialmente
        target_samples = {}
        for label, count in zip(unique_labels, counts):
            # Aggiungi campioni proporzionalmente alla classe inversa
            ratio = num_samples / len(df)
            target_samples[label] = int(count * (1 + ratio))

        # Crea e configura SMOTE-CDNN
        smote_cdnn = SMOTECDNN(
            sampling_strategy=smote_strategy,
            random_state=random_seed,
            smote=SMOTE(
                sampling_strategy=target_samples,
                k_neighbors=k_neighbors,
                random_state=random_seed
            ),
            cdnn=EditedCDNN(
                sampling_strategy=cdnn_strategy,
                n_neighbors=k_neighbors,
                kind_sel=kind_sel
            )
        )

        # Applica SMOTE-CDNN
        X_resampled, y_resampled = smote_cdnn.fit_resample(X_encoded, y)

        # Rimuovi i campioni originali per ottenere solo i sintetici
        def row_key(arr):
            return tuple(np.round(arr, 8).tolist())

        orig_keys = set(row_key(r) for r in X_encoded)
        synthetic_samples = []

        for xi, yi in zip(X_resampled, y_resampled):
            if row_key(xi) not in orig_keys:
                synthetic_samples.append(np.concatenate([xi, [yi]], axis=None))

        # Se abbiamo generato più campioni del necessario, tronca
        if len(synthetic_samples) > num_samples:
            indices = np.random.choice(len(synthetic_samples), num_samples, replace=False)
            synthetic_samples = [synthetic_samples[i] for i in indices]

        # Se abbiamo generato meno campioni, completa con duplicati + rumore
        elif len(synthetic_samples) < num_samples:
            missing = num_samples - len(synthetic_samples)
            if len(synthetic_samples) > 0:
                for _ in range(missing):
                    idx = np.random.randint(0, len(synthetic_samples))
                    sample = synthetic_samples[idx].copy()
                    # Aggiungi piccolo rumore
                    noise = np.random.normal(scale=1e-6, size=len(sample) - 1)
                    sample[:-1] += noise
                    synthetic_samples.append(sample)
            else:
                # Se CDNN ha rimosso tutto, fallback a SMOTE
                print("SMOTE-CDNN: CDNN ha rimosso tutti i campioni, fallback a SMOTE")
                return generate_smote(df, num_samples, random_seed)

        # Ricostruisci DataFrame con colonne originali
        df_synthetic = _reconstruct_df_from_encoded(
            synthetic_samples,
            numeric_cols,
            categorical_cols,
            ohe,
            df,
            target_col
        )

        return df_synthetic[df.columns]

    except Exception as e:
        print(f"SMOTE-CDNN error: {e}, falling back to SMOTE")
        import traceback
        traceback.print_exc()
        return generate_smote(df, num_samples, random_seed)

def generate_smote(df, num_samples, random_seed):
    """
    Usa SMOTE (imblearn) e fa automaticamente One-Hot per le categorical.
    Restituisce SOLO i campioni sintetici con le stesse colonne di `df`.

    Parametri:
      - df: pd.DataFrame (assume target = ultima colonna)
      - num_samples: int, numero di campioni sintetici da generare
      - random_seed: int, seed per riproducibilità
    """
    if num_samples <= 0:
        return pd.DataFrame(columns=df.columns)

    rng = np.random.RandomState(random_seed)
    df = df.copy().reset_index(drop=True)

    # split features / target (target = ultima colonna)
    feature_cols = list(df.columns[:-1])
    target_col = df.columns[-1]

    X_df = df[feature_cols]
    y = df[target_col].values

    # identifico categorical cols come non-numeric
    categorical_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    # preparo encoder se ci sono categorical
    if len(categorical_cols) > 0:
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        cat_values = X_df[categorical_cols].astype(object).values  # ensure object for encoder
        ohe.fit(cat_values)
        cat_encoded = ohe.transform(cat_values)  # numpy array
    else:
        ohe = None
        cat_encoded = np.zeros((len(X_df), 0))

    # numeric matrix
    if len(numeric_cols) > 0:
        X_numeric = X_df[numeric_cols].astype(float).values
    else:
        X_numeric = np.zeros((len(X_df), 0))

    # matrice finale per SMOTE
    X_encoded = np.hstack([X_numeric, cat_encoded]).astype(float)

    # se c'è una sola classe target -> duplico con piccolo rumore
    unique_labels, counts = np.unique(y, return_counts=True)
    if len(unique_labels) == 1:
        synth_rows = []
        for _ in range(num_samples):
            idx = rng.randint(0, X_encoded.shape[0])
            noise = rng.normal(scale=1e-6, size=X_encoded.shape[1])
            x_new = X_encoded[idx] + noise
            synth_rows.append(np.concatenate([x_new, [y[idx]]], axis=None))
        df_synth = _reconstruct_df_from_encoded(synth_rows, numeric_cols, categorical_cols, ohe, df, target_col)
        return df_synth[df.columns]

    # calcolo quanti sintetici per classe (inverso proporzionale alla frequenza)
    inv = 1.0 / counts
    prop = inv / inv.sum()
    raw = prop * num_samples
    synth_per_class = np.floor(raw).astype(int)
    remainder = int(num_samples - synth_per_class.sum())
    if remainder > 0:
        fracts = raw - np.floor(raw)
        idx_order = np.argsort(-fracts)
        for i in range(remainder):
            synth_per_class[idx_order[i]] += 1

    # costruisco sampling_strategy per SMOTE = numero totale che vogliamo per ogni classe
    sampling_strategy = {}
    manual_synth_encoded = []  # per le classi con un solo esempio (SMOTE non funziona)
    classes_for_smote = []
    for label, current_count, add_count in zip(unique_labels, counts, synth_per_class):
        if add_count <= 0:
            continue
        if current_count == 1:
            # duplicazione manuale con piccolo rumore (in spazio encoded)
            idx = np.where(y == label)[0][0]
            for _ in range(add_count):
                noise = rng.normal(scale=1e-6, size=X_encoded.shape[1])
                x_new = X_encoded[idx] + noise
                manual_synth_encoded.append(np.concatenate([x_new, [label]], axis=None))
        else:
            sampling_strategy[label] = int(current_count + add_count)
            classes_for_smote.append(label)

    synth_rows_encoded = []

    if len(sampling_strategy) > 0:
        # k_neighbors dipende dal minimo count tra le classi considerate
        min_count_included = min([np.sum(y == lab) for lab in classes_for_smote])
        k_neighbors = min(5, min_count_included - 1)
        if k_neighbors < 1:
            # fallback: duplicazione manuale se non possibile usare SMOTE
            for label, current_count, add_count in zip(unique_labels, counts, synth_per_class):
                if add_count > 0:
                    idxs = np.where(y == label)[0]
                    for _ in range(add_count):
                        idx = rng.choice(idxs)
                        noise = rng.normal(scale=1e-6, size=X_encoded.shape[1])
                        x_new = X_encoded[idx] + noise
                        synth_rows_encoded.append(np.concatenate([x_new, [label]], axis=None))
        else:
            sm = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=random_seed)
            X_res, y_res = sm.fit_resample(X_encoded, y)

            # rimuovo le occorrenze originali usando un counter (arrotondando per stabilità float)
            def row_key(arr):
                # arrotondo per evitare problemi di float equality
                return tuple(np.round(arr, 8).tolist())

            orig_counter = Counter(row_key(r) for r in X_encoded.tolist())

            for xi, yi in zip(X_res, y_res):
                key = row_key(xi)
                if orig_counter.get(key, 0) > 0:
                    orig_counter[key] -= 1
                else:
                    synth_rows_encoded.append(np.concatenate([xi, [yi]], axis=None))

    # unisco i sintetici manuali e quelli generati da SMOTE
    all_encoded = []
    if len(manual_synth_encoded) > 0:
        all_encoded.extend(manual_synth_encoded)
    if len(synth_rows_encoded) > 0:
        all_encoded.extend(synth_rows_encoded)

    # se per qualche motivo non abbiamo generato nulla -> ritorno DataFrame vuoto con colonne originali
    if len(all_encoded) == 0:
        return pd.DataFrame(columns=df.columns)

    # se abbiamo generato più di num_samples tronchiamo (manteniamo ordine)
    if len(all_encoded) > num_samples:
        all_encoded = all_encoded[:num_samples]

    # se meno, completiamo duplicando alcuni sintetici con piccolo rumore
    if len(all_encoded) < num_samples:
        missing = num_samples - len(all_encoded)
        synth_X = np.array([row[:-1] for row in all_encoded])
        synth_y = np.array([row[-1] for row in all_encoded])
        for i in range(missing):
            idx = rng.randint(0, synth_X.shape[0])
            noise = rng.normal(scale=1e-6, size=synth_X.shape[1])
            x_new = synth_X[idx] + noise
            all_encoded.append(np.concatenate([x_new, [synth_y[idx]]], axis=None))

    # ricostruisco DataFrame con colonne originali
    df_synthetic = _reconstruct_df_from_encoded(all_encoded, numeric_cols, categorical_cols, ohe, df, target_col)

    # garantisco ordine colonne e reset index
    return df_synthetic[df.columns]


def _reconstruct_df_from_encoded(encoded_rows, numeric_cols, categorical_cols, ohe, original_df, target_col):
    """
    Aiuta a ricostruire il DataFrame originario partendo dalle righe encoded (lista di array [X_encoded..., label]).
    numeric_cols, categorical_cols sono le colonne originali.
    original_df serve per prendere i dtypes originali.
    """
    if len(encoded_rows) == 0:
        return pd.DataFrame(columns=list(original_df.columns))

    arr = np.vstack(encoded_rows)
    X_enc = arr[:, :-1]
    y_vals = arr[:, -1]

    # split numeric / cat nell'encoded (sappiamo la dimensione delle parti)
    n_numeric = len(numeric_cols)
    if ohe is not None:
        n_cat_encoded = sum(len(cat) for cat in ohe.categories_)
    else:
        n_cat_encoded = 0

    # ricavo numeric part e cat part
    if n_numeric > 0:
        X_num = X_enc[:, :n_numeric]
    else:
        X_num = np.zeros((X_enc.shape[0], 0))

    if n_cat_encoded > 0:
        X_cat_enc = X_enc[:, n_numeric:]
        # inverse transform delle categorical
        # OneHotEncoder.inverse_transform expects input in same shape used in transform
        # ATTENZIONE: inverse_transform richiede valori molto simili a 0/1; dato che SMOTE produce valori continui,
        # usiamo argmax per ciascun gruppo di categorie per ricavare la categoria più probabile.
        cat_cols_values = []
        start = 0
        for cats in ohe.categories_:
            k = len(cats)
            block = X_cat_enc[:, start:start + k]
            # argmax sul blocco -> indice categoria
            idxs = np.argmax(block, axis=1)
            chosen = np.array([cats[i] for i in idxs])
            cat_cols_values.append(chosen.reshape(-1, 1))
            start += k
        if len(cat_cols_values) > 0:
            X_cat_recon = np.hstack(cat_cols_values)
        else:
            X_cat_recon = np.zeros((X_enc.shape[0], 0))
    else:
        X_cat_recon = np.zeros((X_enc.shape[0], 0))

    # costruisco DataFrame ordinato secondo numeric_cols + categorical_cols
    parts = []
    cols = []
    if n_numeric > 0:
        df_num = pd.DataFrame(X_num, columns=numeric_cols)
        parts.append(df_num)
        cols.extend(numeric_cols)
    if len(categorical_cols) > 0:
        df_cat = pd.DataFrame(X_cat_recon, columns=categorical_cols).astype(object)
        parts.append(df_cat)
        cols.extend(categorical_cols)

    if len(parts) > 0:
        X_recon_df = pd.concat(parts, axis=1)
    else:
        X_recon_df = pd.DataFrame(index=range(X_enc.shape[0]))

    # cast numeric columns al dtype originale se possibile
    for c in numeric_cols:
        try:
            X_recon_df[c] = pd.to_numeric(X_recon_df[c], errors='coerce')
            # opzionale: mantenere dtype originale (float/int)
            orig_dtype = original_df[c].dtype
            X_recon_df[c] = X_recon_df[c].astype(orig_dtype, errors='ignore')
        except Exception:
            pass

    # cast categorical col dtypes come originali (se original erano categorical)
    for c in categorical_cols:
        try:
            orig_dtype = original_df[c].dtype
            # se originale era categorical, assegno categoria
            if pd.api.types.is_categorical_dtype(orig_dtype):
                X_recon_df[c] = X_recon_df[c].astype('category')
            else:
                X_recon_df[c] = X_recon_df[c].astype(original_df[c].dtype, errors='ignore')
        except Exception:
            pass

    # target
    df_out = X_recon_df.copy()
    df_out[target_col] = y_vals

    # mantenere tipi target come nell'originale se possibile
    try:
        df_out[target_col] = df_out[target_col].astype(original_df[target_col].dtype)
    except Exception:
        pass

    # assicurare ordine colonne come original_df
    final_cols = list(original_df.columns)
    for c in final_cols:
        if c not in df_out.columns:
            # colonna mancante (es. se non c'erano feature) -> aggiungo NaN
            df_out[c] = pd.NA
    df_out = df_out[final_cols]

    return df_out.reset_index(drop=True)


def generate_adasyn(df, num_samples, random_seed):
    """Genera dati con ADASYN"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        return generate_random_sampling(df, num_samples, random_seed)

    X = df[numeric_cols].values
    y = np.zeros(len(df))
    minority_size = min(len(df) // 10, 5)
    y[:minority_size] = 1

    try:
        adasyn = ADASYN(random_state=random_seed, n_neighbors=min(5, len(df) - 1))
        X_resampled, _ = adasyn.fit_resample(X, y)

        synthetic_samples = X_resampled[len(X):]

        if len(synthetic_samples) < num_samples:
            indices = np.random.choice(len(synthetic_samples), num_samples, replace=True)
            synthetic_samples = synthetic_samples[indices]
        else:
            synthetic_samples = synthetic_samples[:num_samples]

        df_synthetic = pd.DataFrame(synthetic_samples, columns=numeric_cols)

        for col in df.columns:
            if col not in numeric_cols:
                df_synthetic[col] = np.random.choice(df[col].values, num_samples)

        return df_synthetic[df.columns]

    except Exception as e:
        print(f"ADASYN error: {e}, falling back to random sampling")
        return generate_random_sampling(df, num_samples, random_seed)


def generate_gaussian_copula(df, num_samples, random_seed):
    """Genera dati con Gaussian Copula"""
    np.random.seed(random_seed)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == 0:
        return generate_random_sampling(df, num_samples, random_seed)

    # Trasforma i dati in distribuzioni uniformi
    X = df[numeric_cols].values

    # Calcola ranghi e normalizza
    uniform_data = np.zeros_like(X)
    for i in range(X.shape[1]):
        ranks = stats.rankdata(X[:, i])
        uniform_data[:, i] = ranks / (len(ranks) + 1)

    # Trasforma in normale standard
    normal_data = stats.norm.ppf(uniform_data)

    # Calcola correlazione e genera nuovi dati
    cov_matrix = np.cov(normal_data.T)
    synthetic_normal = np.random.multivariate_normal(
        np.zeros(len(numeric_cols)),
        cov_matrix,
        num_samples
    )

    # Ritrasforma in uniforme e poi nella distribuzione originale
    synthetic_uniform = stats.norm.cdf(synthetic_normal)

    synthetic_data = np.zeros_like(synthetic_uniform)
    for i in range(len(numeric_cols)):
        # Usa la distribuzione empirica
        sorted_vals = np.sort(X[:, i])
        synthetic_data[:, i] = np.percentile(sorted_vals, synthetic_uniform[:, i] * 100)

    df_synthetic = pd.DataFrame(synthetic_data, columns=numeric_cols)

    # Aggiungi colonne categoriche
    for col in df.columns:
        if col not in numeric_cols:
            df_synthetic[col] = np.random.choice(df[col].values, num_samples)

    return df_synthetic[df.columns]


def generate_ctgan(df, num_samples, params, random_seed):
    """Genera dati con CTGAN (richiede sdv library)"""
    try:
        from sdv.single_table import CTGANSynthesizer
        from sdv.metadata import SingleTableMetadata

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)

        synthesizer = CTGANSynthesizer(
            metadata,
            epochs=int(params.get('epochs', 300)),
            batch_size=int(params.get('batch_size', 500)),
            verbose=False
        )

        synthesizer.fit(df)
        synthetic_data = synthesizer.sample(num_samples)

        return synthetic_data

    except ImportError:
        print("SDV library not installed, falling back to Gaussian Copula")
        return generate_gaussian_copula(df, num_samples, random_seed)
    except Exception as e:
        print(f"CTGAN error: {e}, falling back to Gaussian Copula")
        return generate_gaussian_copula(df, num_samples, random_seed)


def generate_tvae(df, num_samples, params, random_seed):
    """Genera dati con TVAE (richiede sdv library)"""
    try:
        from sdv.single_table import TVAESynthesizer
        from sdv.metadata import SingleTableMetadata

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)

        synthesizer = TVAESynthesizer(
            metadata,
            epochs=int(params.get('epochs', 300)),
            verbose=False
        )

        synthesizer.fit(df)
        synthetic_data = synthesizer.sample(num_samples)

        return synthetic_data

    except ImportError:
        print("SDV library not installed, falling back to Gaussian Copula")
        return generate_gaussian_copula(df, num_samples, random_seed)
    except Exception as e:
        print(f"TVAE error: {e}, falling back to Gaussian Copula")
        return generate_gaussian_copula(df, num_samples, random_seed)


def generate_bayesian_network(df, num_samples, random_seed):
    """Genera dati con Bayesian Network (versione semplificata)"""
    np.random.seed(random_seed)

    # Implementazione semplificata: usa dipendenze lineari
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == 0:
        return generate_random_sampling(df, num_samples, random_seed)

    # Calcola medie e deviazioni standard
    means = df[numeric_cols].mean()
    stds = df[numeric_cols].std()

    # Calcola matrice di correlazione
    corr_matrix = df[numeric_cols].corr().values

    # Genera dati correlati
    synthetic_data = np.random.multivariate_normal(
        means.values,
        np.outer(stds.values, stds.values) * corr_matrix,
        num_samples
    )

    df_synthetic = pd.DataFrame(synthetic_data, columns=numeric_cols)

    # Aggiungi colonne categoriche
    for col in df.columns:
        if col not in numeric_cols:
            # Mantieni la distribuzione originale
            value_counts = df[col].value_counts(normalize=True)
            df_synthetic[col] = np.random.choice(
                value_counts.index,
                num_samples,
                p=value_counts.values
            )

    return df_synthetic[df.columns]


def generate_random_sampling(df, num_samples, random_seed):
    """Genera dati con campionamento casuale con sostituzione"""
    np.random.seed(random_seed)
    return df.sample(n=num_samples, replace=True, random_state=random_seed).reset_index(drop=True)


def generate_noise_injection(df, num_samples, noise_level, random_seed):
    """Genera dati aggiungendo rumore gaussiano"""
    np.random.seed(random_seed)

    # Campiona dal dataset originale
    sampled_df = df.sample(n=num_samples, replace=True, random_state=random_seed).reset_index(drop=True)

    # Aggiungi rumore alle colonne numeriche
    numeric_cols = sampled_df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        std = sampled_df[col].std()
        noise = np.random.normal(0, std * noise_level, num_samples)
        sampled_df[col] = sampled_df[col] + noise

    return sampled_df


def generate_kde_sampling(df, num_samples, random_seed):
    """Genera dati usando Kernel Density Estimation"""
    np.random.seed(random_seed)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == 0:
        return generate_random_sampling(df, num_samples, random_seed)

    # Normalizza i dati
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[numeric_cols])

    # Fit KDE
    kde = KernelDensity(bandwidth='scott', kernel='gaussian')
    kde.fit(X_scaled)

    # Genera campioni
    synthetic_scaled = kde.sample(num_samples, random_state=random_seed)
    synthetic_data = scaler.inverse_transform(synthetic_scaled)

    df_synthetic = pd.DataFrame(synthetic_data, columns=numeric_cols)

    # Aggiungi colonne categoriche
    for col in df.columns:
        if col not in numeric_cols:
            df_synthetic[col] = np.random.choice(df[col].values, num_samples)

    return df_synthetic[df.columns]


def generate_dummy(df, num_samples, random_seed):
    """Genera dati completamente casuali basati sulle distribuzioni delle colonne"""
    np.random.seed(random_seed)

    df_synthetic = pd.DataFrame()

    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            # Per numeriche: usa min, max e distribuzione uniforme
            min_val = df[col].min()
            max_val = df[col].max()

            if df[col].dtype == np.int64:
                df_synthetic[col] = np.random.randint(min_val, max_val + 1, num_samples)
            else:
                df_synthetic[col] = np.random.uniform(min_val, max_val, num_samples)
        else:
            # Per categoriche: campiona casualmente
            df_synthetic[col] = np.random.choice(df[col].unique(), num_samples)

    return df_synthetic


def calculate_simple_quality(df_original, df_synthetic):
    """Calcola un quality score semplificato"""
    try:
        numeric_cols = df_original.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return 0.5

        # Confronta statistiche descrittive
        scores = []

        for col in numeric_cols:
            # Confronta medie
            mean_diff = abs(df_original[col].mean() - df_synthetic[col].mean()) / (df_original[col].std() + 1e-10)
            mean_score = max(0, 1 - mean_diff)

            # Confronta deviazioni standard
            std_diff = abs(df_original[col].std() - df_synthetic[col].std()) / (df_original[col].std() + 1e-10)
            std_score = max(0, 1 - std_diff)

            scores.append((mean_score + std_score) / 2)

        return np.mean(scores) if scores else 0.5

    except Exception as e:
        print(f"Error calculating quality: {e}")
        return 0.5


@app.route('/api/download-synthetic', methods=['GET'])
def download_synthetic():
    """Download del dataset sintetico generato"""
    try:
        file_param = request.args.get('file')
        technique = request.args.get('technique', 'synthetic')

        print(f"📥 Download request - Raw file param: {file_param}, Technique: {technique}")

        if not file_param:
            return jsonify({'error': 'Parametro file mancante'}), 400

        upload_folder = app.config.get('UPLOAD_FOLDER', '/tmp/heroes_uploads')

        # Normalizza separatori e rimuove spazi strani
        normalized = file_param.replace('\\', '/').strip()

        # Se ci viene passato solo il filename (es. "synthetic_...csv") -> costruisci il path corretto
        candidate_paths = []

        # 1) Se normalized è un path assoluto (o relativo) che esiste, usalo così com'è (dopo normpath)
        p_norm = os.path.normpath(normalized)
        candidate_paths.append(p_norm)

        # 2) Prova come basename dentro upload_folder
        basename = os.path.basename(normalized)
        candidate_paths.append(os.path.join(upload_folder, basename))

        # 3) Se l'input era una concatenazione senza slash (es: "/tmp/heroes_uploadssynthetic_..."),
        #    prova a inserire una slash tra upload_folder e la parte rimanente
        if upload_folder in normalized and upload_folder + os.sep not in normalized:
            # estrai la parte dopo upload_folder
            suffix = normalized.split(upload_folder)[-1].lstrip('/\\')
            candidate_paths.append(os.path.join(upload_folder, suffix))

        # Normalizza tutti e controlla esistenza
        found = None
        for cp in candidate_paths:
            cp_norm = os.path.normpath(cp)
            print(f"🔎 Checking candidate path: {cp_norm}")
            if os.path.exists(cp_norm):
                found = cp_norm
                break

        if not found:
            # debug listing della cartella
            if os.path.exists(upload_folder):
                files = os.listdir(upload_folder)
                print(f"📂 File disponibili in {upload_folder}: {files}")
            return jsonify({
                'error': 'File non trovato',
                'requested_param': file_param,
                'tried_candidates': candidate_paths
            }), 404

        print(f"✅ Invio file: {found}")

        return send_file(
            found,
            as_attachment=True,
            download_name=f'synthetic_{technique}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )

    except Exception as e:
        print(f"❌ Error in download_synthetic: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': 'Errore durante il download',
            'details': str(e)
        }), 500


if __name__ == '__main__':
    print("""
    ╔═══════════════════════════════════════╗
    ║         🦸 AURAS API Server           ║
    ║                                       ║
    ╚═══════════════════════════════════════╝
    """)

    # Usa queste impostazioni per Docker
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

    app.run(
        host='0.0.0.0',
        port=5088,
        debug=debug_mode,
        use_reloader=False
    )