# Heroes - Environment Variables Configuration
# Copy this file to .env and adjust values as needed

# Flask Configuration
FLASK_ENV=production
FLASK_DEBUG=False
FLASK_APP=backend/app.py

# Server Configuration
HOST=0.0.0.0
PORT=5088

# File Upload Configuration
MAX_FILE_SIZE_MB=100
UPLOAD_FOLDER=/tmp/heroes_uploads
ALLOWED_EXTENSIONS=csv,parquet,pq

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/heroes/app.log

# CORS Configuration
CORS_ORIGINS=*

# Analysis Configuration
MAX_SAMPLES_FOR_ANALYSIS=10000
ENABLE_ASYNC_JOBS=True

# Database Configuration
DATABASE_URL=postgresql://heroes_user:heroes_password@postgres:5432/heroes_db
MYSQL_URL=mysql://heroes_user:heroes_password@mysql:3306/heroes_db
REDIS_URL=redis://redis:6379/0

# Ollama Configuration
OLLAMA_HOST=ollama
OLLAMA_PORT=11434
DEFAULT_OLLAMA_MODEL=llama3.2:3b

# Security
SECRET_KEY=change-this-to-a-random-secret-key-in-production

# Features Flags
ENABLE_EXPORT=True
ENABLE_ADVANCED_METRICS=True
ENABLE_VISUALIZATION=True
ENABLE_SYNTHETIC_GENERATION=True
ENABLE_MULTI_DATASET_COMPARISON=True

# External Database Connections
EXTERNAL_DB_TIMEOUT=30
MAX_DB_CONNECTIONS=10