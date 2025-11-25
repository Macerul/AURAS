"""
Heroes Database Module
Handles SQLite/PostgreSQL storage and external DB connections
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
from datetime import datetime
import json

Base = declarative_base()


class AnalysisResult(Base):
    """Store analysis results"""
    __tablename__ = 'analysis_results'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    dataset_name = Column(String(255))
    original_dataset_rows = Column(Integer)
    augmented_dataset_rows = Column(Integer)
    fidelity_score = Column(Float)
    diversity_score = Column(Float)
    privacy_score = Column(Float)
    utility_score = Column(Float, nullable=True)
    overall_score = Column(Float)
    rating = Column(String(50))
    metrics_json = Column(Text)  # Full metrics as JSON
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'dataset_name': self.dataset_name,
            'original_rows': self.original_dataset_rows,
            'augmented_rows': self.augmented_dataset_rows,
            'fidelity_score': self.fidelity_score,
            'diversity_score': self.diversity_score,
            'privacy_score': self.privacy_score,
            'utility_score': self.utility_score,
            'overall_score': self.overall_score,
            'rating': self.rating,
            'metrics': json.loads(self.metrics_json) if self.metrics_json else {}
        }


class ImprovementSuggestion(Base):
    """Store AI improvement suggestions"""
    __tablename__ = 'improvement_suggestions'
    
    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    suggestion_type = Column(String(100))  # 'fidelity', 'diversity', 'privacy', 'utility', 'general'
    suggestion_text = Column(Text)
    priority = Column(String(50))  # 'high', 'medium', 'low'
    
    def to_dict(self):
        return {
            'id': self.id,
            'analysis_id': self.analysis_id,
            'timestamp': self.timestamp.isoformat(),
            'type': self.suggestion_type,
            'text': self.suggestion_text,
            'priority': self.priority
        }


class DatabaseManager:
    """Manage database connections and operations"""
    
    def __init__(self, db_url='sqlite:///auras.db'):
        """
        Initialize database connection
        
        Args:
            db_url: SQLAlchemy connection string
                Examples:
                - SQLite: 'sqlite:///auras.db'
                - PostgreSQL: 'postgresql://user:pass@localhost/heroes'
                - MySQL: 'mysql+pymysql://user:pass@localhost/heroes'
        """
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def save_analysis(self, dataset_name, original_rows, augmented_rows, 
                     fidelity_score, diversity_score, privacy_score, 
                     overall_score, rating, metrics_dict, utility_score=None):
        """Save analysis results to database"""
        result = AnalysisResult(
            dataset_name=dataset_name,
            original_dataset_rows=original_rows,
            augmented_dataset_rows=augmented_rows,
            fidelity_score=fidelity_score,
            diversity_score=diversity_score,
            privacy_score=privacy_score,
            utility_score=utility_score,
            overall_score=overall_score,
            rating=rating,
            metrics_json=json.dumps(metrics_dict)
        )
        self.session.add(result)
        self.session.commit()
        return result.id
    
    def save_suggestion(self, analysis_id, suggestion_type, text, priority='medium'):
        """Save improvement suggestion"""
        suggestion = ImprovementSuggestion(
            analysis_id=analysis_id,
            suggestion_type=suggestion_type,
            suggestion_text=text,
            priority=priority
        )
        self.session.add(suggestion)
        self.session.commit()
        return suggestion.id
    
    def get_analysis_history(self, limit=10):
        """Get recent analysis results"""
        results = self.session.query(AnalysisResult)\
            .order_by(AnalysisResult.timestamp.desc())\
            .limit(limit)\
            .all()
        return [r.to_dict() for r in results]
    
    def get_analysis_by_id(self, analysis_id):
        """Get specific analysis result"""
        result = self.session.query(AnalysisResult)\
            .filter(AnalysisResult.id == analysis_id)\
            .first()
        return result.to_dict() if result else None
    
    def get_suggestions(self, analysis_id=None, limit=10):
        """Get improvement suggestions"""
        query = self.session.query(ImprovementSuggestion)
        
        if analysis_id:
            query = query.filter(ImprovementSuggestion.analysis_id == analysis_id)
        
        suggestions = query.order_by(ImprovementSuggestion.timestamp.desc())\
            .limit(limit)\
            .all()
        
        return [s.to_dict() for s in suggestions]
    
    def close(self):
        """Close database connection"""
        self.session.close()


def load_from_external_db(db_config):
    """
    Load dataset from external database
    
    Args:
        db_config: dict with keys:
            - db_type: 'mysql' or 'postgresql'
            - host: hostname
            - port: port number
            - database: database name
            - user: username
            - password: password
            - query: SQL query to execute
    
    Returns:
        pandas.DataFrame
    """
    db_type = db_config.get('db_type')
    host = db_config.get('host')
    port = db_config.get('port')
    database = db_config.get('database')
    user = db_config.get('user')
    password = db_config.get('password')
    query = db_config.get('query')
    
    # Build connection string
    if db_type == 'mysql':
        conn_str = f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}'
    elif db_type == 'postgresql':
        conn_str = f'postgresql://{user}:{password}@{host}:{port}/{database}'
    else:
        raise ValueError(f"Unsupported database type: {db_type}")
    
    # Connect and load data
    engine = create_engine(conn_str)
    df = pd.read_sql(query, engine)
    engine.dispose()
    
    return df


def test_db_connection(db_config):
    """
    Test database connection
    
    Returns:
        dict with 'success' boolean and 'message' string
    """
    try:
        db_type = db_config.get('db_type')
        host = db_config.get('host')
        port = db_config.get('port')
        database = db_config.get('database')
        user = db_config.get('user')
        password = db_config.get('password')
        
        if db_type == 'mysql':
            conn_str = f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}'
        elif db_type == 'postgresql':
            conn_str = f'postgresql://{user}:{password}@{host}:{port}/{database}'
        else:
            return {'success': False, 'message': f'Unsupported database type: {db_type}'}
        
        engine = create_engine(conn_str)
        conn = engine.connect()
        conn.close()
        engine.dispose()
        
        return {'success': True, 'message': 'Connection successful'}
    
    except Exception as e:
        return {'success': False, 'message': str(e)}
