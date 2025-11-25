-- Initialize additional databases if needed
CREATE DATABASE IF NOT EXISTS heroes_analytics;

-- Create additional user roles if needed
CREATE USER IF NOT EXISTS analytics_user WITH PASSWORD 'analytics_password';
GRANT ALL PRIVILEGES ON DATABASE heroes_analytics TO analytics_user;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";