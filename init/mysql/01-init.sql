-- Initialize additional databases if needed
CREATE DATABASE IF NOT EXISTS heroes_analytics;

-- Create additional user
CREATE USER IF NOT EXISTS 'analytics_user'@'%' IDENTIFIED BY 'analytics_password';
GRANT ALL PRIVILEGES ON heroes_analytics.* TO 'analytics_user'@'%';

FLUSH PRIVILEGES;