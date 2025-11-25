#!/bin/bash

echo "🦸 Starting Heroes Application Stack..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Create necessary directories
mkdir -p init/ollama init/postgres init/mysql

# Make init scripts executable
chmod +x init/ollama/download-models.sh

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📋 Copying environment file..."
    cp env.example.sh .env
    echo "⚠️  Please review and configure the .env file before starting the application."
fi

# Build and start services
echo "🚀 Building and starting services..."
docker-compose up -d --build

echo "⏳ Waiting for services to start..."
sleep 30

# Check service status
echo "🔍 Checking service status..."
for service in heroes postgres mysql redis ollama; do
    if docker ps | grep -q $service; then
        echo "✅ $service is running"
    else
        echo "❌ $service failed to start"
    fi
done

echo ""
echo "🎉 Heroes Application Stack is starting!"
echo ""
echo "📊 Application URL: http://localhost:5088"
echo "🗄️  PostgreSQL: localhost:5432"
echo "🐬 MySQL: localhost:3306"
echo "🔴 Redis: localhost:6379"
echo "🤖 Ollama: localhost:11434"
echo ""
echo "📝 Check logs with: docker-compose logs -f"
echo "🛑 Stop with: docker-compose down"