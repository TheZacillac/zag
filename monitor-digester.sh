#!/bin/bash

# Monitor script for the digester service
# Checks if digester is running and restarts it if needed

echo "🔍 Checking digester service status..."

# Check if digester container is running
if docker compose ps digester | grep -q "Up"; then
    echo "✅ Digester is running"
    
    # Check if it's actually processing (not stuck)
    if docker compose logs digester --tail 5 | grep -q "Scanning"; then
        echo "✅ Digester is actively scanning"
    else
        echo "⚠️ Digester might be stuck, checking logs..."
        docker compose logs digester --tail 10
    fi
else
    echo "❌ Digester is not running, starting it..."
    docker compose up -d digester
    sleep 5
    echo "✅ Digester started"
fi

# Show current status
echo ""
echo "📊 Current digester status:"
docker compose ps digester
echo ""
echo "📋 Recent logs:"
docker compose logs digester --tail 5
