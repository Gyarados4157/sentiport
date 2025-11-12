#!/bin/bash

echo "=========================================="
echo "AlphaQuest Demo - Starting Streamlit App"
echo "=========================================="
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  Virtual environment not found. Using system Python."
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "⚠️  Streamlit is not installed."
    echo "Installing required packages..."
    pip install streamlit plotly pandas numpy
fi

echo ""
echo "🚀 Launching AlphaQuest Demo..."
echo ""
echo "📊 The demo will open in your browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the streamlit app
streamlit run demo_streamlit.py
