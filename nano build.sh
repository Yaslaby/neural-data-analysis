#!/bin/bash

echo "======================================"
echo "Building Neural Data Analyzer for macOS"
echo "======================================"

# Clean old builds
echo "Cleaning old builds..."
rm -rf build dist

# Build with PyInstaller
echo "Running PyInstaller..."
pyinstaller neural_analyzer.spec

# Check if build succeeded
if [ -d "dist/NeuralDataAnalyzer.app" ]; then
    echo ""
    echo "✅ BUILD SUCCESSFUL!"
    echo "======================================"
    echo "Your app is ready at:"
    echo "dist/NeuralDataAnalyzer.app"
    echo "======================================"
    echo ""
    echo "To test it, run:"
    echo "open dist/NeuralDataAnalyzer.app"
else
    echo ""
    echo "❌ BUILD FAILED!"
    echo "Check the error messages above."
fi