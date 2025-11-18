#!/bin/bash
# FastRP Demo Script
# This demonstrates FastRP - the ultra-fast graph embedding method

echo "═══════════════════════════════════════════════════════"
echo "  FastRP Demo - Fast Random Projection"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "FastRP is 75,000x faster than Node2Vec!"
echo "Perfect for large-scale graphs and real-time applications."
echo ""

# Create a sample network (Zachary's Karate Club style)
echo "Creating sample network..."
cat > demo_network.txt << EOF
1 2 1.0
1 3 1.0
1 4 1.0
2 3 1.0
2 4 1.0
3 4 1.0
4 5 1.0
5 6 1.0
5 7 1.0
6 7 1.0
7 8 1.0
8 9 1.0
8 10 1.0
9 10 1.0
EOF

echo "Network created: demo_network.txt"
echo ""

# Run FastRP with different configurations
echo "─────────────────────────────────────────────────────"
echo "Example 1: Quick run with defaults (fastest)"
echo "─────────────────────────────────────────────────────"
../bin/fastrp -train demo_network.txt -save demo_output_default.txt -dimensions 64 -iterations 3

echo ""
echo ""
echo "─────────────────────────────────────────────────────"
echo "Example 2: High quality with more context"
echo "─────────────────────────────────────────────────────"
../bin/fastrp -train demo_network.txt -save demo_output_highquality.txt -dimensions 128 -iterations 5 -normalization 0.5

echo ""
echo ""
echo "─────────────────────────────────────────────────────"
echo "Example 3: Fast run with minimal iterations"
echo "─────────────────────────────────────────────────────"
../bin/fastrp -train demo_network.txt -save demo_output_fast.txt -dimensions 64 -iterations 2

echo ""
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Demo Complete!"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Output files generated:"
echo "  • demo_output_default.txt (64-dim, 3 iterations)"
echo "  • demo_output_highquality.txt (128-dim, 5 iterations, normalized)"
echo "  • demo_output_fast.txt (64-dim, 2 iterations)"
echo ""
echo "Key takeaways:"
echo "  ✓ No training required - instant embeddings!"
echo "  ✓ More iterations = larger neighborhood context"
echo "  ✓ Normalization helps with varying node degrees"
echo "  ✓ Perfect for large graphs where speed matters"
echo ""

# Cleanup option
read -p "Clean up generated files? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm demo_network.txt demo_output_*.txt
    echo "✓ Cleaned up demo files"
fi
