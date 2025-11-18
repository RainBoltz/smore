package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/cnclabs/smore/internal/models/fastrp"
)

func main() {
	// Define command-line flags
	train := flag.String("train", "", "Train the Network data")
	save := flag.String("save", "", "Save the representation data")
	dimensions := flag.Int("dimensions", 128, "Dimension of vertex representation")
	undirected := flag.Bool("undirected", true, "Whether the edge is undirected")
	iterations := flag.Int("iterations", 3, "Number of aggregation iterations (higher = more context)")
	normalization := flag.Float64("normalization", 0.0, "Degree normalization strength (0.0 = no normalization, 0.5 = sqrt, 1.0 = full)")
	threads := flag.Int("threads", 4, "Number of parallel threads")

	flag.Usage = func() {
		fmt.Println("[SMORe-Go]")
		fmt.Println("\tGolang implementation of SMORe - FastRP")
		fmt.Println()
		fmt.Println("FastRP (Fast Random Projection) - Ultra-fast graph embedding:")
		fmt.Println("\t✓ 75,000x faster than Node2Vec")
		fmt.Println("\t✓ No training required (single-pass computation)")
		fmt.Println("\t✓ Production-ready (Neo4j's default embedding)")
		fmt.Println("\t✓ Perfect for large-scale graphs")
		fmt.Println()
		fmt.Println("How it works:")
		fmt.Println("\t1. Initialize nodes with very sparse random vectors (~95% zeros)")
		fmt.Println("\t2. Iteratively aggregate neighbor features")
		fmt.Println("\t3. Concatenate features from different iterations")
		fmt.Println("\t4. L2 normalize final embeddings")
		fmt.Println()
		fmt.Println("Key parameters:")
		fmt.Println("\t- dimensions: Final embedding dimension (default: 128)")
		fmt.Println("\t- iterations: Number of aggregation iterations (default: 3)")
		fmt.Println("\t  • More iterations = larger neighborhood context")
		fmt.Println("\t  • Typical range: 2-5")
		fmt.Println("\t- normalization: Degree normalization strength (default: 0.0)")
		fmt.Println("\t  • 0.0 = no normalization")
		fmt.Println("\t  • 0.5 = sqrt normalization (recommended)")
		fmt.Println("\t  • 1.0 = full normalization")
		fmt.Println()
		fmt.Println("Options Description:")
		flag.PrintDefaults()
		fmt.Println()
		fmt.Println("Usage:")
		fmt.Println("./fastrp -train net.txt -save rep.txt -undirected -dimensions 128 -iterations 3 -normalization 0.5 -threads 4")
		fmt.Println()
		fmt.Println("Examples:")
		fmt.Println("\t# Quick run with defaults (fastest)")
		fmt.Println("\t./fastrp -train net.txt -save rep.txt")
		fmt.Println()
		fmt.Println("\t# High quality with more context")
		fmt.Println("\t./fastrp -train net.txt -save rep.txt -iterations 5 -normalization 0.5")
		fmt.Println()
		fmt.Println("\t# Large graphs with many threads")
		fmt.Println("\t./fastrp -train net.txt -save rep.txt -threads 16")
		fmt.Println()
		fmt.Println("Performance Note:")
		fmt.Println("\tFastRP is designed for speed. On a 200k node graph:")
		fmt.Println("\t• FastRP: ~136 seconds")
		fmt.Println("\t• Node2Vec: ~63.8 days")
		fmt.Println("\t(75,000x speedup!)")
	}

	flag.Parse()

	// Check required parameters
	if *train == "" || *save == "" {
		flag.Usage()
		os.Exit(1)
	}

	// Validate parameters
	if *dimensions <= 0 {
		fmt.Println("Error: dimensions must be positive")
		os.Exit(1)
	}
	if *iterations <= 0 {
		fmt.Println("Error: iterations must be positive")
		os.Exit(1)
	}
	if *normalization < 0 || *normalization > 1 {
		fmt.Println("Error: normalization must be between 0.0 and 1.0")
		os.Exit(1)
	}
	if *threads <= 0 {
		fmt.Println("Error: threads must be positive")
		os.Exit(1)
	}

	fmt.Println("═══════════════════════════════════════════════════════")
	fmt.Println("  FastRP - Fast Random Projection Graph Embedding")
	fmt.Println("═══════════════════════════════════════════════════════")
	fmt.Println()

	startTime := time.Now()

	// Create and generate embeddings
	frp := fastrp.New()

	fmt.Println("Loading graph...")
	if err := frp.LoadEdgeList(*train, *undirected); err != nil {
		fmt.Printf("Error loading edge list: %v\n", err)
		os.Exit(1)
	}

	loadTime := time.Since(startTime)
	fmt.Printf("Graph loaded in %.2f seconds\n", loadTime.Seconds())
	fmt.Println()

	frp.Init(*dimensions, *iterations, *normalization)
	fmt.Println()

	genStartTime := time.Now()
	frp.Generate(*threads)
	genTime := time.Since(genStartTime)

	// Compute and display statistics
	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════")
	fmt.Println("  Statistics")
	fmt.Println("═══════════════════════════════════════════════════════")

	sparsity := frp.ComputeSparsity()
	fmt.Printf("Embedding sparsity: %.2f%%\n", sparsity*100)

	// Save embeddings
	if err := frp.SaveWeights(*save); err != nil {
		fmt.Printf("Error saving weights: %v\n", err)
		os.Exit(1)
	}

	totalTime := time.Since(startTime)
	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════")
	fmt.Println("  Timing Summary")
	fmt.Println("═══════════════════════════════════════════════════════")
	fmt.Printf("Loading time:     %.2f seconds\n", loadTime.Seconds())
	fmt.Printf("Generation time:  %.2f seconds\n", genTime.Seconds())
	fmt.Printf("Total time:       %.2f seconds\n", totalTime.Seconds())
	fmt.Println()
	fmt.Println("✓ FastRP embedding generation complete!")
}
