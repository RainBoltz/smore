package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/cnclabs/smore/internal/models/sne"
)

func main() {
	// Define command-line flags
	train := flag.String("train", "", "Train on signed network")
	save := flag.String("save", "", "Save embeddings")
	dimensions := flag.Int("dimensions", 64, "Dimension of embeddings (will be doubled: source + target)")
	undirected := flag.Bool("undirected", true, "Whether edges are undirected")
	negativeSamples := flag.Int("negative_samples", 5, "Number of negative samples")
	beta := flag.Float64("beta", 1.0, "Weight for negative edges")
	epochs := flag.Int("epochs", 50, "Number of training epochs")
	samplesPerEdge := flag.Int("samples_per_edge", 5, "Samples per edge per epoch")
	alpha := flag.Float64("alpha", 0.025, "Learning rate")
	threads := flag.Int("threads", 4, "Number of training threads")

	flag.Usage = func() {
		fmt.Println("[SMORe-Go]")
		fmt.Println("\tGolang implementation of SMORe - SNE")
		fmt.Println()
		fmt.Println("SNE (Signed Network Embedding) - For Positive AND Negative Edges:")
		fmt.Println("\t✓ Handles both positive and negative relationships")
		fmt.Println("\t✓ Based on balance theory")
		fmt.Println("\t✓ Two embedding vectors per node")
		fmt.Println("\t✓ Unique capability in graph embeddings")
		fmt.Println()
		fmt.Println("How it works:")
		fmt.Println("\t1. Positive edges → maximize similarity")
		fmt.Println("\t2. Negative edges → minimize similarity")
		fmt.Println("\t3. Balance theory:")
		fmt.Println("\t   • Friend of friend = Friend")
		fmt.Println("\t   • Enemy of enemy = Friend")
		fmt.Println("\t   • Friend of enemy = Enemy")
		fmt.Println()
		fmt.Println("Input format (signed edges):")
		fmt.Println("\tfrom to sign [weight]")
		fmt.Println("\tSign can be: +1, 1, pos, positive OR -1, neg, negative")
		fmt.Println()
		fmt.Println("Examples:")
		fmt.Println("\tAlice Bob +1 1.0    # Alice trusts Bob")
		fmt.Println("\tAlice Eve -1 1.0    # Alice distrusts Eve")
		fmt.Println("\tBob Charlie pos 1.0 # Bob likes Charlie")
		fmt.Println("\tEve Dan negative 1.0 # Eve dislikes Dan")
		fmt.Println()
		fmt.Println("Key parameters:")
		fmt.Println("\t- dimensions: Base embedding dimension (default: 64)")
		fmt.Println("\t  • Output will be 2x (source + target vectors)")
		fmt.Println("\t  • Typical range: 32-128")
		fmt.Println("\t- beta: Weight for negative edges (default: 1.0)")
		fmt.Println("\t  • Higher = stronger enforcement of negative relationships")
		fmt.Println("\t  • Typical range: 0.5-2.0")
		fmt.Println("\t- negative_samples: Negative samples per edge (default: 5)")
		fmt.Println("\t- epochs: Training epochs (default: 50)")
		fmt.Println("\t- samples_per_edge: Samples per edge per epoch (default: 5)")
		fmt.Println()
		fmt.Println("Options Description:")
		flag.PrintDefaults()
		fmt.Println()
		fmt.Println("Usage:")
		fmt.Println("./sne -train signed_net.txt -save embeddings.txt \\")
		fmt.Println("      -dimensions 64 -beta 1.0 -epochs 50 \\")
		fmt.Println("      -negative_samples 5 -samples_per_edge 5 \\")
		fmt.Println("      -alpha 0.025 -threads 4")
		fmt.Println()
		fmt.Println("Examples:")
		fmt.Println("\t# Basic training")
		fmt.Println("\t./sne -train signed_net.txt -save embeddings.txt")
		fmt.Println()
		fmt.Println("\t# Emphasize negative edges")
		fmt.Println("\t./sne -train signed_net.txt -save embeddings.txt -beta 2.0")
		fmt.Println()
		fmt.Println("\t# High-quality training")
		fmt.Println("\t./sne -train signed_net.txt -save embeddings.txt \\")
		fmt.Println("\t     -dimensions 128 -epochs 100 -samples_per_edge 10")
		fmt.Println()
		fmt.Println("Use Cases:")
		fmt.Println("\t✓ Social networks (friend/enemy relationships)")
		fmt.Println("\t✓ Trust networks (trust/distrust)")
		fmt.Println("\t✓ Review systems (upvote/downvote)")
		fmt.Println("\t✓ Political networks (alliance/opposition)")
		fmt.Println("\t✓ Signed collaboration networks")
		fmt.Println()
		fmt.Println("Datasets:")
		fmt.Println("\t• Slashdot (technology news)")
		fmt.Println("\t• Epinions (product reviews)")
		fmt.Println("\t• Wikipedia (editor interactions)")
		fmt.Println("\t• Bitcoin trust networks")
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
	if *negativeSamples <= 0 {
		fmt.Println("Error: negative_samples must be positive")
		os.Exit(1)
	}
	if *beta <= 0 {
		fmt.Println("Error: beta must be positive")
		os.Exit(1)
	}
	if *epochs <= 0 {
		fmt.Println("Error: epochs must be positive")
		os.Exit(1)
	}
	if *samplesPerEdge <= 0 {
		fmt.Println("Error: samples_per_edge must be positive")
		os.Exit(1)
	}
	if *alpha <= 0 {
		fmt.Println("Error: alpha must be positive")
		os.Exit(1)
	}
	if *threads <= 0 {
		fmt.Println("Error: threads must be positive")
		os.Exit(1)
	}

	fmt.Println("═══════════════════════════════════════════════════════")
	fmt.Println("  SNE - Signed Network Embedding")
	fmt.Println("═══════════════════════════════════════════════════════")
	fmt.Println()

	startTime := time.Now()

	// Create and train model
	model := sne.New()

	fmt.Println("Loading signed network...")
	if err := model.LoadEdgeList(*train, *undirected); err != nil {
		fmt.Printf("Error loading network: %v\n", err)
		os.Exit(1)
	}

	loadTime := time.Since(startTime)
	fmt.Printf("Network loaded in %.2f seconds\n", loadTime.Seconds())
	fmt.Println()

	model.Init(*dimensions, *negativeSamples, *beta)

	trainStartTime := time.Now()
	model.Train(*epochs, *samplesPerEdge, *alpha, *threads)
	trainTime := time.Since(trainStartTime)

	// Compute balance ratio
	balanceRatio := model.ComputeBalanceRatio()

	// Save embeddings
	fmt.Println()
	if err := model.SaveEmbeddings(*save); err != nil {
		fmt.Printf("Error saving embeddings: %v\n", err)
		os.Exit(1)
	}

	totalTime := time.Since(startTime)
	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════")
	fmt.Println("  Statistics")
	fmt.Println("═══════════════════════════════════════════════════════")
	fmt.Printf("Balance ratio: %.2f%%\n", balanceRatio*100)
	fmt.Println("\t(% of edges with correctly predicted signs)")
	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════")
	fmt.Println("  Timing Summary")
	fmt.Println("═══════════════════════════════════════════════════════")
	fmt.Printf("Loading time:     %.2f seconds\n", loadTime.Seconds())
	fmt.Printf("Training time:    %.2f seconds\n", trainTime.Seconds())
	fmt.Printf("Total time:       %.2f seconds\n", totalTime.Seconds())
	fmt.Println()
	fmt.Println("✓ SNE training complete!")
	fmt.Println()
	fmt.Println("Next steps:")
	fmt.Println("\t• Use embeddings for sign prediction")
	fmt.Println("\t• Evaluate balance theory preservation")
	fmt.Println("\t• Apply to link sign prediction tasks")
	fmt.Println("\t• Analyze community structure in signed networks")
}
