package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/cnclabs/smore/internal/models/han"
)

func main() {
	// Command-line flags
	train := flag.String("train", "", "Path to training edge list file (format: source src_type target tgt_type edge_type [weight])")
	output := flag.String("output", "", "Path to output embeddings file")
	metaPaths := flag.String("meta-paths", "", "Comma-separated meta-paths (e.g., \"User Item User,User Item Category Item User\")")
	dim := flag.Int("dim", 128, "Embedding dimension")
	walkTimes := flag.Int("walk-times", 10, "Number of random walks per node per meta-path")
	walkSteps := flag.Int("walk-steps", 5, "Steps in each random walk")
	epochs := flag.Int("epochs", 10, "Number of training epochs")
	learningRate := flag.Float64("lr", 0.01, "Learning rate for attention updates")
	workers := flag.Int("workers", 4, "Number of parallel workers")
	undirected := flag.Bool("undirected", false, "Treat graph as undirected")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "HAN - Heterogeneous Attention Network\n\n")
		fmt.Fprintf(os.Stderr, "Usage:\n")
		fmt.Fprintf(os.Stderr, "  %s [options]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Description:\n")
		fmt.Fprintf(os.Stderr, "  HAN learns embeddings for heterogeneous graphs using hierarchical attention:\n")
		fmt.Fprintf(os.Stderr, "  - Node-level attention: learns importance of neighbors\n")
		fmt.Fprintf(os.Stderr, "  - Semantic-level attention: learns importance of meta-paths\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nInput Format:\n")
		fmt.Fprintf(os.Stderr, "  Edge list: source_node source_type target_node target_type edge_type [weight]\n")
		fmt.Fprintf(os.Stderr, "  Example: Alice User Bob User friend 1.0\n")
		fmt.Fprintf(os.Stderr, "  Example: Alice User iPhone Product purchased 1.0\n\n")
		fmt.Fprintf(os.Stderr, "Meta-path Format:\n")
		fmt.Fprintf(os.Stderr, "  Space-separated node types forming a path\n")
		fmt.Fprintf(os.Stderr, "  Example: \"User Item User\" (users who bought similar items)\n")
		fmt.Fprintf(os.Stderr, "  Example: \"User Item Category Item User\" (users via item categories)\n\n")
		fmt.Fprintf(os.Stderr, "Example:\n")
		fmt.Fprintf(os.Stderr, "  # Train on heterogeneous e-commerce graph\n")
		fmt.Fprintf(os.Stderr, "  %s -train ecommerce.txt -output han.emb \\\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "    -meta-paths \"User Item User,User Item Category Item User\" \\\n")
		fmt.Fprintf(os.Stderr, "    -dim 128 -walk-times 10 -epochs 10 -lr 0.01\n\n")
		fmt.Fprintf(os.Stderr, "  # Train on academic citation network\n")
		fmt.Fprintf(os.Stderr, "  %s -train academic.txt -output han.emb \\\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "    -meta-paths \"Paper Author Paper,Paper Venue Paper\" \\\n")
		fmt.Fprintf(os.Stderr, "    -dim 128 -epochs 20\n\n")
		fmt.Fprintf(os.Stderr, "Key Features:\n")
		fmt.Fprintf(os.Stderr, "  ✓ Hierarchical attention mechanisms\n")
		fmt.Fprintf(os.Stderr, "  ✓ Automatic meta-path importance learning\n")
		fmt.Fprintf(os.Stderr, "  ✓ Node-level neighbor weighting\n")
		fmt.Fprintf(os.Stderr, "  ✓ State-of-the-art for heterogeneous graphs\n\n")
		fmt.Fprintf(os.Stderr, "References:\n")
		fmt.Fprintf(os.Stderr, "  Wang et al. \"Heterogeneous Graph Attention Network\", WWW 2019\n")
		fmt.Fprintf(os.Stderr, "  https://arxiv.org/abs/1903.07293\n\n")
	}

	flag.Parse()

	// Validate required arguments
	if *train == "" {
		fmt.Fprintf(os.Stderr, "Error: -train is required\n\n")
		flag.Usage()
		os.Exit(1)
	}

	if *output == "" {
		fmt.Fprintf(os.Stderr, "Error: -output is required\n\n")
		flag.Usage()
		os.Exit(1)
	}

	if *metaPaths == "" {
		fmt.Fprintf(os.Stderr, "Error: -meta-paths is required\n\n")
		flag.Usage()
		os.Exit(1)
	}

	// Validate parameters
	if *dim <= 0 {
		fmt.Fprintf(os.Stderr, "Error: -dim must be positive\n")
		os.Exit(1)
	}

	if *walkTimes <= 0 {
		fmt.Fprintf(os.Stderr, "Error: -walk-times must be positive\n")
		os.Exit(1)
	}

	if *walkSteps <= 0 {
		fmt.Fprintf(os.Stderr, "Error: -walk-steps must be positive\n")
		os.Exit(1)
	}

	if *epochs <= 0 {
		fmt.Fprintf(os.Stderr, "Error: -epochs must be positive\n")
		os.Exit(1)
	}

	if *learningRate <= 0 {
		fmt.Fprintf(os.Stderr, "Error: -lr must be positive\n")
		os.Exit(1)
	}

	if *workers <= 0 {
		fmt.Fprintf(os.Stderr, "Error: -workers must be positive\n")
		os.Exit(1)
	}

	// Print configuration
	fmt.Println("===================================================")
	fmt.Println("HAN - Heterogeneous Attention Network")
	fmt.Println("===================================================")
	fmt.Println()

	// Create HAN model
	model := han.New()

	// Load heterogeneous graph
	fmt.Println("Loading Heterogeneous Graph:")
	fmt.Printf("\tInput: %s\n", *train)
	fmt.Printf("\tUndirected: %v\n", *undirected)
	fmt.Println()

	if err := model.LoadEdgeList(*train, *undirected); err != nil {
		fmt.Fprintf(os.Stderr, "Error loading graph: %v\n", err)
		os.Exit(1)
	}

	// Add meta-paths
	fmt.Println()
	fmt.Println("Adding Meta-paths:")
	pathList := strings.Split(*metaPaths, ",")
	for _, path := range pathList {
		path = strings.TrimSpace(path)
		if path != "" {
			if err := model.AddMetaPath(path); err != nil {
				fmt.Fprintf(os.Stderr, "Error adding meta-path '%s': %v\n", path, err)
				os.Exit(1)
			}
		}
	}

	// Initialize model
	fmt.Println()
	model.Init(*dim, *learningRate)

	// Train model
	fmt.Println()
	model.Train(*walkTimes, *walkSteps, *epochs, *workers)

	// Compute attention statistics
	model.ComputeAttentionStats()

	// Save embeddings
	fmt.Println()
	if err := model.SaveEmbeddings(*output); err != nil {
		fmt.Fprintf(os.Stderr, "Error saving embeddings: %v\n", err)
		os.Exit(1)
	}

	fmt.Println()
	fmt.Println("===================================================")
	fmt.Println("Training Complete!")
	fmt.Println("===================================================")
}
