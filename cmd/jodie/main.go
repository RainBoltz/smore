package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/cnclabs/smore/internal/models/jodie"
)

func main() {
	// Command-line flags
	train := flag.String("train", "", "Path to training interaction file (format: user item timestamp [features...])")
	output := flag.String("output", "", "Path to output embeddings file")
	dim := flag.Int("dim", 128, "Embedding dimension")
	epochs := flag.Int("epochs", 10, "Number of training epochs")
	learningRate := flag.Float64("lr", 0.01, "Learning rate")
	batchSize := flag.Int("batch-size", 128, "Batch size for training")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "JODIE - Joint Dynamic User-Item Embeddings\n\n")
		fmt.Fprintf(os.Stderr, "Usage:\n")
		fmt.Fprintf(os.Stderr, "  %s [options]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Description:\n")
		fmt.Fprintf(os.Stderr, "  JODIE learns dynamic embeddings for users and items that evolve over time.\n")
		fmt.Fprintf(os.Stderr, "  Key features:\n")
		fmt.Fprintf(os.Stderr, "  - RNN-based embedding updates after each interaction\n")
		fmt.Fprintf(os.Stderr, "  - Projects embeddings to predict future states\n")
		fmt.Fprintf(os.Stderr, "  - Handles bipartite user-item interaction networks\n")
		fmt.Fprintf(os.Stderr, "  - 4.4x better than CTDNE for interaction prediction\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nInput Format:\n")
		fmt.Fprintf(os.Stderr, "  Interaction list: user_id item_id timestamp [feature1 feature2 ...]\n")
		fmt.Fprintf(os.Stderr, "  - user_id and item_id can be any string identifiers\n")
		fmt.Fprintf(os.Stderr, "  - Timestamp can be Unix time or any numeric value\n")
		fmt.Fprintf(os.Stderr, "  - Optional features describe interaction properties\n\n")
		fmt.Fprintf(os.Stderr, "  Example:\n")
		fmt.Fprintf(os.Stderr, "    user123 item456 1609459200\n")
		fmt.Fprintf(os.Stderr, "    user123 item789 1609462800 1.0 0.5\n")
		fmt.Fprintf(os.Stderr, "    user456 item123 1609466400 0.8\n\n")
		fmt.Fprintf(os.Stderr, "Output Format:\n")
		fmt.Fprintf(os.Stderr, "  The output file contains:\n")
		fmt.Fprintf(os.Stderr, "  - Header: num_users num_items dimension\n")
		fmt.Fprintf(os.Stderr, "  - User embeddings: U user_name embedding_vector\n")
		fmt.Fprintf(os.Stderr, "  - Item embeddings: I item_name embedding_vector\n\n")
		fmt.Fprintf(os.Stderr, "Examples:\n")
		fmt.Fprintf(os.Stderr, "  # Train on Reddit user-post interactions\n")
		fmt.Fprintf(os.Stderr, "  %s -train reddit.txt -output jodie.emb \\\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "    -dim 128 -epochs 10 -lr 0.01\n\n")
		fmt.Fprintf(os.Stderr, "  # Train on e-commerce purchases\n")
		fmt.Fprintf(os.Stderr, "  %s -train purchases.txt -output jodie.emb \\\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "    -dim 256 -epochs 20 -lr 0.005\n\n")
		fmt.Fprintf(os.Stderr, "  # Train on movie ratings with features\n")
		fmt.Fprintf(os.Stderr, "  %s -train ratings.txt -output jodie.emb \\\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "    -dim 128 -epochs 15 -lr 0.01\n\n")
		fmt.Fprintf(os.Stderr, "Key Features:\n")
		fmt.Fprintf(os.Stderr, "  ✓ Dynamic embeddings that evolve over time\n")
		fmt.Fprintf(os.Stderr, "  ✓ RNN-based temporal dynamics\n")
		fmt.Fprintf(os.Stderr, "  ✓ Future interaction prediction\n")
		fmt.Fprintf(os.Stderr, "  ✓ Handles optional interaction features\n")
		fmt.Fprintf(os.Stderr, "  ✓ State-of-the-art for temporal recommendation\n\n")
		fmt.Fprintf(os.Stderr, "Use Cases:\n")
		fmt.Fprintf(os.Stderr, "  - Temporal recommendation systems\n")
		fmt.Fprintf(os.Stderr, "  - Social network interaction prediction\n")
		fmt.Fprintf(os.Stderr, "  - User behavior forecasting\n")
		fmt.Fprintf(os.Stderr, "  - Item popularity prediction\n")
		fmt.Fprintf(os.Stderr, "  - Churn prediction\n\n")
		fmt.Fprintf(os.Stderr, "References:\n")
		fmt.Fprintf(os.Stderr, "  Kumar et al. \"Predicting Dynamic Embedding Trajectory in\n")
		fmt.Fprintf(os.Stderr, "  Temporal Interaction Networks\", KDD 2019\n")
		fmt.Fprintf(os.Stderr, "  https://arxiv.org/abs/1908.01207\n\n")
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

	// Validate parameters
	if *dim <= 0 {
		fmt.Fprintf(os.Stderr, "Error: -dim must be positive\n")
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

	if *batchSize <= 0 {
		fmt.Fprintf(os.Stderr, "Error: -batch-size must be positive\n")
		os.Exit(1)
	}

	// Print configuration
	fmt.Println("===================================================")
	fmt.Println("JODIE - Joint Dynamic User-Item Embeddings")
	fmt.Println("===================================================")
	fmt.Println()

	// Create JODIE model
	model := jodie.New()

	// Load interaction graph
	fmt.Println("Loading Interaction Graph:")
	fmt.Printf("\tInput: %s\n", *train)
	fmt.Println()

	if err := model.LoadInteractions(*train); err != nil {
		fmt.Fprintf(os.Stderr, "Error loading interactions: %v\n", err)
		os.Exit(1)
	}

	// Initialize model
	fmt.Println()
	model.Init(*dim)

	// Train model
	fmt.Println()
	model.Train(*epochs, *learningRate, *batchSize)

	// Evaluate predictions
	model.EvaluatePredictions()

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
