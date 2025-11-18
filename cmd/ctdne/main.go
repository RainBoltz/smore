package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/cnclabs/smore/internal/models/ctdne"
)

func main() {
	// Command-line flags
	train := flag.String("train", "", "Path to training temporal edge list file (format: source target timestamp)")
	output := flag.String("output", "", "Path to output embeddings file")
	dim := flag.Int("dim", 128, "Embedding dimension")
	walkTimes := flag.Int("walk-times", 10, "Number of random walks per node")
	walkSteps := flag.Int("walk-steps", 80, "Steps in each random walk")
	windowSize := flag.Int("window-size", 10, "Context window size for skip-gram")
	negativeSamples := flag.Int("negative-samples", 5, "Number of negative samples")
	alpha := flag.Float64("alpha", 0.025, "Initial learning rate")
	timeWindow := flag.Float64("time-window", 0, "Time window for temporal walks (0 = auto: 10% of time span)")
	workers := flag.Int("workers", 4, "Number of parallel workers")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "CTDNE - Continuous-Time Dynamic Network Embeddings\n\n")
		fmt.Fprintf(os.Stderr, "Usage:\n")
		fmt.Fprintf(os.Stderr, "  %s [options]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Description:\n")
		fmt.Fprintf(os.Stderr, "  CTDNE learns embeddings for time-evolving graphs using temporal random walks.\n")
		fmt.Fprintf(os.Stderr, "  Unlike standard random walks, temporal walks respect edge timestamps:\n")
		fmt.Fprintf(os.Stderr, "  - Each step must occur after the previous step\n")
		fmt.Fprintf(os.Stderr, "  - Captures temporal dynamics and network evolution\n")
		fmt.Fprintf(os.Stderr, "  - Ideal for social networks, communication networks, transaction graphs\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nInput Format:\n")
		fmt.Fprintf(os.Stderr, "  Temporal edge list: source_node target_node timestamp\n")
		fmt.Fprintf(os.Stderr, "  - Timestamp can be Unix time, day number, or any numeric value\n")
		fmt.Fprintf(os.Stderr, "  - Edges don't need to be sorted (will be sorted internally)\n\n")
		fmt.Fprintf(os.Stderr, "  Example:\n")
		fmt.Fprintf(os.Stderr, "    Alice Bob 1609459200\n")
		fmt.Fprintf(os.Stderr, "    Bob Charlie 1609462800\n")
		fmt.Fprintf(os.Stderr, "    Alice Charlie 1609466400\n\n")
		fmt.Fprintf(os.Stderr, "Time Window:\n")
		fmt.Fprintf(os.Stderr, "  The time window controls how far ahead in time a walk can progress.\n")
		fmt.Fprintf(os.Stderr, "  Larger windows allow longer temporal dependencies.\n")
		fmt.Fprintf(os.Stderr, "  If set to 0, automatically uses 10%% of the graph's time span.\n\n")
		fmt.Fprintf(os.Stderr, "Examples:\n")
		fmt.Fprintf(os.Stderr, "  # Train on social network interactions (auto time window)\n")
		fmt.Fprintf(os.Stderr, "  %s -train social.txt -output ctdne.emb \\\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "    -dim 128 -walk-times 10 -walk-steps 80\n\n")
		fmt.Fprintf(os.Stderr, "  # Train on transaction network (custom time window)\n")
		fmt.Fprintf(os.Stderr, "  %s -train transactions.txt -output ctdne.emb \\\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "    -dim 128 -time-window 86400 -walk-times 20\n\n")
		fmt.Fprintf(os.Stderr, "  # Train on communication network (high-dimensional)\n")
		fmt.Fprintf(os.Stderr, "  %s -train emails.txt -output ctdne.emb \\\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "    -dim 256 -walk-times 15 -workers 8\n\n")
		fmt.Fprintf(os.Stderr, "Key Features:\n")
		fmt.Fprintf(os.Stderr, "  ✓ Time-respecting random walks\n")
		fmt.Fprintf(os.Stderr, "  ✓ Captures temporal network dynamics\n")
		fmt.Fprintf(os.Stderr, "  ✓ Activity-weighted negative sampling\n")
		fmt.Fprintf(os.Stderr, "  ✓ Parallel training for scalability\n\n")
		fmt.Fprintf(os.Stderr, "Use Cases:\n")
		fmt.Fprintf(os.Stderr, "  - Social network evolution analysis\n")
		fmt.Fprintf(os.Stderr, "  - Communication pattern detection\n")
		fmt.Fprintf(os.Stderr, "  - Transaction fraud detection\n")
		fmt.Fprintf(os.Stderr, "  - Citation network dynamics\n")
		fmt.Fprintf(os.Stderr, "  - Temporal link prediction\n\n")
		fmt.Fprintf(os.Stderr, "References:\n")
		fmt.Fprintf(os.Stderr, "  Nguyen et al. \"Continuous-Time Dynamic Network Embeddings\", WWW 2018\n")
		fmt.Fprintf(os.Stderr, "  https://dl.acm.org/doi/10.1145/3184558.3191526\n\n")
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

	if *walkTimes <= 0 {
		fmt.Fprintf(os.Stderr, "Error: -walk-times must be positive\n")
		os.Exit(1)
	}

	if *walkSteps <= 0 {
		fmt.Fprintf(os.Stderr, "Error: -walk-steps must be positive\n")
		os.Exit(1)
	}

	if *windowSize <= 0 {
		fmt.Fprintf(os.Stderr, "Error: -window-size must be positive\n")
		os.Exit(1)
	}

	if *negativeSamples < 0 {
		fmt.Fprintf(os.Stderr, "Error: -negative-samples must be non-negative\n")
		os.Exit(1)
	}

	if *alpha <= 0 {
		fmt.Fprintf(os.Stderr, "Error: -alpha must be positive\n")
		os.Exit(1)
	}

	if *timeWindow < 0 {
		fmt.Fprintf(os.Stderr, "Error: -time-window must be non-negative (0 = auto)\n")
		os.Exit(1)
	}

	if *workers <= 0 {
		fmt.Fprintf(os.Stderr, "Error: -workers must be positive\n")
		os.Exit(1)
	}

	// Print configuration
	fmt.Println("===================================================")
	fmt.Println("CTDNE - Continuous-Time Dynamic Network Embeddings")
	fmt.Println("===================================================")
	fmt.Println()

	// Create CTDNE model
	model := ctdne.New()

	// Load temporal graph
	fmt.Println("Loading Temporal Graph:")
	fmt.Printf("\tInput: %s\n", *train)
	if *timeWindow > 0 {
		fmt.Printf("\tTime Window: %.2f\n", *timeWindow)
	} else {
		fmt.Println("\tTime Window: auto (10% of time span)")
	}
	fmt.Println()

	if err := model.LoadEdgeList(*train); err != nil {
		fmt.Fprintf(os.Stderr, "Error loading graph: %v\n", err)
		os.Exit(1)
	}

	// Initialize model
	fmt.Println()
	model.Init(*dim, *timeWindow)

	// Train model
	fmt.Println()
	model.Train(*walkTimes, *walkSteps, *windowSize, *negativeSamples, *alpha, *workers)

	// Compute temporal coherence
	model.ComputeTemporalCoherence()

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
