package main

import (
	"flag"
	"fmt"
	"os"

	complex_embeddings "github.com/cnclabs/smore/internal/models/complex"
)

func main() {
	// Command-line flags
	train := flag.String("train", "", "Path to training triples file (format: head relation tail)")
	output := flag.String("output", "", "Path to output embeddings file")
	dim := flag.Int("dim", 100, "Embedding dimension")
	epochs := flag.Int("epochs", 100, "Number of training epochs")
	batchSize := flag.Int("batch-size", 128, "Batch size for training")
	negativeSamples := flag.Int("negative-samples", 10, "Number of negative samples per positive triple")
	learningRate := flag.Float64("lr", 0.01, "Learning rate")
	margin := flag.Float64("margin", 1.0, "Margin for ranking loss")
	workers := flag.Int("workers", 4, "Number of parallel workers")
	evalSize := flag.Int("eval-size", 1000, "Number of triples to use for evaluation")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "ComplEx - Complex Embeddings for Knowledge Graphs\n\n")
		fmt.Fprintf(os.Stderr, "Usage:\n")
		fmt.Fprintf(os.Stderr, "  %s [options]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Description:\n")
		fmt.Fprintf(os.Stderr, "  ComplEx learns complex-valued embeddings for knowledge graph entities and relations.\n")
		fmt.Fprintf(os.Stderr, "  Unlike real-valued models, ComplEx can capture:\n")
		fmt.Fprintf(os.Stderr, "  - Symmetric relations (e.g., \"is_similar_to\")\n")
		fmt.Fprintf(os.Stderr, "  - Antisymmetric relations (e.g., \"is_parent_of\")\n")
		fmt.Fprintf(os.Stderr, "  - Inverse relations\n")
		fmt.Fprintf(os.Stderr, "  - Composition patterns\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nInput Format:\n")
		fmt.Fprintf(os.Stderr, "  Triple format: head_entity relation tail_entity\n")
		fmt.Fprintf(os.Stderr, "  Example:\n")
		fmt.Fprintf(os.Stderr, "    Paris capitalOf France\n")
		fmt.Fprintf(os.Stderr, "    France locatedIn Europe\n")
		fmt.Fprintf(os.Stderr, "    Berlin capitalOf Germany\n\n")
		fmt.Fprintf(os.Stderr, "Output Format:\n")
		fmt.Fprintf(os.Stderr, "  Complex-valued embeddings in format:\n")
		fmt.Fprintf(os.Stderr, "  E entity_name real1 imag1i real2 imag2i ...\n")
		fmt.Fprintf(os.Stderr, "  R relation_name real1 imag1i real2 imag2i ...\n\n")
		fmt.Fprintf(os.Stderr, "Examples:\n")
		fmt.Fprintf(os.Stderr, "  # Train on FB15k knowledge graph\n")
		fmt.Fprintf(os.Stderr, "  %s -train fb15k.txt -output complex.emb \\\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "    -dim 100 -epochs 100 -lr 0.01 -negative-samples 10\n\n")
		fmt.Fprintf(os.Stderr, "  # Train on WordNet with higher dimensions\n")
		fmt.Fprintf(os.Stderr, "  %s -train wn18.txt -output complex.emb \\\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "    -dim 200 -epochs 200 -lr 0.005 -margin 2.0\n\n")
		fmt.Fprintf(os.Stderr, "  # Train with more workers for speed\n")
		fmt.Fprintf(os.Stderr, "  %s -train data.txt -output complex.emb \\\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "    -dim 150 -epochs 150 -workers 8\n\n")
		fmt.Fprintf(os.Stderr, "Key Features:\n")
		fmt.Fprintf(os.Stderr, "  ✓ Complex-valued embeddings (uses Go's native complex128)\n")
		fmt.Fprintf(os.Stderr, "  ✓ Handles symmetric and antisymmetric relations\n")
		fmt.Fprintf(os.Stderr, "  ✓ More expressive than real-valued models (TransE)\n")
		fmt.Fprintf(os.Stderr, "  ✓ Parallel training with goroutines\n")
		fmt.Fprintf(os.Stderr, "  ✓ Margin-based ranking loss\n\n")
		fmt.Fprintf(os.Stderr, "Scoring Function:\n")
		fmt.Fprintf(os.Stderr, "  score(h, r, t) = Re(<h, r, conj(t)>)\n")
		fmt.Fprintf(os.Stderr, "  where <> is the trilinear dot product:\n")
		fmt.Fprintf(os.Stderr, "  Σ h_i * r_i * conj(t_i)\n\n")
		fmt.Fprintf(os.Stderr, "References:\n")
		fmt.Fprintf(os.Stderr, "  Trouillon et al. \"Complex Embeddings for Simple Link Prediction\", ICML 2016\n")
		fmt.Fprintf(os.Stderr, "  https://arxiv.org/abs/1606.06357\n\n")
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

	if *batchSize <= 0 {
		fmt.Fprintf(os.Stderr, "Error: -batch-size must be positive\n")
		os.Exit(1)
	}

	if *negativeSamples <= 0 {
		fmt.Fprintf(os.Stderr, "Error: -negative-samples must be positive\n")
		os.Exit(1)
	}

	if *learningRate <= 0 {
		fmt.Fprintf(os.Stderr, "Error: -lr must be positive\n")
		os.Exit(1)
	}

	if *margin <= 0 {
		fmt.Fprintf(os.Stderr, "Error: -margin must be positive\n")
		os.Exit(1)
	}

	if *workers <= 0 {
		fmt.Fprintf(os.Stderr, "Error: -workers must be positive\n")
		os.Exit(1)
	}

	// Print configuration
	fmt.Println("===================================================")
	fmt.Println("ComplEx - Complex Embeddings for Knowledge Graphs")
	fmt.Println("===================================================")
	fmt.Println()

	// Create ComplEx model
	model := complex_embeddings.New()

	// Load knowledge graph
	fmt.Println("Loading Knowledge Graph:")
	fmt.Printf("\tInput: %s\n", *train)
	fmt.Println()

	if err := model.LoadTriples(*train); err != nil {
		fmt.Fprintf(os.Stderr, "Error loading triples: %v\n", err)
		os.Exit(1)
	}

	// Initialize model
	fmt.Println()
	model.Init(*dim, *learningRate, *margin)

	// Train model
	fmt.Println()
	model.Train(*epochs, *batchSize, *negativeSamples, *workers)

	// Evaluate link prediction
	if *evalSize > 0 {
		model.EvaluateLinkPrediction(*evalSize)
	}

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
