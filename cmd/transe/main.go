package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/cnclabs/smore/internal/models/transe"
)

func main() {
	// Define command-line flags
	train := flag.String("train", "", "Train on knowledge graph triples")
	saveEntity := flag.String("save_entity", "", "Save entity embeddings")
	saveRelation := flag.String("save_relation", "", "Save relation embeddings")
	dimensions := flag.Int("dimensions", 50, "Dimension of embeddings")
	margin := flag.Float64("margin", 1.0, "Margin for ranking loss")
	norm := flag.Int("norm", 2, "Norm type: 1 for L1 (Manhattan), 2 for L2 (Euclidean)")
	epochs := flag.Int("epochs", 100, "Number of training epochs")
	batchSize := flag.Int("batch_size", 128, "Batch size for training")
	alpha := flag.Float64("alpha", 0.01, "Learning rate")
	threads := flag.Int("threads", 4, "Number of training threads")

	flag.Usage = func() {
		fmt.Println("[SMORe-Go]")
		fmt.Println("\tGolang implementation of SMORe - TransE")
		fmt.Println()
		fmt.Println("TransE (Translating Embeddings) for Knowledge Graphs:")
		fmt.Println("\t✓ Translation principle: h + r ≈ t")
		fmt.Println("\t✓ Simple and effective baseline for KG embedding")
		fmt.Println("\t✓ Great for link prediction tasks")
		fmt.Println("\t✓ Learns entity and relation embeddings jointly")
		fmt.Println()
		fmt.Println("How it works:")
		fmt.Println("\t1. Entities and relations are embedded in the same space")
		fmt.Println("\t2. Relations are modeled as translations: h + r ≈ t")
		fmt.Println("\t3. Trained with margin-based ranking loss")
		fmt.Println("\t4. Entity embeddings are L2-normalized")
		fmt.Println()
		fmt.Println("Input format (triples):")
		fmt.Println("\thead relation tail [weight]")
		fmt.Println("\tExample: Barack_Obama born_in Hawaii 1.0")
		fmt.Println()
		fmt.Println("Key parameters:")
		fmt.Println("\t- dimensions: Embedding dimension (default: 50)")
		fmt.Println("\t  • Typical range: 50-200")
		fmt.Println("\t- margin: Margin for ranking loss (default: 1.0)")
		fmt.Println("\t  • Higher = stricter separation")
		fmt.Println("\t- norm: Distance metric (default: 2)")
		fmt.Println("\t  • 1 = L1/Manhattan distance")
		fmt.Println("\t  • 2 = L2/Euclidean distance")
		fmt.Println("\t- epochs: Training epochs (default: 100)")
		fmt.Println("\t- batch_size: Batch size (default: 128)")
		fmt.Println()
		fmt.Println("Options Description:")
		flag.PrintDefaults()
		fmt.Println()
		fmt.Println("Usage:")
		fmt.Println("./transe -train kg.txt -save_entity entities.txt -save_relation relations.txt \\")
		fmt.Println("         -dimensions 50 -margin 1.0 -norm 2 -epochs 100 -batch_size 128 \\")
		fmt.Println("         -alpha 0.01 -threads 4")
		fmt.Println()
		fmt.Println("Examples:")
		fmt.Println("\t# Quick training with defaults")
		fmt.Println("\t./transe -train kg.txt -save_entity ent.txt -save_relation rel.txt")
		fmt.Println()
		fmt.Println("\t# High-quality training")
		fmt.Println("\t./transe -train kg.txt -save_entity ent.txt -save_relation rel.txt \\")
		fmt.Println("\t         -dimensions 100 -epochs 500 -alpha 0.001")
		fmt.Println()
		fmt.Println("\t# L1 norm (better for sparse KGs)")
		fmt.Println("\t./transe -train kg.txt -save_entity ent.txt -save_relation rel.txt -norm 1")
		fmt.Println()
		fmt.Println("Applications:")
		fmt.Println("\t✓ Link prediction (predict missing triples)")
		fmt.Println("\t✓ Triplet classification (true/false)")
		fmt.Println("\t✓ Entity resolution")
		fmt.Println("\t✓ Relation extraction")
		fmt.Println()
		fmt.Println("Benchmarks:")
		fmt.Println("\tCommon datasets: FB15k, FB15k-237, WN18, WN18RR, YAGO3-10")
	}

	flag.Parse()

	// Check required parameters
	if *train == "" || *saveEntity == "" || *saveRelation == "" {
		flag.Usage()
		os.Exit(1)
	}

	// Validate parameters
	if *dimensions <= 0 {
		fmt.Println("Error: dimensions must be positive")
		os.Exit(1)
	}
	if *margin <= 0 {
		fmt.Println("Error: margin must be positive")
		os.Exit(1)
	}
	if *norm != 1 && *norm != 2 {
		fmt.Println("Error: norm must be 1 (L1) or 2 (L2)")
		os.Exit(1)
	}
	if *epochs <= 0 {
		fmt.Println("Error: epochs must be positive")
		os.Exit(1)
	}
	if *batchSize <= 0 {
		fmt.Println("Error: batch_size must be positive")
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
	fmt.Println("  TransE - Translating Embeddings for Knowledge Graphs")
	fmt.Println("═══════════════════════════════════════════════════════")
	fmt.Println()

	startTime := time.Now()

	// Create and train model
	te := transe.New()

	fmt.Println("Loading knowledge graph...")
	if err := te.LoadTriples(*train); err != nil {
		fmt.Printf("Error loading triples: %v\n", err)
		os.Exit(1)
	}

	loadTime := time.Since(startTime)
	fmt.Printf("Knowledge graph loaded in %.2f seconds\n", loadTime.Seconds())
	fmt.Println()

	te.Init(*dimensions, *margin, *norm)
	fmt.Println()

	trainStartTime := time.Now()
	te.Train(*epochs, *batchSize, *alpha, *threads)
	trainTime := time.Since(trainStartTime)

	// Save embeddings
	fmt.Println()
	if err := te.SaveEmbeddings(*saveEntity, *saveRelation); err != nil {
		fmt.Printf("Error saving embeddings: %v\n", err)
		os.Exit(1)
	}

	totalTime := time.Since(startTime)
	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════")
	fmt.Println("  Timing Summary")
	fmt.Println("═══════════════════════════════════════════════════════")
	fmt.Printf("Loading time:     %.2f seconds\n", loadTime.Seconds())
	fmt.Printf("Training time:    %.2f seconds\n", trainTime.Seconds())
	fmt.Printf("Total time:       %.2f seconds\n", totalTime.Seconds())
	fmt.Println()
	fmt.Println("✓ TransE training complete!")
	fmt.Println()
	fmt.Println("Next steps:")
	fmt.Println("\t• Use embeddings for link prediction")
	fmt.Println("\t• Evaluate on test triples")
	fmt.Println("\t• Try different hyperparameters for better results")
}
