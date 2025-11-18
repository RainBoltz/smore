package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/cnclabs/smore/internal/models/rotate"
)

func main() {
	// Define command-line flags
	train := flag.String("train", "", "Train on knowledge graph triples")
	saveEntity := flag.String("save_entity", "", "Save entity embeddings")
	saveRelation := flag.String("save_relation", "", "Save relation embeddings")
	dimensions := flag.Int("dimensions", 100, "Dimension of embeddings (will be halved for complex)")
	margin := flag.Float64("margin", 6.0, "Margin for ranking loss")
	adversarialTemp := flag.Float64("adversarial_temp", 1.0, "Temperature for self-adversarial negative sampling")
	epochs := flag.Int("epochs", 100, "Number of training epochs")
	batchSize := flag.Int("batch_size", 128, "Batch size for training")
	alpha := flag.Float64("alpha", 0.0001, "Learning rate")
	threads := flag.Int("threads", 4, "Number of training threads")

	flag.Usage = func() {
		fmt.Println("[SMORe-Go]")
		fmt.Println("\tGolang implementation of SMORe - RotatE")
		fmt.Println()
		fmt.Println("RotatE - Rotation-based Knowledge Graph Embedding:")
		fmt.Println("\t✓ Uses complex-valued embeddings")
		fmt.Println("\t✓ Models relations as rotations: h ∘ r ≈ t")
		fmt.Println("\t✓ State-of-the-art on FB15k, WN18, etc.")
		fmt.Println("\t✓ Handles symmetry, antisymmetry, inversion, composition")
		fmt.Println("\t✓ Better than TransE for complex relation patterns")
		fmt.Println()
		fmt.Println("How it works:")
		fmt.Println("\t1. Entities: complex-valued embeddings")
		fmt.Println("\t2. Relations: unit complex numbers (rotations)")
		fmt.Println("\t3. Score: ||h ∘ r - t|| where ∘ is complex multiplication")
		fmt.Println("\t4. Self-adversarial negative sampling")
		fmt.Println()
		fmt.Println("Input format (triples):")
		fmt.Println("\thead relation tail [weight]")
		fmt.Println("\tExample: Barack_Obama born_in Hawaii 1.0")
		fmt.Println()
		fmt.Println("Key parameters:")
		fmt.Println("\t- dimensions: Embedding dimension (default: 100)")
		fmt.Println("\t  • Will be halved for complex representation")
		fmt.Println("\t  • Example: 100 dims → 50 complex numbers")
		fmt.Println("\t  • Typical range: 100-1000")
		fmt.Println("\t- margin: Margin for ranking loss (default: 6.0)")
		fmt.Println("\t  • Higher than TransE due to different scoring")
		fmt.Println("\t  • Typical range: 3-12")
		fmt.Println("\t- adversarial_temp: Self-adversarial temperature (default: 1.0)")
		fmt.Println("\t  • Higher = more uniform negative sampling")
		fmt.Println("\t  • Lower = focus on hard negatives")
		fmt.Println("\t- epochs: Training epochs (default: 100)")
		fmt.Println("\t- batch_size: Batch size (default: 128)")
		fmt.Println()
		fmt.Println("Options Description:")
		flag.PrintDefaults()
		fmt.Println()
		fmt.Println("Usage:")
		fmt.Println("./rotate -train kg.txt -save_entity entities.txt -save_relation relations.txt \\")
		fmt.Println("         -dimensions 100 -margin 6.0 -adversarial_temp 1.0 \\")
		fmt.Println("         -epochs 100 -batch_size 128 -alpha 0.0001 -threads 4")
		fmt.Println()
		fmt.Println("Examples:")
		fmt.Println("\t# Quick training with defaults")
		fmt.Println("\t./rotate -train kg.txt -save_entity ent.txt -save_relation rel.txt")
		fmt.Println()
		fmt.Println("\t# High-dimensional training (better quality)")
		fmt.Println("\t./rotate -train kg.txt -save_entity ent.txt -save_relation rel.txt \\")
		fmt.Println("\t         -dimensions 500 -epochs 500")
		fmt.Println()
		fmt.Println("\t# Focus on hard negatives")
		fmt.Println("\t./rotate -train kg.txt -save_entity ent.txt -save_relation rel.txt \\")
		fmt.Println("\t         -adversarial_temp 0.5")
		fmt.Println()
		fmt.Println("Advantages over TransE:")
		fmt.Println("\t✓ Symmetry: r(h,t) = r(t,h)")
		fmt.Println("\t✓ Antisymmetry: r(h,t) ⟹ ¬r(t,h)")
		fmt.Println("\t✓ Inversion: r₂(h,t) ⟹ r₁(t,h)")
		fmt.Println("\t✓ Composition: r₁(h,x) ∧ r₂(x,t) ⟹ r₃(h,t)")
		fmt.Println()
		fmt.Println("Applications:")
		fmt.Println("\t✓ Link prediction (especially complex relations)")
		fmt.Println("\t✓ Relation pattern modeling")
		fmt.Println("\t✓ Knowledge graph completion")
		fmt.Println("\t✓ Multi-hop reasoning")
		fmt.Println()
		fmt.Println("Benchmarks:")
		fmt.Println("\tState-of-the-art on: FB15k, FB15k-237, WN18, WN18RR")
		fmt.Println("\tTypical MRR improvement: 5-10% over TransE")
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
	if *adversarialTemp < 0 {
		fmt.Println("Error: adversarial_temp must be non-negative")
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
	fmt.Println("  RotatE - Rotation-based Knowledge Graph Embedding")
	fmt.Println("═══════════════════════════════════════════════════════")
	fmt.Println()

	startTime := time.Now()

	// Create and train model
	re := rotate.New()

	fmt.Println("Loading knowledge graph...")
	if err := re.LoadTriples(*train); err != nil {
		fmt.Printf("Error loading triples: %v\n", err)
		os.Exit(1)
	}

	loadTime := time.Since(startTime)
	fmt.Printf("Knowledge graph loaded in %.2f seconds\n", loadTime.Seconds())
	fmt.Println()

	re.Init(*dimensions, *margin, *adversarialTemp)

	trainStartTime := time.Now()
	re.Train(*epochs, *batchSize, *alpha, *threads)
	trainTime := time.Since(trainStartTime)

	// Save embeddings
	fmt.Println()
	if err := re.SaveEmbeddings(*saveEntity, *saveRelation); err != nil {
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
	fmt.Println("✓ RotatE training complete!")
	fmt.Println()
	fmt.Println("Next steps:")
	fmt.Println("\t• Evaluate on link prediction tasks")
	fmt.Println("\t• Compare with TransE on same dataset")
	fmt.Println("\t• Try different hyperparameters")
	fmt.Println("\t• Test on complex relation patterns")
}
