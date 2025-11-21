package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/cnclabs/smore/internal/models/recdenoiser"
)

func main() {
	train := flag.String("train", "", "Train the Network data (user-item edge list)")
	save := flag.String("save", "", "Save the representation data")
	dimensions := flag.Int("dimensions", 64, "Dimension of item representation")
	maxSeqLen := flag.Int("max_seq_len", 50, "Maximum sequence length")
	numBlocks := flag.Int("num_blocks", 2, "Number of self-attention blocks")
	numHeads := flag.Int("num_heads", 1, "Number of attention heads")
	dropout := flag.Float64("dropout", 0.2, "Dropout rate")
	sparsity := flag.Float64("sparsity", 0.3, "Target sparsity rate for attention pruning (0.3 = 30%)")
	lambdaSparsity := flag.Float64("lambda_sparsity", 0.01, "Sparsity regularization weight")
	epochs := flag.Int("epochs", 10, "Number of training epochs")
	batchSize := flag.Int("batch_size", 128, "Batch size")
	negativeSamples := flag.Int("negative_samples", 1, "Number of negative samples")
	alpha := flag.Float64("alpha", 0.001, "Learning rate")
	threads := flag.Int("threads", 1, "Number of training threads")

	flag.Usage = func() {
		fmt.Println("[SMORe-Go]")
		fmt.Println("\tGolang implementation of SMORe - Rec-Denoiser")
		fmt.Println()
		fmt.Println("Description:")
		fmt.Println("\t⭐ RecSys 2022 Best Paper Award Winner ⭐")
		fmt.Println("\tDenoising Self-Attentive Sequential Recommendation")
		fmt.Println("\tBy Huiyuan Chen, Yusan Lin, Menghai Pan, et al.")
		fmt.Println()
		fmt.Println("Key Innovation:")
		fmt.Println("\tTrainable binary masks to prune noisy attentions")
		fmt.Println("\tSparse and clean attention distributions")
		fmt.Println("\t+5.05% to +19.55% performance gains over SASRec")
		fmt.Println("\tRobustness against input perturbations")
		fmt.Println()
		fmt.Println("Options:")
		flag.PrintDefaults()
		fmt.Println()
		fmt.Println("Usage:")
		fmt.Println("  ./recdenoiser -train user_item.txt -save embeddings.txt -dimensions 64 -max_seq_len 50 -num_blocks 2 -sparsity 0.3 -lambda_sparsity 0.01 -epochs 10 -alpha 0.001 -threads 4")
		fmt.Println()
		fmt.Println("Input Format:")
		fmt.Println("  user_id item_id (one interaction per line, chronologically ordered)")
		fmt.Println()
		fmt.Println("Sparsity Parameter:")
		fmt.Println("  sparsity=0.3:  30% of attentions pruned (recommended)")
		fmt.Println("  sparsity=0.5:  50% pruning (more aggressive)")
		fmt.Println("  sparsity=0.1:  10% pruning (conservative)")
		fmt.Println()
		fmt.Println("How Denoising Works:")
		fmt.Println("  1. Each attention layer learns a trainable binary mask")
		fmt.Println("  2. Masks identify and prune noisy item-item dependencies")
		fmt.Println("  3. Results in sparse, clean attention distributions")
		fmt.Println("  4. Improved recommendations by filtering irrelevant items")
	}

	flag.Parse()

	if *train == "" || *save == "" {
		fmt.Println("Error: -train and -save are required")
		flag.Usage()
		os.Exit(1)
	}

	// Create and initialize model
	model := recdenoiser.New()

	// Load graph
	fmt.Println("Loading graph...")
	if err := model.LoadEdgeList(*train, false); err != nil {
		fmt.Printf("Error loading graph: %v\n", err)
		os.Exit(1)
	}

	// Build sequences from graph
	model.BuildSequencesFromGraph()

	// Initialize model with denoising parameters
	model.Init(*dimensions, *maxSeqLen, *numBlocks, *numHeads, *dropout, *sparsity)

	// Train model with trainable masks
	model.Train(*epochs, *batchSize, *negativeSamples, *alpha, *lambdaSparsity, *threads)

	// Save embeddings
	if err := model.SaveWeights(*save); err != nil {
		fmt.Printf("Error saving weights: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("Training completed successfully!")
	fmt.Println("\n✨ This model implements the RecSys 2022 Best Paper Award winner!")
	fmt.Println("💡 Noisy attentions have been pruned for cleaner recommendations!")
}
