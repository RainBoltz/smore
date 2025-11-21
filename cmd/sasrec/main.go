package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/cnclabs/smore/internal/models/sasrec"
)

func main() {
	train := flag.String("train", "", "Train the Network data (user-item edge list)")
	save := flag.String("save", "", "Save the representation data")
	dimensions := flag.Int("dimensions", 64, "Dimension of item representation")
	maxSeqLen := flag.Int("max_seq_len", 50, "Maximum sequence length")
	numBlocks := flag.Int("num_blocks", 2, "Number of self-attention blocks")
	numHeads := flag.Int("num_heads", 1, "Number of attention heads")
	dropout := flag.Float64("dropout", 0.2, "Dropout rate")
	epochs := flag.Int("epochs", 10, "Number of training epochs")
	batchSize := flag.Int("batch_size", 128, "Batch size")
	negativeSamples := flag.Int("negative_samples", 1, "Number of negative samples")
	alpha := flag.Float64("alpha", 0.001, "Learning rate")
	threads := flag.Int("threads", 1, "Number of training threads")

	flag.Usage = func() {
		fmt.Println("[SMORe-Go]")
		fmt.Println("\tGolang implementation of SMORe - SASRec")
		fmt.Println()
		fmt.Println("Description:")
		fmt.Println("\tSelf-Attentive Sequential Recommendation (ICDM 2018)")
		fmt.Println("\tBy Wang-Cheng Kang and Julian McAuley")
		fmt.Println()
		fmt.Println("Options:")
		flag.PrintDefaults()
		fmt.Println()
		fmt.Println("Usage:")
		fmt.Println("  ./sasrec -train user_item.txt -save embeddings.txt -dimensions 64 -max_seq_len 50 -num_blocks 2 -epochs 10 -alpha 0.001 -threads 4")
		fmt.Println()
		fmt.Println("Input Format:")
		fmt.Println("  user_id item_id (one interaction per line, chronologically ordered)")
	}

	flag.Parse()

	if *train == "" || *save == "" {
		fmt.Println("Error: -train and -save are required")
		flag.Usage()
		os.Exit(1)
	}

	// Create and initialize model
	model := sasrec.New()

	// Load graph
	fmt.Println("Loading graph...")
	if err := model.LoadEdgeList(*train, false); err != nil {
		fmt.Printf("Error loading graph: %v\n", err)
		os.Exit(1)
	}

	// Build sequences from graph
	model.BuildSequencesFromGraph()

	// Initialize model
	model.Init(*dimensions, *maxSeqLen, *numBlocks, *numHeads, *dropout)

	// Train model
	model.Train(*epochs, *batchSize, *negativeSamples, *alpha, *threads)

	// Save embeddings
	if err := model.SaveWeights(*save); err != nil {
		fmt.Printf("Error saving weights: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("Training completed successfully!")
}
