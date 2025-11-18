package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/cnclabs/smore/internal/models/gsasrec"
)

func main() {
	train := flag.String("train", "", "Train the Network data (user-item edge list)")
	save := flag.String("save", "", "Save the representation data")
	dimensions := flag.Int("dimensions", 64, "Dimension of item representation")
	maxSeqLen := flag.Int("max_seq_len", 50, "Maximum sequence length")
	numBlocks := flag.Int("num_blocks", 2, "Number of self-attention blocks")
	numHeads := flag.Int("num_heads", 1, "Number of attention heads")
	dropout := flag.Float64("dropout", 0.2, "Dropout rate")
	beta := flag.Float64("beta", 0.5, "Beta parameter for generalized BCE loss (0.5 = balanced)")
	epochs := flag.Int("epochs", 10, "Number of training epochs")
	batchSize := flag.Int("batch_size", 128, "Batch size")
	negativeSamples := flag.Int("negative_samples", 1, "Number of negative samples")
	alpha := flag.Float64("alpha", 0.001, "Learning rate")
	threads := flag.Int("threads", 1, "Number of training threads")

	flag.Usage = func() {
		fmt.Println("[SMORe-Go]")
		fmt.Println("\tGolang implementation of SMORe - gSASRec")
		fmt.Println()
		fmt.Println("Description:")
		fmt.Println("\t⭐ RecSys 2023 Best Paper Award Winner ⭐")
		fmt.Println("\tgSASRec: Reducing Overconfidence in Sequential Recommendation")
		fmt.Println("\tBy Aleksandr V. Petrov and Craig Macdonald")
		fmt.Println()
		fmt.Println("Key Innovation:")
		fmt.Println("\tGeneralized Binary Cross-Entropy (gBCE) loss to mitigate overconfidence")
		fmt.Println("\t+9.47% NDCG improvement over BERT4Rec on MovieLens-1M")
		fmt.Println("\t-73% training time reduction compared to BERT4Rec")
		fmt.Println()
		fmt.Println("Options:")
		flag.PrintDefaults()
		fmt.Println()
		fmt.Println("Usage:")
		fmt.Println("  ./gsasrec -train user_item.txt -save embeddings.txt -dimensions 64 -max_seq_len 50 -num_blocks 2 -beta 0.5 -epochs 10 -alpha 0.001 -threads 4")
		fmt.Println()
		fmt.Println("Input Format:")
		fmt.Println("  user_id item_id (one interaction per line, chronologically ordered)")
		fmt.Println()
		fmt.Println("Beta Parameter:")
		fmt.Println("  beta=0.5: Balanced (recommended)")
		fmt.Println("  beta>0.5: More weight on positive samples")
		fmt.Println("  beta<0.5: More weight on negative samples")
	}

	flag.Parse()

	if *train == "" || *save == "" {
		fmt.Println("Error: -train and -save are required")
		flag.Usage()
		os.Exit(1)
	}

	// Create and initialize model
	model := gsasrec.New()

	// Load graph
	fmt.Println("Loading graph...")
	if err := model.LoadEdgeList(*train, false); err != nil {
		fmt.Printf("Error loading graph: %v\n", err)
		os.Exit(1)
	}

	// Build sequences from graph
	model.BuildSequencesFromGraph()

	// Initialize model with beta parameter
	model.Init(*dimensions, *maxSeqLen, *numBlocks, *numHeads, *dropout, *beta)

	// Train model with gBCE loss
	model.Train(*epochs, *batchSize, *negativeSamples, *alpha, *threads)

	// Save embeddings
	if err := model.SaveWeights(*save); err != nil {
		fmt.Printf("Error saving weights: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("Training completed successfully!")
	fmt.Println("\n✨ This model implements the RecSys 2023 Best Paper Award winner!")
}
