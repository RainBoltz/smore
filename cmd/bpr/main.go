package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/cnclabs/smore/internal/models/bpr"
)

func main() {
	// Define command-line flags
	train := flag.String("train", "", "Train the Network data")
	save := flag.String("save", "", "Save the representation data")
	dimensions := flag.Int("dimensions", 64, "Dimension of vertex representation")
	undirected := flag.Bool("undirected", false, "Whether the edge is undirected")
	sampleTimes := flag.Int("sample_times", 10, "Number of training iterations")
	threads := flag.Int("threads", 1, "Number of training threads")
	alpha := flag.Float64("alpha", 0.025, "Init learning rate")
	lambda := flag.Float64("lambda", 0.001, "Regularization parameter")

	flag.Usage = func() {
		fmt.Println("[SMORe-Go]")
		fmt.Println("\tGolang implementation of SMORe - BPR")
		fmt.Println()
		fmt.Println("Options Description:")
		flag.PrintDefaults()
		fmt.Println()
		fmt.Println("Usage:")
		fmt.Println("./bpr -train net.txt -save rep.txt -dimensions 64 -sample_times 10 -alpha 0.025 -lambda 0.001 -threads 1")
	}

	flag.Parse()

	// Check required parameters
	if *train == "" || *save == "" {
		flag.Usage()
		os.Exit(1)
	}

	// Create and train model
	b := bpr.New()

	if err := b.LoadEdgeList(*train, *undirected); err != nil {
		fmt.Printf("Error loading edge list: %v\n", err)
		os.Exit(1)
	}

	b.Init(*dimensions)
	b.Train(*sampleTimes, *alpha, *lambda, *threads)

	if err := b.SaveWeights(*save); err != nil {
		fmt.Printf("Error saving weights: %v\n", err)
		os.Exit(1)
	}
}
