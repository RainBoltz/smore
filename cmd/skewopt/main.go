package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/cnclabs/smore/internal/models/skewopt"
)

func main() {
	train := flag.String("train", "", "Train the Network data")
	save := flag.String("save", "", "Save the representation data")
	dimensions := flag.Int("dimensions", 64, "Dimension of vertex representation")
	undirected := flag.Bool("undirected", true, "Whether the edge is undirected")
	negativeSamples := flag.Int("negative_samples", 5, "Number of negative examples")
	sampleTimes := flag.Int("sample_times", 10, "Number of training samples (in millions)")
	reg := flag.Float64("reg", 0.01, "Regularization term")
	xi := flag.Float64("xi", 10.0, "Skewness parameter")
	omega := flag.Float64("omega", 3.0, "Proximity weight")
	eta := flag.Int("eta", 3, "Balance parameter (exponent)")
	alpha := flag.Float64("alpha", 0.025, "Init learning rate")
	threads := flag.Int("threads", 1, "Number of training threads")

	flag.Usage = func() {
		fmt.Println("[SMORe-Go]")
		fmt.Println("\tGolang implementation of SMORe - Skew-Opt")
		fmt.Println()
		fmt.Println("Description:")
		fmt.Println("\tSkew-Opt: Skewness Ranking Optimization for personalized recommendation")
		fmt.Println()
		fmt.Println("Options:")
		flag.PrintDefaults()
		fmt.Println()
		fmt.Println("Usage:")
		fmt.Println("  ./skewopt -train net.txt -save rep.txt -dimensions 64 -sample_times 10 -xi 10.0 -omega 3.0 -eta 3 -alpha 0.025 -threads 1")
		fmt.Println()
		fmt.Println("Parameters:")
		fmt.Println("  xi:    Skewness parameter (controls the skewness of the preference distribution)")
		fmt.Println("  omega: Proximity weight (scales the proximity score)")
		fmt.Println("  eta:   Balance parameter (exponent for the skewness function)")
	}

	flag.Parse()

	if *train == "" || *save == "" {
		fmt.Println("Error: -train and -save are required")
		flag.Usage()
		os.Exit(1)
	}

	// Create and initialize model
	model := skewopt.New()

	// Load graph
	fmt.Println("Loading graph...")
	if err := model.LoadEdgeList(*train, *undirected); err != nil {
		fmt.Printf("Error loading graph: %v\n", err)
		os.Exit(1)
	}

	// Initialize embeddings
	model.Init(*dimensions)

	// Train model
	model.Train(*sampleTimes, *negativeSamples, *alpha, *reg, *xi, *omega, *eta, *threads)

	// Save embeddings
	if err := model.SaveWeights(*save); err != nil {
		fmt.Printf("Error saving weights: %v\n", err)
		os.Exit(1)
	}
}
