package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/cnclabs/smore/internal/models/deepwalk"
)

func main() {
	// Define command-line flags
	train := flag.String("train", "", "Train the Network data")
	save := flag.String("save", "", "Save the representation data")
	dimensions := flag.Int("dimensions", 64, "Dimension of vertex representation")
	undirected := flag.Bool("undirected", true, "Whether the edge is undirected")
	negativeSamples := flag.Int("negative_samples", 5, "Number of negative examples")
	windowSize := flag.Int("window_size", 5, "Size of skip-gram window")
	walkTimes := flag.Int("walk_times", 10, "Times of being starting vertex")
	walkSteps := flag.Int("walk_steps", 40, "Step of random walk")
	threads := flag.Int("threads", 1, "Number of training threads")
	alpha := flag.Float64("alpha", 0.025, "Init learning rate")

	flag.Usage = func() {
		fmt.Println("[SMORe-Go]")
		fmt.Println("\tGolang implementation of SMORe - DeepWalk")
		fmt.Println()
		fmt.Println("Options Description:")
		flag.PrintDefaults()
		fmt.Println()
		fmt.Println("Usage:")
		fmt.Println("./deepwalk -train net.txt -save rep.txt -undirected -dimensions 64 -walk_times 10 -walk_steps 40 -window_size 5 -negative_samples 5 -alpha 0.025 -threads 1")
	}

	flag.Parse()

	// Check required parameters
	if *train == "" || *save == "" {
		flag.Usage()
		os.Exit(1)
	}

	// Create and train model
	dw := deepwalk.New()

	if err := dw.LoadEdgeList(*train, *undirected); err != nil {
		fmt.Printf("Error loading edge list: %v\n", err)
		os.Exit(1)
	}

	dw.Init(*dimensions)
	dw.Train(*walkTimes, *walkSteps, *windowSize, *negativeSamples, *alpha, *threads)

	if err := dw.SaveWeights(*save); err != nil {
		fmt.Printf("Error saving weights: %v\n", err)
		os.Exit(1)
	}
}
