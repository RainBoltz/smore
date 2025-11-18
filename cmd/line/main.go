package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/cnclabs/smore/internal/models/line"
)

func main() {
	// Define command-line flags
	train := flag.String("train", "", "Train the Network data")
	save := flag.String("save", "", "Save the representation data")
	dimensions := flag.Int("dimensions", 64, "Dimension of vertex representation")
	undirected := flag.Bool("undirected", true, "Whether the edge is undirected")
	negativeSamples := flag.Int("negative_samples", 5, "Number of negative examples")
	order := flag.Int("order", 2, "Order of proximity (1 or 2)")
	sampleTimes := flag.Int("sample_times", 10, "Number of training iterations")
	threads := flag.Int("threads", 1, "Number of training threads")
	alpha := flag.Float64("alpha", 0.025, "Init learning rate")

	flag.Usage = func() {
		fmt.Println("[SMORe-Go]")
		fmt.Println("\tGolang implementation of SMORe - LINE")
		fmt.Println()
		fmt.Println("Options Description:")
		flag.PrintDefaults()
		fmt.Println()
		fmt.Println("Usage:")
		fmt.Println("./line -train net.txt -save rep.txt -undirected -dimensions 64 -order 2 -sample_times 10 -negative_samples 5 -alpha 0.025 -threads 1")
	}

	flag.Parse()

	// Check required parameters
	if *train == "" || *save == "" {
		flag.Usage()
		os.Exit(1)
	}

	// Validate order
	var lineOrder line.Order
	if *order == 1 {
		lineOrder = line.First
	} else if *order == 2 {
		lineOrder = line.Second
	} else {
		fmt.Println("Error: order must be 1 or 2")
		os.Exit(1)
	}

	// Create and train model
	l := line.New()

	if err := l.LoadEdgeList(*train, *undirected); err != nil {
		fmt.Printf("Error loading edge list: %v\n", err)
		os.Exit(1)
	}

	l.Init(*dimensions, lineOrder)
	l.Train(*sampleTimes, *negativeSamples, *alpha, *threads)

	if err := l.SaveWeights(*save); err != nil {
		fmt.Printf("Error saving weights: %v\n", err)
		os.Exit(1)
	}
}
