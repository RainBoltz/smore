package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/cnclabs/smore/internal/models/node2vec"
)

func main() {
	// Define command-line flags
	train := flag.String("train", "", "Train the Network data")
	save := flag.String("save", "", "Save the representation data")
	dimensions := flag.Int("dimensions", 64, "Dimension of vertex representation")
	undirected := flag.Bool("undirected", true, "Whether the edge is undirected")
	negativeSamples := flag.Int("negative_samples", 5, "Number of negative examples")
	windowSize := flag.Int("window_size", 10, "Size of skip-gram window")
	walkTimes := flag.Int("walk_times", 10, "Times of being starting vertex")
	walkSteps := flag.Int("walk_steps", 80, "Step of random walk")
	p := flag.Float64("p", 1.0, "Return parameter (controls likelihood to return to previous node)")
	q := flag.Float64("q", 1.0, "In-out parameter (BFS vs DFS: q > 1 = BFS, q < 1 = DFS)")
	threads := flag.Int("threads", 1, "Number of training threads")
	alpha := flag.Float64("alpha", 0.025, "Init learning rate")

	flag.Usage = func() {
		fmt.Println("[SMORe-Go]")
		fmt.Println("\tGolang implementation of SMORe - Node2Vec")
		fmt.Println()
		fmt.Println("Node2Vec extends DeepWalk with biased random walks:")
		fmt.Println("\t- p parameter: controls return probability (higher p = less likely to return)")
		fmt.Println("\t- q parameter: controls BFS vs DFS (q > 1 = BFS, q < 1 = DFS)")
		fmt.Println()
		fmt.Println("Common parameter combinations:")
		fmt.Println("\t- p=1, q=1: Unbiased (equivalent to DeepWalk)")
		fmt.Println("\t- p=1, q=0.5: DFS-like (explores outward, good for structural equivalence)")
		fmt.Println("\t- p=1, q=2: BFS-like (stays local, good for homophily)")
		fmt.Println()
		fmt.Println("Options Description:")
		flag.PrintDefaults()
		fmt.Println()
		fmt.Println("Usage:")
		fmt.Println("./node2vec -train net.txt -save rep.txt -undirected -dimensions 64 -p 1 -q 1 -walk_times 10 -walk_steps 80 -window_size 10 -negative_samples 5 -alpha 0.025 -threads 4")
		fmt.Println()
		fmt.Println("Examples:")
		fmt.Println("\t# Homophily (local structure)")
		fmt.Println("\t./node2vec -train net.txt -save rep.txt -p 1 -q 2")
		fmt.Println()
		fmt.Println("\t# Structural equivalence (global structure)")
		fmt.Println("\t./node2vec -train net.txt -save rep.txt -p 1 -q 0.5")
	}

	flag.Parse()

	// Check required parameters
	if *train == "" || *save == "" {
		flag.Usage()
		os.Exit(1)
	}

	// Validate parameters
	if *p <= 0 || *q <= 0 {
		fmt.Println("Error: p and q must be positive")
		os.Exit(1)
	}

	// Create and train model
	n2v := node2vec.New()

	if err := n2v.LoadEdgeList(*train, *undirected); err != nil {
		fmt.Printf("Error loading edge list: %v\n", err)
		os.Exit(1)
	}

	n2v.Init(*dimensions, *p, *q)
	n2v.Train(*walkTimes, *walkSteps, *windowSize, *negativeSamples, *alpha, *threads)

	// Compute and display homophily
	homophily := n2v.ComputeHomophily()
	fmt.Printf("\nHomophily ratio: %.4f\n", homophily)

	if err := n2v.SaveWeights(*save); err != nil {
		fmt.Printf("Error saving weights: %v\n", err)
		os.Exit(1)
	}
}
