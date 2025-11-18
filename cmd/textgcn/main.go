package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/cnclabs/smore/internal/models/textgcn"
)

func main() {
	train := flag.String("train", "", "Train the Network data")
	save := flag.String("save", "", "Save the representation data")
	field := flag.String("field", "", "Field metadata file (vertex_name field_type)")
	dimensions := flag.Int("dimensions", 64, "Dimension of vertex representation")
	undirected := flag.Bool("undirected", true, "Whether the edge is undirected")
	negativeSamples := flag.Int("negative_samples", 5, "Number of negative examples")
	walkSteps := flag.Int("walk_steps", 5, "Step of aggregation/context window")
	sampleTimes := flag.Int("sample_times", 10, "Number of training samples (in millions)")
	reg := flag.Float64("reg", 0.01, "Regularization term")
	alpha := flag.Float64("alpha", 0.025, "Init learning rate")
	threads := flag.Int("threads", 1, "Number of training threads")

	flag.Usage = func() {
		fmt.Println("[SMORe-Go]")
		fmt.Println("\tGolang implementation of SMORe - TextGCN")
		fmt.Println()
		fmt.Println("Description:")
		fmt.Println("\tGraph Convolutional Networks for Text Classification")
		fmt.Println("\tBased on AAAI 2019 paper by Yao et al.")
		fmt.Println()
		fmt.Println("Options:")
		flag.PrintDefaults()
		fmt.Println()
		fmt.Println("Usage:")
		fmt.Println("  ./textgcn -train net.txt -field meta.txt -save rep.txt -dimensions 64 -sample_times 5 -walk_steps 5 -negative_samples 5 -alpha 0.025 -threads 1")
		fmt.Println()
		fmt.Println("Field File Format:")
		fmt.Println("  vertex_name field_type")
		fmt.Println("  Field types: 0=document, 1=filtered, 2=word")
	}

	flag.Parse()

	if *train == "" || *save == "" {
		fmt.Println("Error: -train and -save are required")
		flag.Usage()
		os.Exit(1)
	}

	// Create and initialize model
	model := textgcn.New()

	// Load graph
	fmt.Println("Loading graph...")
	if err := model.LoadEdgeList(*train, *undirected); err != nil {
		fmt.Printf("Error loading graph: %v\n", err)
		os.Exit(1)
	}

	// Load field metadata if provided
	if *field != "" {
		fmt.Println("Loading field metadata...")
		if err := model.LoadFieldMeta(*field); err != nil {
			fmt.Printf("Error loading field metadata: %v\n", err)
			os.Exit(1)
		}
	}

	// Initialize embeddings
	model.Init(*dimensions)

	// Train model
	model.Train(*sampleTimes, *walkSteps, *negativeSamples, *reg, *alpha, *threads)

	// Save embeddings
	if err := model.SaveWeights(*save); err != nil {
		fmt.Printf("Error saving weights: %v\n", err)
		os.Exit(1)
	}
}
