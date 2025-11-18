package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/cnclabs/smore/internal/models/cpr"
)

func main() {
	trainTarget := flag.String("train_target", "", "Train the target domain network data")
	trainSource := flag.String("train_source", "", "Train the source domain network data")
	saveUser := flag.String("save_user", "", "Save the user embeddings")
	saveTarget := flag.String("save_target", "", "Save the target domain item embeddings")
	saveSource := flag.String("save_source", "", "Save the source domain item embeddings")
	dimensions := flag.Int("dimensions", 64, "Dimension of vertex representation")
	undirected := flag.Bool("undirected", true, "Whether the edges are undirected")
	updateTimes := flag.Int("update_times", 10, "Number of update iterations (in millions)")
	negativeSamples := flag.Int("negative_samples", 5, "Number of negative examples")
	alpha := flag.Float64("alpha", 0.1, "Initial learning rate")
	userReg := flag.Float64("user_reg", 0.01, "User embedding regularization")
	itemReg := flag.Float64("item_reg", 0.01, "Item embedding regularization")
	margin := flag.Float64("margin", 8.0, "Margin for BPR loss")
	threads := flag.Int("threads", 1, "Number of training threads")

	flag.Usage = func() {
		fmt.Println("[SMORe-Go]")
		fmt.Println("\tGolang implementation of SMORe - CPR")
		fmt.Println()
		fmt.Println("Description:")
		fmt.Println("\tCPR: Cross-Domain Preference Ranking with User Transformation")
		fmt.Println("\tLearns user preferences across two domains for improved recommendations")
		fmt.Println()
		fmt.Println("Options:")
		flag.PrintDefaults()
		fmt.Println()
		fmt.Println("Usage:")
		fmt.Println("  ./cpr -train_target target.txt -train_source source.txt \\")
		fmt.Println("        -save_user users.txt -save_target items_t.txt -save_source items_s.txt \\")
		fmt.Println("        -dimensions 64 -update_times 10 -alpha 0.1 -margin 8.0 -threads 4")
		fmt.Println()
		fmt.Println("Input Format:")
		fmt.Println("  Target domain: user_id item_id [weight]")
		fmt.Println("  Source domain: user_id item_id [weight]")
		fmt.Println("  (User IDs should be consistent across both domains)")
		fmt.Println()
		fmt.Println("Parameters:")
		fmt.Println("  margin:    BPR margin threshold (default: 8.0)")
		fmt.Println("  user_reg:  L2 regularization for user embeddings (default: 0.01)")
		fmt.Println("  item_reg:  L2 regularization for item embeddings (default: 0.01)")
	}

	flag.Parse()

	if *trainTarget == "" || *trainSource == "" {
		fmt.Println("Error: -train_target and -train_source are required")
		flag.Usage()
		os.Exit(1)
	}

	if *saveUser == "" && *saveTarget == "" && *saveSource == "" {
		fmt.Println("Error: at least one of -save_user, -save_target, or -save_source is required")
		flag.Usage()
		os.Exit(1)
	}

	// Create and initialize model
	model := cpr.New()

	// Load target domain graph
	fmt.Println("Loading target domain...")
	if err := model.LoadTargetDomain(*trainTarget, *undirected); err != nil {
		fmt.Printf("Error loading target domain: %v\n", err)
		os.Exit(1)
	}

	// Load source domain graph
	fmt.Println("Loading source domain...")
	if err := model.LoadSourceDomain(*trainSource, *undirected); err != nil {
		fmt.Printf("Error loading source domain: %v\n", err)
		os.Exit(1)
	}

	// Initialize embeddings
	model.Init(*dimensions)

	// Train model
	model.Train(*updateTimes, *negativeSamples, *alpha, *userReg, *itemReg, *margin, *threads)

	// Save embeddings
	if err := model.SaveWeights(*saveUser, *saveTarget, *saveSource); err != nil {
		fmt.Printf("Error saving weights: %v\n", err)
		os.Exit(1)
	}
}
