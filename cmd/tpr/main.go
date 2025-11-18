package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/cnclabs/smore/internal/models/tpr"
)

func main() {
	trainUI := flag.String("train_ui", "", "User-item interaction graph file")
	trainIW := flag.String("train_iw", "", "Item-word feature graph file")
	saveUser := flag.String("save_user", "", "Output file for user embeddings")
	saveItem := flag.String("save_item", "", "Output file for item embeddings")
	saveWord := flag.String("save_word", "", "Output file for word embeddings")
	dimensions := flag.Int("dimensions", 64, "Dimension of embeddings")
	undirected := flag.Bool("undirected", true, "Whether edges are undirected")
	sampleTimes := flag.Int("sample_times", 10, "Number of training epochs")
	alpha := flag.Float64("alpha", 0.025, "Initial learning rate")
	lambda := flag.Float64("lambda", 0.025, "L2 regularization parameter")
	textWeight := flag.Float64("text_weight", 0.5, "Weight for text component (0.0-1.0)")
	threads := flag.Int("threads", 1, "Number of training threads")

	flag.Usage = func() {
		fmt.Println("[SMORe-Go]")
		fmt.Println("\tGolang implementation of SMORe - TPR")
		fmt.Println()
		fmt.Println("Description:")
		fmt.Println("\tTPR: Text-aware Preference Ranking for Recommender Systems")
		fmt.Println("\tCombines collaborative filtering with text-based content features")
		fmt.Println()
		fmt.Println("Options:")
		flag.PrintDefaults()
		fmt.Println()
		fmt.Println("Usage:")
		fmt.Println("  ./tpr -train_ui user_item.txt -train_iw item_word.txt \\")
		fmt.Println("        -save_user users.txt -save_item items.txt -save_word words.txt \\")
		fmt.Println("        -dimensions 64 -sample_times 10 -text_weight 0.5 -threads 4")
		fmt.Println()
		fmt.Println("Input Format:")
		fmt.Println("  User-Item Graph:  user_id item_id [weight]")
		fmt.Println("  Item-Word Graph:  item_id word_id [weight]")
		fmt.Println()
		fmt.Println("Parameters:")
		fmt.Println("  text_weight:  Balance between collaborative (0.0) and content (1.0) signals")
		fmt.Println("  lambda:       L2 regularization strength (default: 0.025)")
		fmt.Println("  alpha:        Initial learning rate (default: 0.025)")
	}

	flag.Parse()

	if *trainUI == "" || *trainIW == "" {
		fmt.Println("Error: -train_ui and -train_iw are required")
		flag.Usage()
		os.Exit(1)
	}

	if *saveUser == "" && *saveItem == "" && *saveWord == "" {
		fmt.Println("Error: at least one of -save_user, -save_item, or -save_word is required")
		flag.Usage()
		os.Exit(1)
	}

	if *textWeight < 0.0 || *textWeight > 1.0 {
		fmt.Println("Error: text_weight must be between 0.0 and 1.0")
		os.Exit(1)
	}

	// Create and initialize model
	model := tpr.New()

	// Load user-item graph
	if err := model.LoadUserItemGraph(*trainUI, *undirected); err != nil {
		fmt.Printf("Error loading user-item graph: %v\n", err)
		os.Exit(1)
	}

	// Load item-word graph
	if err := model.LoadItemWordGraph(*trainIW, *undirected); err != nil {
		fmt.Printf("Error loading item-word graph: %v\n", err)
		os.Exit(1)
	}

	// Initialize embeddings
	model.Init(*dimensions, *textWeight)

	// Train model
	model.Train(*sampleTimes, *alpha, *lambda, *threads)

	// Save embeddings
	if err := model.SaveWeights(*saveUser, *saveItem, *saveWord); err != nil {
		fmt.Printf("Error saving weights: %v\n", err)
		os.Exit(1)
	}
}
