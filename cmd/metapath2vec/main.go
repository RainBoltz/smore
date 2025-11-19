package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/cnclabs/smore/internal/models/metapath2vec"
)

func main() {
	// Define command-line flags
	train := flag.String("train", "", "Train on heterogeneous graph")
	save := flag.String("save", "", "Save embeddings")
	dimensions := flag.Int("dimensions", 128, "Dimension of embeddings")
	undirected := flag.Bool("undirected", true, "Whether edges are undirected")
	metaPaths := flag.String("metapaths", "", "Meta-paths (comma-separated), e.g., \"User Item User,User Item Category Item User\"")
	negativeSamples := flag.Int("negative_samples", 5, "Number of negative samples")
	windowSize := flag.Int("window_size", 10, "Size of skip-gram window")
	walkTimes := flag.Int("walk_times", 10, "Times of being starting vertex")
	walkSteps := flag.Int("walk_steps", 80, "Steps of meta-path walk")
	threads := flag.Int("threads", 4, "Number of training threads")
	alpha := flag.Float64("alpha", 0.025, "Initial learning rate")

	flag.Usage = func() {
		fmt.Println("[SMORe-Go]")
		fmt.Println("\tGolang implementation of SMORe - Metapath2Vec")
		fmt.Println()
		fmt.Println("Metapath2Vec - Heterogeneous Graph Embedding:")
		fmt.Println("\t✓ Handles graphs with multiple node/edge types")
		fmt.Println("\t✓ Meta-path-guided random walks")
		fmt.Println("\t✓ Preserves type-specific relationships")
		fmt.Println("\t✓ Perfect for realistic multi-type networks")
		fmt.Println()
		fmt.Println("How it works:")
		fmt.Println("\t1. Define meta-paths (sequences of node types)")
		fmt.Println("\t2. Perform meta-path-guided random walks")
		fmt.Println("\t3. Train embeddings using skip-gram")
		fmt.Println("\t4. Learn type-aware representations")
		fmt.Println()
		fmt.Println("Input format (heterogeneous edges):")
		fmt.Println("\tsource_node source_type target_node target_type edge_type [weight]")
		fmt.Println()
		fmt.Println("Examples:")
		fmt.Println("\t# E-commerce network")
		fmt.Println("\tAlice User iPhone Product purchased 1.0")
		fmt.Println("\tiPhone Product Electronics Category belongs_to 1.0")
		fmt.Println("\tBob User iPhone Product purchased 1.0")
		fmt.Println()
		fmt.Println("\t# Academic network")
		fmt.Println("\tAlice Author Paper1 Paper wrote 1.0")
		fmt.Println("\tPaper1 Paper KDD Venue published_at 1.0")
		fmt.Println("\tBob Author Paper1 Paper wrote 1.0")
		fmt.Println()
		fmt.Println("Meta-paths:")
		fmt.Println("\tDefine walking patterns through typed nodes")
		fmt.Println("\tFormat: Space-separated node types")
		fmt.Println()
		fmt.Println("Meta-path Examples:")
		fmt.Println("\tE-commerce:")
		fmt.Println("\t  • \"User Item User\" - users who bought same items")
		fmt.Println("\t  • \"User Item Category Item User\" - category-based similarity")
		fmt.Println()
		fmt.Println("\tAcademic:")
		fmt.Println("\t  • \"Author Paper Author\" - co-authors")
		fmt.Println("\t  • \"Author Paper Venue Paper Author\" - same venue authors")
		fmt.Println()
		fmt.Println("\tSocial:")
		fmt.Println("\t  • \"User Post User\" - users who post on same topics")
		fmt.Println("\t  • \"User Group User\" - users in same groups")
		fmt.Println()
		fmt.Println("Key parameters:")
		fmt.Println("\t- metapaths: Comma-separated meta-paths (REQUIRED)")
		fmt.Println("\t  Example: \"User Item User,User Item Category Item User\"")
		fmt.Println("\t- dimensions: Embedding dimension (default: 128)")
		fmt.Println("\t- walk_times: Number of walks per node (default: 10)")
		fmt.Println("\t- walk_steps: Steps per walk (default: 80)")
		fmt.Println()
		fmt.Println("Options Description:")
		flag.PrintDefaults()
		fmt.Println()
		fmt.Println("Usage:")
		fmt.Println("./metapath2vec -train graph.txt -save embeddings.txt \\")
		fmt.Println("               -metapaths \"User Item User,User Item Category Item User\" \\")
		fmt.Println("               -dimensions 128 -walk_times 10 -walk_steps 80 \\")
		fmt.Println("               -window_size 10 -negative_samples 5 \\")
		fmt.Println("               -alpha 0.025 -threads 4")
		fmt.Println()
		fmt.Println("Examples:")
		fmt.Println("\t# E-commerce recommendation")
		fmt.Println("\t./metapath2vec -train ecommerce.txt -save emb.txt \\")
		fmt.Println("\t               -metapaths \"User Item User,User Item Category Item User\"")
		fmt.Println()
		fmt.Println("\t# Academic network")
		fmt.Println("\t./metapath2vec -train dblp.txt -save emb.txt \\")
		fmt.Println("\t               -metapaths \"Author Paper Author,Author Paper Venue Paper Author\"")
		fmt.Println()
		fmt.Println("\t# Movie recommendations")
		fmt.Println("\t./metapath2vec -train movies.txt -save emb.txt \\")
		fmt.Println("\t               -metapaths \"User Movie User,User Movie Genre Movie User\"")
		fmt.Println()
		fmt.Println("Use Cases:")
		fmt.Println("\t✓ E-commerce: User-Item-Category networks")
		fmt.Println("\t✓ Academic: Author-Paper-Venue networks")
		fmt.Println("\t✓ Healthcare: Patient-Treatment-Disease networks")
		fmt.Println("\t✓ Social: User-Post-Topic networks")
		fmt.Println("\t✓ Movies: User-Movie-Genre-Actor networks")
		fmt.Println()
		fmt.Println("Datasets:")
		fmt.Println("\t• DBLP (academic)")
		fmt.Println("\t• Yelp (business reviews)")
		fmt.Println("\t• IMDB (movies)")
		fmt.Println("\t• Amazon (e-commerce)")
	}

	flag.Parse()

	// Check required parameters
	if *train == "" || *save == "" {
		flag.Usage()
		os.Exit(1)
	}

	if *metaPaths == "" {
		fmt.Println("Error: -metapaths is required")
		fmt.Println()
		fmt.Println("Example meta-paths:")
		fmt.Println("  E-commerce: \"User Item User,User Item Category Item User\"")
		fmt.Println("  Academic:   \"Author Paper Author,Author Paper Venue Paper Author\"")
		fmt.Println("  Social:     \"User Post User,User Group User\"")
		os.Exit(1)
	}

	// Validate parameters
	if *dimensions <= 0 {
		fmt.Println("Error: dimensions must be positive")
		os.Exit(1)
	}
	if *walkTimes <= 0 {
		fmt.Println("Error: walk_times must be positive")
		os.Exit(1)
	}
	if *walkSteps <= 0 {
		fmt.Println("Error: walk_steps must be positive")
		os.Exit(1)
	}
	if *threads <= 0 {
		fmt.Println("Error: threads must be positive")
		os.Exit(1)
	}

	fmt.Println("═══════════════════════════════════════════════════════")
	fmt.Println("  Metapath2Vec - Heterogeneous Graph Embedding")
	fmt.Println("═══════════════════════════════════════════════════════")
	fmt.Println()

	startTime := time.Now()

	// Create model
	mp := metapath2vec.New()

	// Load graph
	fmt.Println("Loading heterogeneous graph...")
	if err := mp.LoadEdgeList(*train, *undirected); err != nil {
		fmt.Printf("Error loading graph: %v\n", err)
		os.Exit(1)
	}

	loadTime := time.Since(startTime)
	fmt.Printf("Graph loaded in %.2f seconds\n", loadTime.Seconds())
	fmt.Println()

	// Parse and add meta-paths
	fmt.Println("Parsing meta-paths...")
	pathList := strings.Split(*metaPaths, ",")
	for _, path := range pathList {
		path = strings.TrimSpace(path)
		if path == "" {
			continue
		}
		if err := mp.AddMetaPath(path); err != nil {
			fmt.Printf("Error adding meta-path: %v\n", err)
			os.Exit(1)
		}
	}
	fmt.Println()

	// Initialize
	mp.Init(*dimensions)

	// Train
	trainStartTime := time.Now()
	mp.Train(*walkTimes, *walkSteps, *windowSize, *negativeSamples, *alpha, *threads)
	trainTime := time.Since(trainStartTime)

	// Compute type homogeneity
	fmt.Println()
	fmt.Println("Computing type homogeneity...")
	homogeneity := mp.ComputeTypeHomogeneity()

	// Save
	fmt.Println()
	if err := mp.SaveEmbeddings(*save); err != nil {
		fmt.Printf("Error saving embeddings: %v\n", err)
		os.Exit(1)
	}

	totalTime := time.Since(startTime)
	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════")
	fmt.Println("  Statistics")
	fmt.Println("═══════════════════════════════════════════════════════")
	fmt.Println("Type Homogeneity (same-type similarity):")
	for typeName, score := range homogeneity {
		fmt.Printf("\t%s: %.4f\n", typeName, score)
	}
	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════")
	fmt.Println("  Timing Summary")
	fmt.Println("═══════════════════════════════════════════════════════")
	fmt.Printf("Loading time:     %.2f seconds\n", loadTime.Seconds())
	fmt.Printf("Training time:    %.2f seconds\n", trainTime.Seconds())
	fmt.Printf("Total time:       %.2f seconds\n", totalTime.Seconds())
	fmt.Println()
	fmt.Println("✓ Metapath2Vec training complete!")
	fmt.Println()
	fmt.Println("Next steps:")
	fmt.Println("\t• Use embeddings for type-aware recommendations")
	fmt.Println("\t• Try different meta-path combinations")
	fmt.Println("\t• Evaluate on link prediction tasks")
	fmt.Println("\t• Analyze type-specific clustering")
}
