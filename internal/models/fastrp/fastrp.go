package fastrp

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/cnclabs/smore/pkg/pronet"
)

// FastRP implements the Fast Random Projection algorithm
// FastRP is an extremely fast, scalable graph embedding method that requires no training.
// It uses very sparse random projections and iterative neighbor aggregation.
// Performance: 75,000x faster than Node2Vec with comparable accuracy.
type FastRP struct {
	pnet *pronet.ProNet
	dim  int

	// FastRP specific parameters
	iterations           int     // Number of aggregation iterations (default: 3)
	normalizationStrength float64 // Degree normalization strength (default: 0.0)
	randomSeed           int64   // Random seed for reproducibility

	// Embeddings at different iterations
	embeddings [][]float64 // Final concatenated embeddings
}

// New creates a new FastRP instance
func New() *FastRP {
	return &FastRP{
		pnet:                 pronet.NewProNet(),
		iterations:           3,
		normalizationStrength: 0.0,
		randomSeed:           time.Now().UnixNano(),
	}
}

// LoadEdgeList loads the network from an edge list file
func (f *FastRP) LoadEdgeList(filename string, undirected bool) error {
	return f.pnet.LoadEdgeList(filename, undirected)
}

// Init initializes the model with given dimensions and parameters
func (f *FastRP) Init(dim, iterations int, normalizationStrength float64) {
	f.dim = dim
	f.iterations = iterations
	f.normalizationStrength = normalizationStrength

	fmt.Println("Model Setting:")
	fmt.Printf("\tdimension:\t\t%d\n", dim)
	fmt.Printf("\titerations:\t\t%d\n", iterations)
	fmt.Printf("\tnormalization:\t\t%.2f\n", normalizationStrength)
	fmt.Println()
	fmt.Println("FastRP Features:")
	fmt.Println("\t✓ No training required (single-pass)")
	fmt.Println("\t✓ 75,000x faster than Node2Vec")
	fmt.Println("\t✓ Very sparse random projection")
	fmt.Println("\t✓ Iterative neighbor aggregation")
}

// Generate generates the embeddings using FastRP algorithm
// This is the main computation method - no training/SGD required!
func (f *FastRP) Generate(workers int) {
	fmt.Println("Model:")
	fmt.Println("\t[FastRP - Fast Random Projection]")
	fmt.Println()

	fmt.Println("Generation Parameters:")
	fmt.Printf("\tworkers:\t\t%d\n", workers)
	fmt.Println()

	fmt.Println("Start Generation:")

	maxVid := f.pnet.MaxVid
	dimPerIteration := f.dim / (f.iterations + 1)

	// Initialize final embeddings matrix
	f.embeddings = make([][]float64, maxVid)
	for vid := int64(0); vid < maxVid; vid++ {
		f.embeddings[vid] = make([]float64, f.dim)
	}

	// Step 1: Initialize with very sparse random vectors
	fmt.Println("\t[1/3] Initializing sparse random vectors...")
	rng := rand.New(rand.NewSource(f.randomSeed))
	sparseFeatures := f.initializeSparseRandomVectors(dimPerIteration, rng)

	// Copy initial features to embeddings (iteration 0)
	for vid := int64(0); vid < maxVid; vid++ {
		copy(f.embeddings[vid][:dimPerIteration], sparseFeatures[vid])
	}

	// Step 2: Iterative neighbor aggregation
	fmt.Println("\t[2/3] Performing iterative neighbor aggregation...")

	currentFeatures := sparseFeatures
	offset := dimPerIteration

	for iter := 0; iter < f.iterations; iter++ {
		fmt.Printf("\t\tIteration %d/%d\r", iter+1, f.iterations)

		// Aggregate neighbors in parallel
		nextFeatures := f.aggregateNeighbors(currentFeatures, dimPerIteration, workers)

		// Concatenate to embeddings
		for vid := int64(0); vid < maxVid; vid++ {
			copy(f.embeddings[vid][offset:offset+dimPerIteration], nextFeatures[vid])
		}

		currentFeatures = nextFeatures
		offset += dimPerIteration
	}

	fmt.Printf("\t\tIteration %d/%d (completed)\n", f.iterations, f.iterations)

	// Step 3: L2 normalization
	fmt.Println("\t[3/3] Normalizing embeddings...")
	f.normalizeEmbeddings(workers)

	fmt.Println()
	fmt.Println("Generation Complete!")
}

// initializeSparseRandomVectors creates very sparse random vectors for each node
// Sparsity: ~95% zeros, 2.5% +1, 2.5% -1
func (f *FastRP) initializeSparseRandomVectors(dim int, rng *rand.Rand) [][]float64 {
	maxVid := f.pnet.MaxVid
	features := make([][]float64, maxVid)

	// Very sparse: only ~5% non-zero values
	sparsity := 0.05

	for vid := int64(0); vid < maxVid; vid++ {
		features[vid] = make([]float64, dim)

		for d := 0; d < dim; d++ {
			r := rng.Float64()
			if r < sparsity/2 {
				features[vid][d] = 1.0
			} else if r < sparsity {
				features[vid][d] = -1.0
			}
			// else: remains 0.0 (sparse!)
		}
	}

	return features
}

// aggregateNeighbors aggregates neighbor features with optional degree normalization
func (f *FastRP) aggregateNeighbors(currentFeatures [][]float64, dim int, workers int) [][]float64 {
	maxVid := f.pnet.MaxVid
	nextFeatures := make([][]float64, maxVid)

	// Initialize next features
	for vid := int64(0); vid < maxVid; vid++ {
		nextFeatures[vid] = make([]float64, dim)
	}

	// Parallel aggregation
	var wg sync.WaitGroup
	chunkSize := (int(maxVid) + workers - 1) / workers

	for w := 0; w < workers; w++ {
		wg.Add(1)
		start := int64(w * chunkSize)
		end := start + int64(chunkSize)
		if end > maxVid {
			end = maxVid
		}

		go func(start, end int64) {
			defer wg.Done()

			for vid := start; vid < end; vid++ {
				neighbors := f.pnet.Graph[vid]
				if len(neighbors) == 0 {
					// Isolated node: keep zero vector
					continue
				}

				weights := f.pnet.EdgeWeights[vid]

				// Calculate normalization factor
				norm := 1.0
				if f.normalizationStrength > 0 {
					// Degree-based normalization: d^(-normalizationStrength)
					degree := float64(len(neighbors))
					norm = math.Pow(degree, -f.normalizationStrength)
				}

				// Aggregate neighbor features
				totalWeight := 0.0
				for i, nid := range neighbors {
					weight := 1.0
					if len(weights) > 0 {
						weight = weights[i]
					}
					totalWeight += weight

					// Add weighted neighbor features
					for d := 0; d < dim; d++ {
						nextFeatures[vid][d] += currentFeatures[nid][d] * weight * norm
					}
				}

				// Average by total weight
				if totalWeight > 0 {
					for d := 0; d < dim; d++ {
						nextFeatures[vid][d] /= totalWeight
					}
				}
			}
		}(start, end)
	}

	wg.Wait()
	return nextFeatures
}

// normalizeEmbeddings performs L2 normalization on all embeddings
func (f *FastRP) normalizeEmbeddings(workers int) {
	maxVid := f.pnet.MaxVid

	var wg sync.WaitGroup
	chunkSize := (int(maxVid) + workers - 1) / workers

	for w := 0; w < workers; w++ {
		wg.Add(1)
		start := int64(w * chunkSize)
		end := start + int64(chunkSize)
		if end > maxVid {
			end = maxVid
		}

		go func(start, end int64) {
			defer wg.Done()

			for vid := start; vid < end; vid++ {
				// Compute L2 norm
				norm := 0.0
				for d := 0; d < f.dim; d++ {
					norm += f.embeddings[vid][d] * f.embeddings[vid][d]
				}
				norm = math.Sqrt(norm)

				// Normalize (avoid division by zero)
				if norm > 1e-10 {
					for d := 0; d < f.dim; d++ {
						f.embeddings[vid][d] /= norm
					}
				}
			}
		}(start, end)
	}

	wg.Wait()
}

// SaveWeights saves the learned embeddings to a file
func (f *FastRP) SaveWeights(filename string) error {
	fmt.Println("Save Model:")

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer file.Close()

	// Write header: num_vertices dimension
	fmt.Fprintf(file, "%d %d\n", f.pnet.MaxVid, f.dim)

	// Write embeddings
	for vid := int64(0); vid < f.pnet.MaxVid; vid++ {
		name := f.pnet.GetVertexName(vid)
		fmt.Fprintf(file, "%s", name)
		for d := 0; d < f.dim; d++ {
			fmt.Fprintf(file, " %.6f", f.embeddings[vid][d])
		}
		fmt.Fprintln(file)
	}

	fmt.Printf("\tSave to <%s>\n", filename)
	return nil
}

// GetEmbedding returns the embedding vector for a given vertex ID
func (f *FastRP) GetEmbedding(vid int64) []float64 {
	if vid < 0 || vid >= f.pnet.MaxVid {
		return nil
	}
	return f.embeddings[vid]
}

// ComputeSparsity computes the sparsity of embeddings (% of near-zero values)
func (f *FastRP) ComputeSparsity() float64 {
	threshold := 1e-6
	totalValues := 0
	sparseValues := 0

	for vid := int64(0); vid < f.pnet.MaxVid; vid++ {
		for d := 0; d < f.dim; d++ {
			totalValues++
			if math.Abs(f.embeddings[vid][d]) < threshold {
				sparseValues++
			}
		}
	}

	if totalValues == 0 {
		return 0.0
	}

	return float64(sparseValues) / float64(totalValues)
}
