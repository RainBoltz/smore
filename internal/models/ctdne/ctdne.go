package ctdne

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/cnclabs/smore/pkg/pronet"
	"github.com/cnclabs/smore/pkg/temporal"
)

// CTDNE implements Continuous-Time Dynamic Network Embeddings
// Extends Node2Vec to handle time-evolving graphs with temporal random walks
type CTDNE struct {
	tg  *temporal.TemporalGraph
	dim int

	// Embeddings
	embeddings        [][]float64
	contextEmbeddings [][]float64

	// Temporal parameters
	timeWindow float64 // Time window for temporal walks (in time units)
}

// New creates a new CTDNE instance
func New() *CTDNE {
	return &CTDNE{
		tg: temporal.NewTemporalGraph(),
	}
}

// LoadEdgeList loads the temporal graph
func (ctdne *CTDNE) LoadEdgeList(filename string) error {
	return ctdne.tg.LoadEdgeList(filename)
}

// Init initializes the model
func (ctdne *CTDNE) Init(dim int, timeWindow float64) {
	ctdne.dim = dim
	ctdne.timeWindow = timeWindow

	// If timeWindow not specified, use a reasonable default (10% of time span)
	if ctdne.timeWindow <= 0 {
		timeSpan := ctdne.tg.MaxTime - ctdne.tg.MinTime
		ctdne.timeWindow = timeSpan * 0.1
		fmt.Printf("Auto-set time window: %.2f (10%% of time span)\n", ctdne.timeWindow)
	}

	fmt.Println("Model Setting:")
	fmt.Printf("\tdimension:\t\t%d\n", dim)
	fmt.Printf("\ttime_window:\t\t%.2f\n", ctdne.timeWindow)
	fmt.Println()

	fmt.Println("CTDNE Principle:")
	fmt.Println("\t✓ Temporal random walks (respects edge timestamps)")
	fmt.Println("\t✓ Time-constrained neighbor sampling")
	fmt.Println("\t✓ Captures temporal dynamics of network evolution")

	// Initialize embeddings
	ctdne.embeddings = make([][]float64, ctdne.tg.NumNodes)
	for i := int64(0); i < ctdne.tg.NumNodes; i++ {
		ctdne.embeddings[i] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			ctdne.embeddings[i][d] = (rand.Float64() - 0.5) / float64(dim)
		}
	}

	// Initialize context embeddings
	ctdne.contextEmbeddings = make([][]float64, ctdne.tg.NumNodes)
	for i := int64(0); i < ctdne.tg.NumNodes; i++ {
		ctdne.contextEmbeddings[i] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			ctdne.contextEmbeddings[i][d] = (rand.Float64() - 0.5) / float64(dim)
		}
	}
}

// Train trains the CTDNE model
func (ctdne *CTDNE) Train(walkTimes, walkSteps, windowSize, negativeSamples int, alpha float64, workers int) {
	fmt.Println()
	fmt.Println("Model:")
	fmt.Println("\t[CTDNE - Continuous-Time Dynamic Network Embeddings]")
	fmt.Println()

	fmt.Println("Learning Parameters:")
	fmt.Printf("\twalk_times:\t\t%d\n", walkTimes)
	fmt.Printf("\twalk_steps:\t\t%d\n", walkSteps)
	fmt.Printf("\twindow_size:\t\t%d\n", windowSize)
	fmt.Printf("\tnegative_samples:\t%d\n", negativeSamples)
	fmt.Printf("\talpha:\t\t\t%.6f\n", alpha)
	fmt.Printf("\tworkers:\t\t%d\n", workers)
	fmt.Println()

	fmt.Println("Start Training:")

	total := int64(walkTimes) * ctdne.tg.NumNodes
	alphaMin := alpha * 0.0001
	currentAlpha := alpha
	count := int64(0)

	var countMu sync.Mutex

	// Create ProNet instance for skip-gram training
	pnet := pronet.NewProNet()
	pnet.MaxVid = ctdne.tg.NumNodes

	// Build negative sampling alias table
	negDistribution := make([]float64, ctdne.tg.NumNodes)
	for i := int64(0); i < ctdne.tg.NumNodes; i++ {
		// Weight by node activity (number of edges)
		activity := ctdne.tg.GetNodeActivity(i)
		if activity > 0 {
			negDistribution[i] = float64(activity)
		} else {
			negDistribution[i] = 1.0
		}
	}
	pnet.NegativeAT = pronet.BuildAliasMethod(negDistribution, 0.75)

	for t := 0; t < walkTimes; t++ {
		// Shuffle nodes for random access
		randomKeys := make([]int64, ctdne.tg.NumNodes)
		for i := int64(0); i < ctdne.tg.NumNodes; i++ {
			randomKeys[i] = i
		}
		// Fisher-Yates shuffle
		for i := int64(0); i < ctdne.tg.NumNodes; i++ {
			j := i + rand.Int63n(ctdne.tg.NumNodes-i)
			randomKeys[i], randomKeys[j] = randomKeys[j], randomKeys[i]
		}

		// Parallel training
		var wg sync.WaitGroup
		chunkSize := (int(ctdne.tg.NumNodes) + workers - 1) / workers

		for w := 0; w < workers; w++ {
			wg.Add(1)
			start := w * chunkSize
			end := start + chunkSize
			if end > int(ctdne.tg.NumNodes) {
				end = int(ctdne.tg.NumNodes)
			}

			go func(start, end int) {
				defer wg.Done()

				rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(start)))

				for i := start; i < end; i++ {
					nodeID := randomKeys[i]

					// Get node's active time range
					minTime, maxTime := ctdne.tg.GetActiveTimeRange(nodeID)
					if minTime == 0 && maxTime == 0 {
						// Node has no edges
						countMu.Lock()
						count++
						countMu.Unlock()
						continue
					}

					// Sample a random starting time for the walk
					timeRange := maxTime - minTime
					if timeRange == 0 {
						timeRange = ctdne.timeWindow
					}
					startTime := minTime + rng.Float64()*timeRange

					// Perform temporal random walk
					walk := ctdne.tg.TemporalRandomWalk(nodeID, startTime, walkSteps, ctdne.timeWindow, rng)

					if len(walk) < 2 {
						countMu.Lock()
						count++
						countMu.Unlock()
						continue
					}

					// Generate skip-gram pairs
					vertices, contexts := pnet.SkipGrams(walk, windowSize)

					// Update embeddings
					pnet.UpdatePairs(ctdne.embeddings, ctdne.contextEmbeddings, vertices, contexts, ctdne.dim, negativeSamples, currentAlpha, rng)

					// Update progress
					countMu.Lock()
					count++
					if count%pronet.Monitor == 0 {
						currentAlpha = alpha * (1.0 - float64(count)/float64(total))
						if currentAlpha < alphaMin {
							currentAlpha = alphaMin
						}
						fmt.Printf("\tAlpha: %.6f\tProgress: %.3f %%\r", currentAlpha, float64(count)/float64(total)*100)
					}
					countMu.Unlock()
				}
			}(start, end)
		}

		wg.Wait()
	}

	fmt.Printf("\tAlpha: %.6f\tProgress: 100.00 %%\n", currentAlpha)
	fmt.Println("\nTraining Complete!")
}

// SaveEmbeddings saves the learned embeddings
func (ctdne *CTDNE) SaveEmbeddings(filename string) error {
	fmt.Println("Save Model:")

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer file.Close()

	// Write header: num_nodes dimension
	fmt.Fprintf(file, "%d %d\n", ctdne.tg.NumNodes, ctdne.dim)

	// Write embeddings
	for i := int64(0); i < ctdne.tg.NumNodes; i++ {
		name := ctdne.tg.GetNodeName(i)
		fmt.Fprintf(file, "%s", name)

		for d := 0; d < ctdne.dim; d++ {
			fmt.Fprintf(file, " %.6f", ctdne.embeddings[i][d])
		}
		fmt.Fprintln(file)
	}

	fmt.Printf("\tSave to <%s>\n", filename)
	return nil
}

// GetEmbedding returns the embedding for a node
func (ctdne *CTDNE) GetEmbedding(nodeID int64) []float64 {
	if nodeID < 0 || nodeID >= ctdne.tg.NumNodes {
		return nil
	}
	return ctdne.embeddings[nodeID]
}

// ComputeTemporalCoherence measures how well embeddings capture temporal proximity
// Nodes that interact closer in time should have more similar embeddings
func (ctdne *CTDNE) ComputeTemporalCoherence() float64 {
	fmt.Println("\nTemporal Coherence Analysis:")

	samples := 1000
	if samples > int(ctdne.tg.NumEdges) {
		samples = int(ctdne.tg.NumEdges)
	}

	totalSimilarity := 0.0
	validSamples := 0

	for i := 0; i < samples; i++ {
		// Sample a random edge
		edgeIdx := rand.Intn(len(ctdne.tg.Edges))
		edge := ctdne.tg.Edges[edgeIdx]

		// Compute cosine similarity
		sim := cosineSimilarity(ctdne.embeddings[edge.From], ctdne.embeddings[edge.To])
		if !isNaN(sim) {
			totalSimilarity += sim
			validSamples++
		}
	}

	if validSamples == 0 {
		return 0.0
	}

	avgSimilarity := totalSimilarity / float64(validSamples)
	fmt.Printf("\tAverage similarity of temporally connected nodes: %.4f\n", avgSimilarity)

	return avgSimilarity
}

// cosineSimilarity computes cosine similarity
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0.0
	}

	dotProduct := 0.0
	normA := 0.0
	normB := 0.0

	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0.0
	}

	return dotProduct / math.Sqrt(normA*normB)
}

// isNaN checks if a float64 is NaN
func isNaN(f float64) bool {
	return f != f
}
