package deepwalk

import (
	"fmt"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/cnclabs/smore/pkg/pronet"
)

// DeepWalk implements the DeepWalk algorithm
type DeepWalk struct {
	pnet     *pronet.ProNet
	dim      int
	wVertex  [][]float64
	wContext [][]float64
}

// New creates a new DeepWalk instance
func New() *DeepWalk {
	return &DeepWalk{
		pnet: pronet.NewProNet(),
	}
}

// LoadEdgeList loads the network from an edge list file
func (dw *DeepWalk) LoadEdgeList(filename string, undirected bool) error {
	return dw.pnet.LoadEdgeList(filename, undirected)
}

// Init initializes the model with given dimensions
func (dw *DeepWalk) Init(dim int) {
	dw.dim = dim
	maxVid := dw.pnet.MaxVid

	fmt.Println("Model Setting:")
	fmt.Printf("\tdimension:\t\t%d\n", dim)

	// Initialize vertex embeddings
	dw.wVertex = make([][]float64, maxVid)
	for vid := int64(0); vid < maxVid; vid++ {
		dw.wVertex[vid] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			dw.wVertex[vid][d] = (rand.Float64() - 0.5) / float64(dim)
		}
	}

	// Initialize context embeddings
	dw.wContext = make([][]float64, maxVid)
	for vid := int64(0); vid < maxVid; vid++ {
		dw.wContext[vid] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			dw.wContext[vid][d] = (rand.Float64() - 0.5) / float64(dim)
		}
	}
}

// Train trains the DeepWalk model
func (dw *DeepWalk) Train(walkTimes, walkSteps, windowSize, negativeSamples int, alpha float64, workers int) {
	fmt.Println("Model:")
	fmt.Println("\t[DeepWalk]")

	fmt.Println("Learning Parameters:")
	fmt.Printf("\twalk_times:\t\t%d\n", walkTimes)
	fmt.Printf("\twalk_steps:\t\t%d\n", walkSteps)
	fmt.Printf("\twindow_size:\t\t%d\n", windowSize)
	fmt.Printf("\tnegative_samples:\t%d\n", negativeSamples)
	fmt.Printf("\talpha:\t\t\t%.6f\n", alpha)
	fmt.Printf("\tworkers:\t\t%d\n", workers)

	fmt.Println("Start Training:")

	total := int64(walkTimes) * dw.pnet.MaxVid
	alphaMin := alpha * 0.0001
	currentAlpha := alpha
	count := int64(0)

	var countMu sync.Mutex

	for t := 0; t < walkTimes; t++ {
		// Shuffle vertices for random access
		randomKeys := make([]int64, dw.pnet.MaxVid)
		for vid := int64(0); vid < dw.pnet.MaxVid; vid++ {
			randomKeys[vid] = vid
		}
		// Fisher-Yates shuffle
		for vid := int64(0); vid < dw.pnet.MaxVid; vid++ {
			j := vid + rand.Int63n(dw.pnet.MaxVid-vid)
			randomKeys[vid], randomKeys[j] = randomKeys[j], randomKeys[vid]
		}

		// Parallel training using goroutines
		var wg sync.WaitGroup
		chunkSize := (int(dw.pnet.MaxVid) + workers - 1) / workers

		for w := 0; w < workers; w++ {
			wg.Add(1)
			start := w * chunkSize
			end := start + chunkSize
			if end > int(dw.pnet.MaxVid) {
				end = int(dw.pnet.MaxVid)
			}

			go func(start, end int) {
				defer wg.Done()

				// Each worker has its own RNG for thread safety
				rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(start)))

				for vid := start; vid < end; vid++ {
					// Perform random walk
					walks := dw.pnet.RandomWalk(randomKeys[vid], walkSteps, rng)

					// Generate skip-gram pairs
					vertices, contexts := dw.pnet.SkipGrams(walks, windowSize)

					// Update embeddings
					dw.pnet.UpdatePairs(dw.wVertex, dw.wContext, vertices, contexts, dw.dim, negativeSamples, currentAlpha, rng)

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
}

// SaveWeights saves the learned embeddings to a file
func (dw *DeepWalk) SaveWeights(filename string) error {
	fmt.Println("Save Model:")

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer file.Close()

	// Write header: num_vertices dimension
	fmt.Fprintf(file, "%d %d\n", dw.pnet.MaxVid, dw.dim)

	// Write embeddings
	for vid := int64(0); vid < dw.pnet.MaxVid; vid++ {
		name := dw.pnet.GetVertexName(vid)
		fmt.Fprintf(file, "%s", name)
		for d := 0; d < dw.dim; d++ {
			fmt.Fprintf(file, " %.6f", dw.wVertex[vid][d])
		}
		fmt.Fprintln(file)
	}

	fmt.Printf("\tSave to <%s>\n", filename)
	return nil
}
