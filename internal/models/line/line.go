package line

import (
	"fmt"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/cnclabs/smore/pkg/pronet"
)

// Order specifies LINE order (1st or 2nd)
type Order int

const (
	First  Order = 1
	Second Order = 2
)

// LINE implements the LINE (Large-scale Information Network Embedding) algorithm
type LINE struct {
	pnet     *pronet.ProNet
	dim      int
	order    Order
	wVertex  [][]float64
	wContext [][]float64
}

// New creates a new LINE instance
func New() *LINE {
	return &LINE{
		pnet:  pronet.NewProNet(),
		order: Second, // Default to 2nd order
	}
}

// LoadEdgeList loads the network from an edge list file
func (l *LINE) LoadEdgeList(filename string, undirected bool) error {
	return l.pnet.LoadEdgeList(filename, undirected)
}

// Init initializes the model with given dimensions and order
func (l *LINE) Init(dim int, order Order) {
	l.dim = dim
	l.order = order
	maxVid := l.pnet.MaxVid

	fmt.Println("Model Setting:")
	fmt.Printf("\tdimension:\t\t%d\n", dim)
	fmt.Printf("\torder:\t\t\t%d\n", order)

	// Initialize vertex embeddings
	l.wVertex = make([][]float64, maxVid)
	for vid := int64(0); vid < maxVid; vid++ {
		l.wVertex[vid] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			l.wVertex[vid][d] = (rand.Float64() - 0.5) / float64(dim)
		}
	}

	// Initialize context embeddings (for 2nd order proximity)
	l.wContext = make([][]float64, maxVid)
	for vid := int64(0); vid < maxVid; vid++ {
		l.wContext[vid] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			l.wContext[vid][d] = (rand.Float64() - 0.5) / float64(dim)
		}
	}
}

// Train trains the LINE model
func (l *LINE) Train(sampleTimes, negativeSamples int, alpha float64, workers int) {
	fmt.Println("Model:")
	fmt.Println("\t[LINE]")

	fmt.Println("Learning Parameters:")
	fmt.Printf("\tsample_times:\t\t%d\n", sampleTimes)
	fmt.Printf("\tnegative_samples:\t%d\n", negativeSamples)
	fmt.Printf("\talpha:\t\t\t%.6f\n", alpha)
	fmt.Printf("\tworkers:\t\t%d\n", workers)

	fmt.Println("Start Training:")

	total := int64(sampleTimes) * l.pnet.MaxLine
	alphaMin := alpha * 0.0001
	currentAlpha := alpha
	count := int64(0)

	var countMu sync.Mutex

	// Parallel training
	var wg sync.WaitGroup
	chunkSize := (sampleTimes + workers - 1) / workers

	for w := 0; w < workers; w++ {
		wg.Add(1)
		start := w * chunkSize
		end := start + chunkSize
		if end > sampleTimes {
			end = sampleTimes
		}

		go func(start, end int) {
			defer wg.Done()

			// Each worker has its own RNG for thread safety
			rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(start)))

			for i := start; i < end; i++ {
				// Sample edges
				for j := int64(0); j < l.pnet.MaxLine; j++ {
					// Sample source vertex
					source := l.pnet.SourceSample(rng)
					if source == -1 {
						continue
					}

					// Sample target vertex from source's neighbors
					target := l.pnet.TargetSample(source, rng)
					if target == -1 {
						continue
					}

					// Update based on order
					if l.order == First {
						l.updateFirstOrder(source, target, negativeSamples, currentAlpha, rng)
					} else {
						l.updateSecondOrder(source, target, negativeSamples, currentAlpha, rng)
					}

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
			}
		}(start, end)
	}

	wg.Wait()
	fmt.Printf("\tAlpha: %.6f\tProgress: 100.00 %%\n", currentAlpha)
}

// updateFirstOrder updates embeddings for 1st order proximity
func (l *LINE) updateFirstOrder(source, target int64, negativeSamples int, alpha float64, rng *rand.Rand) {
	// For 1st order, we model direct edges
	// Positive sample
	vertexGrad := make([]float64, l.dim)
	contextGrad := make([]float64, l.dim)

	// Compute score
	score := 0.0
	for d := 0; d < l.dim; d++ {
		score += l.wVertex[source][d] * l.wVertex[target][d]
	}

	// Sigmoid and gradient
	pred := l.pnet.FastSigmoid(score)
	grad := alpha * (1.0 - pred)

	for d := 0; d < l.dim; d++ {
		vertexGrad[d] = grad * l.wVertex[target][d]
		contextGrad[d] = grad * l.wVertex[source][d]
	}

	// Negative samples
	for i := 0; i < negativeSamples; i++ {
		negSample := l.pnet.NegativeSample(rng)
		if negSample == target || negSample == source {
			continue
		}

		score := 0.0
		for d := 0; d < l.dim; d++ {
			score += l.wVertex[source][d] * l.wVertex[negSample][d]
		}

		pred := l.pnet.FastSigmoid(score)
		grad := alpha * (0.0 - pred)

		for d := 0; d < l.dim; d++ {
			vertexGrad[d] += grad * l.wVertex[negSample][d]
			l.wVertex[negSample][d] += grad * l.wVertex[source][d]
		}
	}

	// Update embeddings
	for d := 0; d < l.dim; d++ {
		l.wVertex[source][d] += vertexGrad[d]
		l.wVertex[target][d] += contextGrad[d]
	}
}

// updateSecondOrder updates embeddings for 2nd order proximity
func (l *LINE) updateSecondOrder(source, target int64, negativeSamples int, alpha float64, rng *rand.Rand) {
	// For 2nd order, we use context vectors
	l.pnet.UpdatePair(l.wVertex, l.wContext, source, target, l.dim, negativeSamples, alpha, rng)
}

// SaveWeights saves the learned embeddings to a file
func (l *LINE) SaveWeights(filename string) error {
	fmt.Println("Save Model:")

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer file.Close()

	// Write header
	fmt.Fprintf(file, "%d %d\n", l.pnet.MaxVid, l.dim)

	// Write embeddings
	for vid := int64(0); vid < l.pnet.MaxVid; vid++ {
		name := l.pnet.GetVertexName(vid)
		fmt.Fprintf(file, "%s", name)
		for d := 0; d < l.dim; d++ {
			fmt.Fprintf(file, " %.6f", l.wVertex[vid][d])
		}
		fmt.Fprintln(file)
	}

	fmt.Printf("\tSave to <%s>\n", filename)
	return nil
}
