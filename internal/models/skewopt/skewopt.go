package skewopt

import (
	"fmt"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/cnclabs/smore/pkg/pronet"
)

// SkewOpt implements Skewed Preference Ranking for network embedding
// This model uses skewed BPR optimization for learning embeddings
type SkewOpt struct {
	pnet    *pronet.ProNet
	dim     int
	wVertex [][]float64
}

// New creates a new SkewOpt model
func New() *SkewOpt {
	return &SkewOpt{
		pnet: pronet.NewProNet(),
	}
}

// LoadEdgeList loads the graph from an edge list file
func (s *SkewOpt) LoadEdgeList(filename string, undirected bool) error {
	return s.pnet.LoadEdgeList(filename, undirected)
}

// Init initializes the model with the given dimension
func (s *SkewOpt) Init(dim int) {
	s.dim = dim
	maxVid := s.pnet.MaxVid

	fmt.Println("Model Setting:")
	fmt.Printf("\tdimension:\t\t%d\n", dim)

	// Initialize vertex embeddings with small offset
	s.wVertex = make([][]float64, maxVid)
	for vid := int64(0); vid < maxVid; vid++ {
		s.wVertex[vid] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			s.wVertex[vid][d] = (rand.Float64()-0.5)/float64(dim) + 0.01
		}
	}
}

// Train trains the SkewOpt model
// Parameters:
// - sampleTimes: number of training samples (in millions)
// - negativeSamples: number of negative samples
// - alpha: initial learning rate
// - reg: L2 regularization parameter
// - xi: skewness parameter (default 10.0)
// - omega: proximity weight (default 3.0)
// - eta: balance parameter/exponent (default 3)
// - workers: number of parallel workers
func (s *SkewOpt) Train(sampleTimes, negativeSamples int, alpha, reg, xi, omega float64, eta, workers int) {
	fmt.Println("Model:")
	fmt.Println("\t[Skew-OPT]")

	fmt.Println("Learning Parameters:")
	fmt.Printf("\tsample_times:\t\t%d\n", sampleTimes)
	fmt.Printf("\talpha:\t\t\t%.6f\n", alpha)
	fmt.Printf("\tregularization:\t\t%.6f\n", reg)
	fmt.Printf("\txi:\t\t\t%.6f\n", xi)
	fmt.Printf("\tomega:\t\t\t%.6f\n", omega)
	fmt.Printf("\teta:\t\t\t%d\n", eta)
	fmt.Printf("\tworkers:\t\t%d\n", workers)

	fmt.Println("Start Training:")

	total := int64(sampleTimes) * 1000000
	alphaMin := alpha * 0.0001
	currentAlpha := alpha
	count := int64(0)

	var countMu sync.Mutex
	var wg sync.WaitGroup
	chunkSize := int64(sampleTimes) * 1000000 / int64(workers)

	for w := 0; w < workers; w++ {
		wg.Add(1)
		start := int64(w) * chunkSize
		end := start + chunkSize
		if w == workers-1 {
			end = total
		}

		go func(start, end int64) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(time.Now().UnixNano() + start))

			for i := start; i < end; i++ {
				// Sample source and target vertices
				v1 := s.pnet.SourceSample(rng)
				v2 := s.pnet.TargetSample(v1, rng)
				if v2 == -1 {
					continue
				}

				// Update using Skewed BPR
				s.pnet.UpdateSBPRPair(s.wVertex, s.wVertex, v1, v2, s.dim, reg, xi, omega, eta, currentAlpha, rng)

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
	fmt.Printf("\tAlpha: %.6f\tProgress: 100.00 %%\n", currentAlpha)
}

// SaveWeights saves the learned embeddings to a file
func (s *SkewOpt) SaveWeights(filename string) error {
	fmt.Println("Save Model:")

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer file.Close()

	fmt.Fprintf(file, "%d %d\n", s.pnet.MaxVid, s.dim)

	for vid := int64(0); vid < s.pnet.MaxVid; vid++ {
		name := s.pnet.GetVertexName(vid)
		fmt.Fprintf(file, "%s", name)
		for d := 0; d < s.dim; d++ {
			fmt.Fprintf(file, " %.6f", s.wVertex[vid][d])
		}
		fmt.Fprintln(file)
	}

	fmt.Printf("\tSave to <%s>\n", filename)
	return nil
}
