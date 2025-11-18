package hpe

import (
	"fmt"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/cnclabs/smore/pkg/pronet"
)

// HPE implements Heterogeneous Preference Embedding
type HPE struct {
	pnet     *pronet.ProNet
	dim      int
	wVertex  [][]float64
	wContext [][]float64
}

// New creates a new HPE instance
func New() *HPE {
	return &HPE{
		pnet: pronet.NewProNet(),
	}
}

// LoadEdgeList loads the network from an edge list file
func (h *HPE) LoadEdgeList(filename string, undirected bool) error {
	return h.pnet.LoadEdgeList(filename, undirected)
}

// Init initializes the model
func (h *HPE) Init(dim int) {
	h.dim = dim
	maxVid := h.pnet.MaxVid

	fmt.Println("Model Setting:")
	fmt.Printf("\tdimension:\t\t%d\n", dim)

	// Initialize embeddings
	h.wVertex = make([][]float64, maxVid)
	for vid := int64(0); vid < maxVid; vid++ {
		h.wVertex[vid] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			h.wVertex[vid][d] = (rand.Float64() - 0.5) / float64(dim)
		}
	}

	h.wContext = make([][]float64, maxVid)
	for vid := int64(0); vid < maxVid; vid++ {
		h.wContext[vid] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			h.wContext[vid][d] = (rand.Float64() - 0.5) / float64(dim)
		}
	}
}

// Train trains the HPE model
func (h *HPE) Train(sampleTimes, negativeSamples int, alpha float64, workers int) {
	fmt.Println("Model:")
	fmt.Println("\t[HPE]")

	fmt.Println("Learning Parameters:")
	fmt.Printf("\tsample_times:\t\t%d\n", sampleTimes)
	fmt.Printf("\tnegative_samples:\t%d\n", negativeSamples)
	fmt.Printf("\talpha:\t\t\t%.6f\n", alpha)
	fmt.Printf("\tworkers:\t\t%d\n", workers)

	fmt.Println("Start Training:")

	total := int64(sampleTimes) * h.pnet.MaxLine
	alphaMin := alpha * 0.0001
	currentAlpha := alpha
	count := int64(0)

	var countMu sync.Mutex

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

			rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(start)))

			for i := start; i < end; i++ {
				for j := int64(0); j < h.pnet.MaxLine; j++ {
					source := h.pnet.SourceSample(rng)
					if source == -1 {
						continue
					}

					target := h.pnet.TargetSample(source, rng)
					if target == -1 {
						continue
					}

					h.pnet.UpdatePair(h.wVertex, h.wContext, source, target, h.dim, negativeSamples, currentAlpha, rng)

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

// SaveWeights saves the learned embeddings
func (h *HPE) SaveWeights(filename string) error {
	fmt.Println("Save Model:")

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer file.Close()

	fmt.Fprintf(file, "%d %d\n", h.pnet.MaxVid, h.dim)

	for vid := int64(0); vid < h.pnet.MaxVid; vid++ {
		name := h.pnet.GetVertexName(vid)
		fmt.Fprintf(file, "%s", name)
		for d := 0; d < h.dim; d++ {
			fmt.Fprintf(file, " %.6f", h.wVertex[vid][d])
		}
		fmt.Fprintln(file)
	}

	fmt.Printf("\tSave to <%s>\n", filename)
	return nil
}
