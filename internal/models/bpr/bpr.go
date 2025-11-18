package bpr

import (
	"fmt"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/cnclabs/smore/pkg/pronet"
)

// BPR implements Bayesian Personalized Ranking
type BPR struct {
	pnet     *pronet.ProNet
	dim      int
	wVertex  [][]float64
	wContext [][]float64
}

// New creates a new BPR instance
func New() *BPR {
	return &BPR{
		pnet: pronet.NewProNet(),
	}
}

// LoadEdgeList loads the network from an edge list file
func (b *BPR) LoadEdgeList(filename string, undirected bool) error {
	return b.pnet.LoadEdgeList(filename, undirected)
}

// Init initializes the model
func (b *BPR) Init(dim int) {
	b.dim = dim
	maxVid := b.pnet.MaxVid

	fmt.Println("Model Setting:")
	fmt.Printf("\tdimension:\t\t%d\n", dim)

	// Initialize vertex embeddings
	b.wVertex = make([][]float64, maxVid)
	for vid := int64(0); vid < maxVid; vid++ {
		b.wVertex[vid] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			b.wVertex[vid][d] = (rand.Float64() - 0.5) / float64(dim)
		}
	}

	// Initialize context (item) embeddings
	b.wContext = make([][]float64, maxVid)
	for vid := int64(0); vid < maxVid; vid++ {
		b.wContext[vid] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			b.wContext[vid][d] = (rand.Float64() - 0.5) / float64(dim)
		}
	}
}

// Train trains the BPR model
func (b *BPR) Train(sampleTimes int, alpha, lambda float64, workers int) {
	fmt.Println("Model:")
	fmt.Println("\t[BPR]")

	fmt.Println("Learning Parameters:")
	fmt.Printf("\tsample_times:\t\t%d\n", sampleTimes)
	fmt.Printf("\talpha:\t\t\t%.6f\n", alpha)
	fmt.Printf("\tlambda:\t\t\t%.6f\n", lambda)
	fmt.Printf("\tworkers:\t\t%d\n", workers)

	fmt.Println("Start Training:")

	total := int64(sampleTimes) * b.pnet.MaxLine
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

			rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(start)))

			for i := start; i < end; i++ {
				for j := int64(0); j < b.pnet.MaxLine; j++ {
					// Sample user
					user := b.pnet.SourceSample(rng)
					if user == -1 {
						continue
					}

					// Sample positive item
					posItem := b.pnet.TargetSample(user, rng)
					if posItem == -1 {
						continue
					}

					// Sample negative item
					negItem := b.pnet.NegativeSample(rng)

					// Update using BPR
					b.pnet.UpdateBPRPair(b.wVertex, b.wContext, user, posItem, negItem, b.dim, currentAlpha, lambda, rng)

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

// SaveWeights saves the learned embeddings
func (b *BPR) SaveWeights(filename string) error {
	fmt.Println("Save Model:")

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer file.Close()

	fmt.Fprintf(file, "%d %d\n", b.pnet.MaxVid, b.dim)

	for vid := int64(0); vid < b.pnet.MaxVid; vid++ {
		name := b.pnet.GetVertexName(vid)
		fmt.Fprintf(file, "%s", name)
		for d := 0; d < b.dim; d++ {
			fmt.Fprintf(file, " %.6f", b.wVertex[vid][d])
		}
		fmt.Fprintln(file)
	}

	fmt.Printf("\tSave to <%s>\n", filename)
	return nil
}
