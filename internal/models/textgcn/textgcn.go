package textgcn

import (
	"fmt"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/cnclabs/smore/pkg/pronet"
)

// TextGCN implements Text Graph Convolutional Network for text classification
// Based on "Graph Convolutional Networks for Text Classification" (AAAI 2019)
type TextGCN struct {
	pnet      *pronet.ProNet
	dim       int
	wVertex   [][]float64
	wContext  [][]float64
}

// New creates a new TextGCN model
func New() *TextGCN {
	return &TextGCN{
		pnet: pronet.NewProNet(),
	}
}

// LoadEdgeList loads the graph from an edge list file
func (t *TextGCN) LoadEdgeList(filename string, undirected bool) error {
	return t.pnet.LoadEdgeList(filename, undirected)
}

// LoadFieldMeta loads field metadata (document/word types)
func (t *TextGCN) LoadFieldMeta(filename string) error {
	return t.pnet.LoadFieldMeta(filename)
}

// Init initializes the model with the given dimension
func (t *TextGCN) Init(dim int) {
	t.dim = dim
	maxVid := t.pnet.MaxVid

	fmt.Println("Model Setting:")
	fmt.Printf("\tdimension:\t\t%d\n", dim)

	// Initialize vertex embeddings
	t.wVertex = make([][]float64, maxVid)
	for vid := int64(0); vid < maxVid; vid++ {
		t.wVertex[vid] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			t.wVertex[vid][d] = (rand.Float64() - 0.5) / float64(dim)
		}
	}

	// Initialize context embeddings
	t.wContext = make([][]float64, maxVid)
	for vid := int64(0); vid < maxVid; vid++ {
		t.wContext[vid] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			t.wContext[vid][d] = (rand.Float64() - 0.5) / float64(dim)
		}
	}
}

// Train trains the TextGCN model
func (t *TextGCN) Train(sampleTimes, walkSteps, negativeSamples int, reg, alpha float64, workers int) {
	fmt.Println("Model:")
	fmt.Println("\t[TextGCN]")

	fmt.Println("Learning Parameters:")
	fmt.Printf("\tsample_times:\t\t%d\n", sampleTimes)
	fmt.Printf("\tnegative_samples:\t%d\n", negativeSamples)
	fmt.Printf("\twalk_steps:\t\t%d\n", walkSteps)
	fmt.Printf("\tregularization:\t\t%.6f\n", reg)
	fmt.Printf("\talpha:\t\t\t%.6f\n", alpha)
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
				// Sample source vertex (must be field type 0 - documents)
				v1 := t.pnet.SourceSample(rng)
				if len(t.pnet.Fields) > 0 && len(t.pnet.Fields[v1].Fields) > 0 {
					for t.pnet.Fields[v1].Fields[0] != 0 {
						v1 = t.pnet.SourceSample(rng)
					}
				}

				// Sample target vertex (words connected to the document)
				v2 := t.pnet.TargetSample(v1, rng)
				if v2 == -1 {
					continue
				}

				// Collect context vertices for CBOW
				contexts := make([]int64, 0, walkSteps)
				contexts = append(contexts, v2)

				// Sample additional context vertices if walkSteps > 1
				for step := 1; step < walkSteps; step++ {
					ctx := t.pnet.TargetSample(v1, rng)
					if ctx != -1 {
						contexts = append(contexts, ctx)
					}
				}

				// Update using CBOW: predict document from word contexts
				if len(contexts) > 0 {
					t.pnet.UpdateCBOW(t.wVertex, t.wContext, contexts, v1, t.dim, negativeSamples, currentAlpha, rng)
				}

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
func (t *TextGCN) SaveWeights(filename string) error {
	fmt.Println("Save Model:")

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer file.Close()

	// Count vertices to save (exclude field type 1)
	counter := 0
	for vid := int64(0); vid < t.pnet.MaxVid; vid++ {
		if len(t.pnet.Fields) == 0 || len(t.pnet.Fields[vid].Fields) == 0 || t.pnet.Fields[vid].Fields[0] != 1 {
			counter++
		}
	}

	fmt.Fprintf(file, "%d %d\n", counter, t.dim)

	// Temporary vector for aggregating neighbor embeddings
	wAvg := make([]float64, t.dim)

	for vid := int64(0); vid < t.pnet.MaxVid; vid++ {
		// Skip field type 1
		if len(t.pnet.Fields) > 0 && len(t.pnet.Fields[vid].Fields) > 0 && t.pnet.Fields[vid].Fields[0] == 1 {
			continue
		}

		name := t.pnet.GetVertexName(vid)
		fmt.Fprintf(file, "%s", name)

		// For field type 0 (documents): aggregate neighbor embeddings
		if len(t.pnet.Fields) > 0 && len(t.pnet.Fields[vid].Fields) > 0 && t.pnet.Fields[vid].Fields[0] == 0 {
			// Reset aggregation vector
			for d := 0; d < t.dim; d++ {
				wAvg[d] = 0.0
			}

			// Get neighbors from graph
			if neighbors, exists := t.pnet.Graph[vid]; exists {
				for _, neighborVid := range neighbors {
					for d := 0; d < t.dim; d++ {
						wAvg[d] += t.wVertex[neighborVid][d]
					}
				}
			}

			// Write aggregated embedding
			for d := 0; d < t.dim; d++ {
				fmt.Fprintf(file, " %.6f", wAvg[d])
			}
		} else if len(t.pnet.Fields) > 0 && len(t.pnet.Fields[vid].Fields) > 0 && t.pnet.Fields[vid].Fields[0] == 2 {
			// For field type 2 (words): use direct embedding
			for d := 0; d < t.dim; d++ {
				fmt.Fprintf(file, " %.6f", t.wVertex[vid][d])
			}
		} else {
			// Default: use direct embedding
			for d := 0; d < t.dim; d++ {
				fmt.Fprintf(file, " %.6f", t.wVertex[vid][d])
			}
		}

		fmt.Fprintln(file)
	}

	fmt.Printf("\tSave to <%s>\n", filename)
	return nil
}
