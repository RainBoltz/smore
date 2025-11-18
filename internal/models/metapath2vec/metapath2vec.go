package metapath2vec

import (
	"fmt"
	"math/rand"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/cnclabs/smore/pkg/hetero"
	"github.com/cnclabs/smore/pkg/pronet"
)

// Metapath2Vec implements heterogeneous graph embedding via meta-paths
// Meta-paths define walks through typed nodes (e.g., User->Item->Category->Item->User)
type Metapath2Vec struct {
	hg  *hetero.HeteroGraph
	dim int

	// Embeddings
	embeddings [][]float64

	// Meta-paths: sequences of node types
	metaPaths [][]string

	// Training context
	contextEmbeddings [][]float64
}

// New creates a new Metapath2Vec instance
func New() *Metapath2Vec {
	return &Metapath2Vec{
		hg:        hetero.NewHeteroGraph(),
		metaPaths: make([][]string, 0),
	}
}

// LoadEdgeList loads the heterogeneous graph
func (mp *Metapath2Vec) LoadEdgeList(filename string, undirected bool) error {
	return mp.hg.LoadEdgeList(filename, undirected)
}

// AddMetaPath adds a meta-path for training
// metaPath: sequence of node types (e.g., "User Item Category Item User")
func (mp *Metapath2Vec) AddMetaPath(metaPath string) error {
	types := strings.Fields(metaPath)
	if len(types) < 2 {
		return fmt.Errorf("meta-path must have at least 2 types, got: %s", metaPath)
	}

	// Validate meta-path
	if err := mp.hg.ValidateMetaPath(types); err != nil {
		return fmt.Errorf("invalid meta-path: %v", err)
	}

	mp.metaPaths = append(mp.metaPaths, types)
	fmt.Printf("Added meta-path: %s\n", metaPath)

	return nil
}

// Init initializes the model
func (mp *Metapath2Vec) Init(dim int) {
	mp.dim = dim

	fmt.Println("Model Setting:")
	fmt.Printf("\tdimension:\t\t%d\n", dim)
	fmt.Printf("\tmeta-paths:\t\t%d\n", len(mp.metaPaths))

	if len(mp.metaPaths) == 0 {
		fmt.Println("\t⚠ Warning: No meta-paths defined. Please add meta-paths before training.")
	} else {
		fmt.Println("\nMeta-paths:")
		for i, mp := range mp.metaPaths {
			fmt.Printf("\t[%d] %s\n", i+1, strings.Join(mp, " -> "))
		}
	}

	fmt.Println()
	fmt.Println("Metapath2Vec Principle:")
	fmt.Println("\t✓ Heterogeneous graphs with typed nodes")
	fmt.Println("\t✓ Meta-path-guided random walks")
	fmt.Println("\t✓ Preserves type-specific relationships")

	// Initialize embeddings
	mp.embeddings = make([][]float64, mp.hg.NumNodes)
	for i := int64(0); i < mp.hg.NumNodes; i++ {
		mp.embeddings[i] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			mp.embeddings[i][d] = (rand.Float64() - 0.5) / float64(dim)
		}
	}

	// Initialize context embeddings
	mp.contextEmbeddings = make([][]float64, mp.hg.NumNodes)
	for i := int64(0); i < mp.hg.NumNodes; i++ {
		mp.contextEmbeddings[i] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			mp.contextEmbeddings[i][d] = (rand.Float64() - 0.5) / float64(dim)
		}
	}
}

// Train trains the Metapath2Vec model
func (mp *Metapath2Vec) Train(walkTimes, walkSteps, windowSize, negativeSamples int, alpha float64, workers int) {
	if len(mp.metaPaths) == 0 {
		fmt.Println("Error: No meta-paths defined. Use AddMetaPath() before training.")
		return
	}

	fmt.Println()
	fmt.Println("Model:")
	fmt.Println("\t[Metapath2Vec - Heterogeneous Graph Embedding]")
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

	total := int64(walkTimes) * mp.hg.NumNodes
	alphaMin := alpha * 0.0001
	currentAlpha := alpha
	count := int64(0)

	var countMu sync.Mutex

	// Create ProNet instance for skip-gram training
	pnet := pronet.NewProNet()
	pnet.MaxVid = mp.hg.NumNodes

	// Build negative sampling alias table
	negDistribution := make([]float64, mp.hg.NumNodes)
	for i := int64(0); i < mp.hg.NumNodes; i++ {
		// Use uniform distribution for simplicity (could be weighted by degree)
		negDistribution[i] = 1.0
	}
	pnet.NegativeAT = pronet.BuildAliasMethod(negDistribution, 0.75)

	for t := 0; t < walkTimes; t++ {
		// Shuffle nodes for random access
		randomKeys := make([]int64, mp.hg.NumNodes)
		for i := int64(0); i < mp.hg.NumNodes; i++ {
			randomKeys[i] = i
		}
		// Fisher-Yates shuffle
		for i := int64(0); i < mp.hg.NumNodes; i++ {
			j := i + rand.Int63n(mp.hg.NumNodes-i)
			randomKeys[i], randomKeys[j] = randomKeys[j], randomKeys[i]
		}

		// Parallel training
		var wg sync.WaitGroup
		chunkSize := (int(mp.hg.NumNodes) + workers - 1) / workers

		for w := 0; w < workers; w++ {
			wg.Add(1)
			start := w * chunkSize
			end := start + chunkSize
			if end > int(mp.hg.NumNodes) {
				end = int(mp.hg.NumNodes)
			}

			go func(start, end int) {
				defer wg.Done()

				rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(start)))

				for i := start; i < end; i++ {
					nodeID := randomKeys[i]

					// Select a random meta-path
					metaPath := mp.metaPaths[rng.Intn(len(mp.metaPaths))]

					// Perform meta-path walk
					walk := mp.hg.MetaPathWalk(nodeID, metaPath, walkSteps, rng)

					if len(walk) < 2 {
						continue
					}

					// Generate skip-gram pairs
					vertices, contexts := pnet.SkipGrams(walk, windowSize)

					// Update embeddings
					pnet.UpdatePairs(mp.embeddings, mp.contextEmbeddings, vertices, contexts, mp.dim, negativeSamples, currentAlpha, rng)

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
func (mp *Metapath2Vec) SaveEmbeddings(filename string) error {
	fmt.Println("Save Model:")

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer file.Close()

	// Write header: num_nodes dimension
	fmt.Fprintf(file, "%d %d\n", mp.hg.NumNodes, mp.dim)

	// Write embeddings with node types
	for i := int64(0); i < mp.hg.NumNodes; i++ {
		name := mp.hg.GetNodeName(i)
		nodeType := mp.hg.GetNodeType(i)
		fmt.Fprintf(file, "%s[%s]", name, nodeType)

		for d := 0; d < mp.dim; d++ {
			fmt.Fprintf(file, " %.6f", mp.embeddings[i][d])
		}
		fmt.Fprintln(file)
	}

	fmt.Printf("\tSave to <%s>\n", filename)
	return nil
}

// GetEmbedding returns the embedding for a node
func (mp *Metapath2Vec) GetEmbedding(nodeID int64) []float64 {
	if nodeID < 0 || nodeID >= mp.hg.NumNodes {
		return nil
	}
	return mp.embeddings[nodeID]
}

// ComputeTypeHomogeneity computes how well nodes of same type cluster together
func (mp *Metapath2Vec) ComputeTypeHomogeneity() map[string]float64 {
	homogeneity := make(map[string]float64)
	typeCounts := make(map[string]int)

	// Sample pairs for efficiency
	samples := 1000
	if samples > int(mp.hg.NumNodes)*10 {
		samples = int(mp.hg.NumNodes) * 10
	}

	for i := 0; i < samples; i++ {
		// Random node
		node1 := rand.Int63n(mp.hg.NumNodes)
		node2 := rand.Int63n(mp.hg.NumNodes)

		if node1 == node2 {
			continue
		}

		type1 := mp.hg.GetNodeType(node1)
		type2 := mp.hg.GetNodeType(node2)

		if type1 == type2 {
			// Compute cosine similarity
			sim := cosineSimilarity(mp.embeddings[node1], mp.embeddings[node2])
			homogeneity[type1] += sim
			typeCounts[type1]++
		}
	}

	// Average
	for t := range homogeneity {
		if typeCounts[t] > 0 {
			homogeneity[t] /= float64(typeCounts[t])
		}
	}

	return homogeneity
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

	return dotProduct / (normA * normB)
}
