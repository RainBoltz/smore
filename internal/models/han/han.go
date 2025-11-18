package han

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/cnclabs/smore/pkg/hetero"
)

// HAN implements Heterogeneous Attention Network
// Combines node-level and semantic-level attention for heterogeneous graphs
type HAN struct {
	hg  *hetero.HeteroGraph
	dim int

	// Embeddings
	embeddings [][]float64

	// Meta-paths for semantic attention
	metaPaths [][]string

	// Attention parameters
	nodeAttention     [][][]float64 // [metapath][node][neighbor] -> attention weight
	semanticAttention [][]float64   // [node][metapath] -> semantic weight

	// Transformation matrices for each meta-path
	transformations [][][]float64 // [metapath][dim][dim]

	// Attention vectors
	attentionVectors [][]float64 // [metapath][dim]

	// Semantic attention vector
	semanticVector []float64 // [dim]

	// Training parameters
	learningRate float64
}

// New creates a new HAN instance
func New() *HAN {
	return &HAN{
		hg:        hetero.NewHeteroGraph(),
		metaPaths: make([][]string, 0),
	}
}

// LoadEdgeList loads the heterogeneous graph
func (han *HAN) LoadEdgeList(filename string, undirected bool) error {
	return han.hg.LoadEdgeList(filename, undirected)
}

// AddMetaPath adds a meta-path for training
func (han *HAN) AddMetaPath(metaPath string) error {
	types := strings.Fields(metaPath)
	if len(types) < 2 {
		return fmt.Errorf("meta-path must have at least 2 types, got: %s", metaPath)
	}

	// Validate meta-path
	if err := han.hg.ValidateMetaPath(types); err != nil {
		return fmt.Errorf("invalid meta-path: %v", err)
	}

	han.metaPaths = append(han.metaPaths, types)
	fmt.Printf("Added meta-path: %s\n", metaPath)

	return nil
}

// Init initializes the model
func (han *HAN) Init(dim int, learningRate float64) {
	han.dim = dim
	han.learningRate = learningRate

	fmt.Println("Model Setting:")
	fmt.Printf("\tdimension:\t\t%d\n", dim)
	fmt.Printf("\tlearning_rate:\t\t%.6f\n", learningRate)
	fmt.Printf("\tmeta-paths:\t\t%d\n", len(han.metaPaths))

	if len(han.metaPaths) == 0 {
		fmt.Println("\t⚠ Warning: No meta-paths defined. Please add meta-paths before training.")
	} else {
		fmt.Println("\nMeta-paths:")
		for i, mp := range han.metaPaths {
			fmt.Printf("\t[%d] %s\n", i+1, strings.Join(mp, " -> "))
		}
	}

	fmt.Println()
	fmt.Println("HAN Principle:")
	fmt.Println("\t✓ Node-level attention: learns neighbor importance")
	fmt.Println("\t✓ Semantic-level attention: learns meta-path importance")
	fmt.Println("\t✓ Hierarchical attention aggregation")

	// Initialize embeddings
	han.embeddings = make([][]float64, han.hg.NumNodes)
	for i := int64(0); i < han.hg.NumNodes; i++ {
		han.embeddings[i] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			han.embeddings[i][d] = (rand.Float64() - 0.5) / float64(dim)
		}
	}

	// Initialize transformation matrices for each meta-path
	han.transformations = make([][][]float64, len(han.metaPaths))
	for p := range han.metaPaths {
		han.transformations[p] = make([][]float64, dim)
		for i := 0; i < dim; i++ {
			han.transformations[p][i] = make([]float64, dim)
			for j := 0; j < dim; j++ {
				if i == j {
					han.transformations[p][i][j] = 1.0
				} else {
					han.transformations[p][i][j] = (rand.Float64() - 0.5) / float64(dim)
				}
			}
		}
	}

	// Initialize attention vectors for each meta-path
	han.attentionVectors = make([][]float64, len(han.metaPaths))
	for p := range han.metaPaths {
		han.attentionVectors[p] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			han.attentionVectors[p][d] = (rand.Float64() - 0.5) / float64(dim)
		}
	}

	// Initialize semantic attention vector
	han.semanticVector = make([]float64, dim)
	for d := 0; d < dim; d++ {
		han.semanticVector[d] = (rand.Float64() - 0.5) / float64(dim)
	}

	// Initialize attention storage
	han.nodeAttention = make([][][]float64, len(han.metaPaths))
	for p := range han.metaPaths {
		han.nodeAttention[p] = make([][]float64, han.hg.NumNodes)
	}

	han.semanticAttention = make([][]float64, han.hg.NumNodes)
	for i := int64(0); i < han.hg.NumNodes; i++ {
		han.semanticAttention[i] = make([]float64, len(han.metaPaths))
	}
}

// computeNodeAttention computes attention weights for neighbors via a meta-path
func (han *HAN) computeNodeAttention(nodeID int64, metaPathIdx int, neighbors []int64) []float64 {
	if len(neighbors) == 0 {
		return nil
	}

	attention := make([]float64, len(neighbors))
	totalExp := 0.0

	// Compute attention scores using transformed features
	for i, neighbor := range neighbors {
		// Transform node and neighbor embeddings
		nodeTransformed := han.transformEmbedding(han.embeddings[nodeID], metaPathIdx)
		neighborTransformed := han.transformEmbedding(han.embeddings[neighbor], metaPathIdx)

		// Compute attention score: a^T * (W*h_i || W*h_j)
		score := 0.0
		for d := 0; d < han.dim; d++ {
			score += han.attentionVectors[metaPathIdx][d] * (nodeTransformed[d] + neighborTransformed[d])
		}

		// LeakyReLU activation
		if score < 0 {
			score = 0.01 * score
		}

		attention[i] = math.Exp(score)
		totalExp += attention[i]
	}

	// Normalize with softmax
	if totalExp > 0 {
		for i := range attention {
			attention[i] /= totalExp
		}
	}

	return attention
}

// transformEmbedding applies transformation matrix
func (han *HAN) transformEmbedding(embedding []float64, metaPathIdx int) []float64 {
	result := make([]float64, han.dim)
	for i := 0; i < han.dim; i++ {
		for j := 0; j < han.dim; j++ {
			result[i] += han.transformations[metaPathIdx][i][j] * embedding[j]
		}
	}
	return result
}

// aggregateWithNodeAttention aggregates neighbor embeddings with attention
func (han *HAN) aggregateWithNodeAttention(nodeID int64, metaPathIdx int, neighbors []int64, attention []float64) []float64 {
	aggregated := make([]float64, han.dim)

	for i, neighbor := range neighbors {
		transformed := han.transformEmbedding(han.embeddings[neighbor], metaPathIdx)
		for d := 0; d < han.dim; d++ {
			aggregated[d] += attention[i] * transformed[d]
		}
	}

	return aggregated
}

// computeSemanticAttention computes attention weights for meta-paths
func (han *HAN) computeSemanticAttention(nodeID int64, metaPathEmbeddings [][]float64) []float64 {
	if len(metaPathEmbeddings) == 0 {
		return nil
	}

	attention := make([]float64, len(metaPathEmbeddings))
	totalExp := 0.0

	for p, embedding := range metaPathEmbeddings {
		if embedding == nil {
			continue
		}

		// Compute attention score: q^T * tanh(W * z_p)
		score := 0.0
		for d := 0; d < han.dim; d++ {
			// Apply tanh activation
			tanhValue := math.Tanh(embedding[d])
			score += han.semanticVector[d] * tanhValue
		}

		attention[p] = math.Exp(score)
		totalExp += attention[p]
	}

	// Normalize with softmax
	if totalExp > 0 {
		for p := range attention {
			attention[p] /= totalExp
		}
	}

	return attention
}

// Train trains the HAN model
func (han *HAN) Train(walkTimes, walkSteps, epochs int, workers int) {
	if len(han.metaPaths) == 0 {
		fmt.Println("Error: No meta-paths defined. Use AddMetaPath() before training.")
		return
	}

	fmt.Println()
	fmt.Println("Model:")
	fmt.Println("\t[HAN - Heterogeneous Attention Network]")
	fmt.Println()

	fmt.Println("Learning Parameters:")
	fmt.Printf("\twalk_times:\t\t%d\n", walkTimes)
	fmt.Printf("\twalk_steps:\t\t%d\n", walkSteps)
	fmt.Printf("\tepochs:\t\t\t%d\n", epochs)
	fmt.Printf("\tlearning_rate:\t\t%.6f\n", han.learningRate)
	fmt.Printf("\tworkers:\t\t%d\n", workers)
	fmt.Println()

	fmt.Println("Start Training:")

	for epoch := 0; epoch < epochs; epoch++ {
		fmt.Printf("\nEpoch %d/%d:\n", epoch+1, epochs)

		// Shuffle nodes
		randomKeys := make([]int64, han.hg.NumNodes)
		for i := int64(0); i < han.hg.NumNodes; i++ {
			randomKeys[i] = i
		}
		// Fisher-Yates shuffle
		for i := int64(0); i < han.hg.NumNodes; i++ {
			j := i + rand.Int63n(han.hg.NumNodes-i)
			randomKeys[i], randomKeys[j] = randomKeys[j], randomKeys[i]
		}

		// Parallel training
		var wg sync.WaitGroup
		chunkSize := (int(han.hg.NumNodes) + workers - 1) / workers
		progressCount := int64(0)
		var progressMu sync.Mutex

		for w := 0; w < workers; w++ {
			wg.Add(1)
			start := w * chunkSize
			end := start + chunkSize
			if end > int(han.hg.NumNodes) {
				end = int(han.hg.NumNodes)
			}

			go func(start, end int) {
				defer wg.Done()

				rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(start)))

				for i := start; i < end; i++ {
					nodeID := randomKeys[i]

					// Collect meta-path-specific embeddings
					metaPathEmbeddings := make([][]float64, len(han.metaPaths))

					for p, metaPath := range han.metaPaths {
						// Sample neighbors via meta-path walk
						neighbors := make([]int64, 0)
						for w := 0; w < walkTimes; w++ {
							walk := han.hg.MetaPathWalk(nodeID, metaPath, walkSteps, rng)
							if len(walk) > 1 {
								// Add end node as neighbor
								neighbors = append(neighbors, walk[len(walk)-1])
							}
						}

						if len(neighbors) == 0 {
							continue
						}

						// Compute node-level attention
						attention := han.computeNodeAttention(nodeID, p, neighbors)

						// Aggregate with attention
						metaPathEmbeddings[p] = han.aggregateWithNodeAttention(nodeID, p, neighbors, attention)
					}

					// Compute semantic-level attention
					semanticAttention := han.computeSemanticAttention(nodeID, metaPathEmbeddings)

					// Aggregate across meta-paths
					finalEmbedding := make([]float64, han.dim)
					for p, embedding := range metaPathEmbeddings {
						if embedding != nil {
							for d := 0; d < han.dim; d++ {
								finalEmbedding[d] += semanticAttention[p] * embedding[d]
							}
						}
					}

					// Update node embedding with gradient descent
					for d := 0; d < han.dim; d++ {
						han.embeddings[nodeID][d] += han.learningRate * finalEmbedding[d]
					}

					// Normalize embedding
					norm := 0.0
					for d := 0; d < han.dim; d++ {
						norm += han.embeddings[nodeID][d] * han.embeddings[nodeID][d]
					}
					norm = math.Sqrt(norm)
					if norm > 0 {
						for d := 0; d < han.dim; d++ {
							han.embeddings[nodeID][d] /= norm
						}
					}

					// Update progress
					progressMu.Lock()
					progressCount++
					if progressCount%1000 == 0 {
						progress := float64(progressCount) / float64(han.hg.NumNodes) * 100
						fmt.Printf("\tProgress: %.2f%%\r", progress)
					}
					progressMu.Unlock()
				}
			}(start, end)
		}

		wg.Wait()
		fmt.Printf("\tProgress: 100.00%%\n")
	}

	fmt.Println("\nTraining Complete!")
}

// SaveEmbeddings saves the learned embeddings
func (han *HAN) SaveEmbeddings(filename string) error {
	fmt.Println("Save Model:")

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer file.Close()

	// Write header: num_nodes dimension
	fmt.Fprintf(file, "%d %d\n", han.hg.NumNodes, han.dim)

	// Write embeddings with node types
	for i := int64(0); i < han.hg.NumNodes; i++ {
		name := han.hg.GetNodeName(i)
		nodeType := han.hg.GetNodeType(i)
		fmt.Fprintf(file, "%s[%s]", name, nodeType)

		for d := 0; d < han.dim; d++ {
			fmt.Fprintf(file, " %.6f", han.embeddings[i][d])
		}
		fmt.Fprintln(file)
	}

	fmt.Printf("\tSave to <%s>\n", filename)
	return nil
}

// GetEmbedding returns the embedding for a node
func (han *HAN) GetEmbedding(nodeID int64) []float64 {
	if nodeID < 0 || nodeID >= han.hg.NumNodes {
		return nil
	}
	return han.embeddings[nodeID]
}

// ComputeAttentionStats computes statistics about learned attention
func (han *HAN) ComputeAttentionStats() {
	fmt.Println("\nAttention Statistics:")

	// Sample some nodes to check semantic attention
	samples := 5
	if samples > int(han.hg.NumNodes) {
		samples = int(han.hg.NumNodes)
	}

	fmt.Println("\nSample Semantic Attention Weights:")
	for i := 0; i < samples; i++ {
		nodeID := rand.Int63n(han.hg.NumNodes)
		name := han.hg.GetNodeName(nodeID)
		nodeType := han.hg.GetNodeType(nodeID)

		fmt.Printf("\tNode: %s[%s]\n", name, nodeType)

		// Compute current semantic attention for this node
		metaPathEmbeddings := make([][]float64, len(han.metaPaths))
		rng := rand.New(rand.NewSource(time.Now().UnixNano()))

		for p, metaPath := range han.metaPaths {
			// Sample a few neighbors
			neighbors := make([]int64, 0)
			for w := 0; w < 5; w++ {
				walk := han.hg.MetaPathWalk(nodeID, metaPath, 3, rng)
				if len(walk) > 1 {
					neighbors = append(neighbors, walk[len(walk)-1])
				}
			}

			if len(neighbors) > 0 {
				attention := han.computeNodeAttention(nodeID, p, neighbors)
				metaPathEmbeddings[p] = han.aggregateWithNodeAttention(nodeID, p, neighbors, attention)
			}
		}

		semanticAttention := han.computeSemanticAttention(nodeID, metaPathEmbeddings)
		for p, mp := range han.metaPaths {
			if metaPathEmbeddings[p] != nil {
				fmt.Printf("\t\t%s: %.4f\n", strings.Join(mp, "->"), semanticAttention[p])
			}
		}
	}
}
