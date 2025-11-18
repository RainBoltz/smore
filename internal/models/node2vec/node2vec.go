package node2vec

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/cnclabs/smore/pkg/pronet"
)

// Node2Vec implements the Node2Vec algorithm with biased random walks
// Node2Vec extends DeepWalk by using biased random walks that balance
// BFS and DFS exploration strategies via p (return) and q (in-out) parameters
type Node2Vec struct {
	pnet     *pronet.ProNet
	dim      int
	wVertex  [][]float64
	wContext [][]float64

	// Node2Vec specific parameters
	p        float64  // Return parameter (controls likelihood to return to previous node)
	q        float64  // In-out parameter (BFS vs DFS: q > 1 = BFS, q < 1 = DFS)
}

// New creates a new Node2Vec instance
func New() *Node2Vec {
	return &Node2Vec{
		pnet: pronet.NewProNet(),
		p:    1.0,  // Default: unbiased
		q:    1.0,  // Default: unbiased
	}
}

// LoadEdgeList loads the network from an edge list file
func (n2v *Node2Vec) LoadEdgeList(filename string, undirected bool) error {
	return n2v.pnet.LoadEdgeList(filename, undirected)
}

// Init initializes the model with given dimensions and bias parameters
func (n2v *Node2Vec) Init(dim int, p, q float64) {
	n2v.dim = dim
	n2v.p = p
	n2v.q = q
	maxVid := n2v.pnet.MaxVid

	fmt.Println("Model Setting:")
	fmt.Printf("\tdimension:\t\t%d\n", dim)
	fmt.Printf("\tp (return param):\t%.2f\n", p)
	fmt.Printf("\tq (in-out param):\t%.2f\n", q)

	if q > 1.0 {
		fmt.Println("\t(BFS-like exploration: local neighborhood)")
	} else if q < 1.0 {
		fmt.Println("\t(DFS-like exploration: outward expansion)")
	} else {
		fmt.Println("\t(Balanced BFS-DFS exploration)")
	}

	// Initialize vertex embeddings
	n2v.wVertex = make([][]float64, maxVid)
	for vid := int64(0); vid < maxVid; vid++ {
		n2v.wVertex[vid] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			n2v.wVertex[vid][d] = (rand.Float64() - 0.5) / float64(dim)
		}
	}

	// Initialize context embeddings
	n2v.wContext = make([][]float64, maxVid)
	for vid := int64(0); vid < maxVid; vid++ {
		n2v.wContext[vid] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			n2v.wContext[vid][d] = (rand.Float64() - 0.5) / float64(dim)
		}
	}
}

// biasedRandomWalk performs a second-order random walk with bias parameters p and q
func (n2v *Node2Vec) biasedRandomWalk(start int64, steps int, rng *rand.Rand) []int64 {
	walk := make([]int64, 0, steps+1)
	walk = append(walk, start)

	if steps == 0 {
		return walk
	}

	// First step is unbiased
	firstNeighbor := n2v.pnet.TargetSample(start, rng)
	if firstNeighbor == -1 {
		return walk
	}
	walk = append(walk, firstNeighbor)

	// Subsequent steps use biased sampling
	for i := 1; i < steps; i++ {
		current := walk[len(walk)-1]
		previous := walk[len(walk)-2]

		next := n2v.biasedTargetSample(previous, current, rng)
		if next == -1 {
			break
		}
		walk = append(walk, next)
	}

	return walk
}

// biasedTargetSample samples the next node based on the second-order random walk
// with bias parameters p (return) and q (in-out)
func (n2v *Node2Vec) biasedTargetSample(prev, current int64, rng *rand.Rand) int64 {
	neighbors := n2v.pnet.Graph[current]
	if len(neighbors) == 0 {
		return -1
	}

	weights := n2v.pnet.EdgeWeights[current]

	// Calculate biased weights
	biasedWeights := make([]float64, len(neighbors))
	totalWeight := 0.0

	for i, neighbor := range neighbors {
		baseWeight := 1.0
		if len(weights) > 0 {
			baseWeight = weights[i]
		}

		// Apply bias based on relationship to previous node
		var bias float64
		if neighbor == prev {
			// Return to previous node
			bias = 1.0 / n2v.p
		} else if n2v.areNeighbors(prev, neighbor) {
			// Neighbor is also connected to previous (distance 1)
			bias = 1.0
		} else {
			// Neighbor is not connected to previous (distance 2)
			bias = 1.0 / n2v.q
		}

		biasedWeights[i] = baseWeight * bias
		totalWeight += biasedWeights[i]
	}

	// Sample based on biased weights
	if totalWeight == 0 {
		return neighbors[rng.Intn(len(neighbors))]
	}

	r := rng.Float64() * totalWeight
	cumWeight := 0.0
	for i, w := range biasedWeights {
		cumWeight += w
		if r <= cumWeight {
			return neighbors[i]
		}
	}

	return neighbors[len(neighbors)-1]
}

// areNeighbors checks if two nodes are connected
func (n2v *Node2Vec) areNeighbors(vid1, vid2 int64) bool {
	neighbors := n2v.pnet.Graph[vid1]
	for _, n := range neighbors {
		if n == vid2 {
			return true
		}
	}
	return false
}

// Train trains the Node2Vec model
func (n2v *Node2Vec) Train(walkTimes, walkSteps, windowSize, negativeSamples int, alpha float64, workers int) {
	fmt.Println("Model:")
	fmt.Println("\t[Node2Vec]")

	fmt.Println("Learning Parameters:")
	fmt.Printf("\twalk_times:\t\t%d\n", walkTimes)
	fmt.Printf("\twalk_steps:\t\t%d\n", walkSteps)
	fmt.Printf("\twindow_size:\t\t%d\n", windowSize)
	fmt.Printf("\tnegative_samples:\t%d\n", negativeSamples)
	fmt.Printf("\talpha:\t\t\t%.6f\n", alpha)
	fmt.Printf("\tworkers:\t\t%d\n", workers)

	fmt.Println("Start Training:")

	total := int64(walkTimes) * n2v.pnet.MaxVid
	alphaMin := alpha * 0.0001
	currentAlpha := alpha
	count := int64(0)

	var countMu sync.Mutex

	for t := 0; t < walkTimes; t++ {
		// Shuffle vertices for random access
		randomKeys := make([]int64, n2v.pnet.MaxVid)
		for vid := int64(0); vid < n2v.pnet.MaxVid; vid++ {
			randomKeys[vid] = vid
		}
		// Fisher-Yates shuffle
		for vid := int64(0); vid < n2v.pnet.MaxVid; vid++ {
			j := vid + rand.Int63n(n2v.pnet.MaxVid-vid)
			randomKeys[vid], randomKeys[j] = randomKeys[j], randomKeys[vid]
		}

		// Parallel training using goroutines
		var wg sync.WaitGroup
		chunkSize := (int(n2v.pnet.MaxVid) + workers - 1) / workers

		for w := 0; w < workers; w++ {
			wg.Add(1)
			start := w * chunkSize
			end := start + chunkSize
			if end > int(n2v.pnet.MaxVid) {
				end = int(n2v.pnet.MaxVid)
			}

			go func(start, end int) {
				defer wg.Done()

				// Each worker has its own RNG for thread safety
				rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(start)))

				for vid := start; vid < end; vid++ {
					// Perform biased random walk (Node2Vec)
					walks := n2v.biasedRandomWalk(randomKeys[vid], walkSteps, rng)

					// Generate skip-gram pairs
					vertices, contexts := n2v.pnet.SkipGrams(walks, windowSize)

					// Update embeddings
					n2v.pnet.UpdatePairs(n2v.wVertex, n2v.wContext, vertices, contexts, n2v.dim, negativeSamples, currentAlpha, rng)

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
func (n2v *Node2Vec) SaveWeights(filename string) error {
	fmt.Println("Save Model:")

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer file.Close()

	// Write header: num_vertices dimension
	fmt.Fprintf(file, "%d %d\n", n2v.pnet.MaxVid, n2v.dim)

	// Write embeddings
	for vid := int64(0); vid < n2v.pnet.MaxVid; vid++ {
		name := n2v.pnet.GetVertexName(vid)
		fmt.Fprintf(file, "%s", name)
		for d := 0; d < n2v.dim; d++ {
			fmt.Fprintf(file, " %.6f", n2v.wVertex[vid][d])
		}
		fmt.Fprintln(file)
	}

	fmt.Printf("\tSave to <%s>\n", filename)
	return nil
}

// ComputeHomophily computes the homophily ratio for the learned embeddings
// This helps assess the quality of local vs global structure preservation
func (n2v *Node2Vec) ComputeHomophily() float64 {
	totalEdges := 0
	similarEdges := 0

	threshold := 0.5  // Cosine similarity threshold

	for vid := int64(0); vid < n2v.pnet.MaxVid; vid++ {
		neighbors := n2v.pnet.Graph[vid]
		for _, neighbor := range neighbors {
			totalEdges++

			// Compute cosine similarity
			similarity := n2v.cosineSimilarity(n2v.wVertex[vid], n2v.wVertex[neighbor])
			if similarity > threshold {
				similarEdges++
			}
		}
	}

	if totalEdges == 0 {
		return 0.0
	}

	return float64(similarEdges) / float64(totalEdges)
}

// cosineSimilarity computes cosine similarity between two vectors
func (n2v *Node2Vec) cosineSimilarity(a, b []float64) float64 {
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

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}
