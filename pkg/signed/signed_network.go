package signed

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
)

// SignedEdge represents a signed edge (positive or negative)
type SignedEdge struct {
	From   int64
	To     int64
	Sign   float64 // +1 for positive, -1 for negative
	Weight float64 // Edge weight/strength
}

// SignedNetwork represents a network with signed edges
type SignedNetwork struct {
	// Vertex mapping
	VertexHash map[string]int64
	VertexKeys []string

	// Signed edges
	PositiveEdges map[int64][]int64 // Positive neighbors
	NegativeEdges map[int64][]int64 // Negative neighbors
	PositiveWeights map[int64][]float64
	NegativeWeights map[int64][]float64

	// Statistics
	NumVertices      int64
	NumPositiveEdges int64
	NumNegativeEdges int64
	TotalEdges       int64

	// Degrees
	PositiveDegree map[int64]float64
	NegativeDegree map[int64]float64
}

// NewSignedNetwork creates a new signed network instance
func NewSignedNetwork() *SignedNetwork {
	return &SignedNetwork{
		VertexHash:      make(map[string]int64),
		VertexKeys:      make([]string, 0),
		PositiveEdges:   make(map[int64][]int64),
		NegativeEdges:   make(map[int64][]int64),
		PositiveWeights: make(map[int64][]float64),
		NegativeWeights: make(map[int64][]float64),
		PositiveDegree:  make(map[int64]float64),
		NegativeDegree:  make(map[int64]float64),
	}
}

// LoadEdgeList loads signed network from edge list file
// Format: from to sign [weight]
// Example: "Alice Bob +1 1.0" or "Alice Charlie -1 1.0"
// Sign can be: +1, 1, pos, positive (positive) or -1, neg, negative (negative)
func (sn *SignedNetwork) LoadEdgeList(filename string, undirected bool) error {
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open file %s: %v", filename, err)
	}
	defer file.Close()

	fmt.Println("Loading signed network from:", filename)

	scanner := bufio.NewScanner(file)
	lineCount := int64(0)

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Fields(line)

		if len(parts) < 3 {
			continue
		}

		from := parts[0]
		to := parts[1]
		signStr := parts[2]

		// Parse sign
		var sign float64
		switch strings.ToLower(signStr) {
		case "+1", "1", "pos", "positive":
			sign = 1.0
		case "-1", "neg", "negative":
			sign = -1.0
		default:
			s, err := strconv.ParseFloat(signStr, 64)
			if err != nil {
				fmt.Printf("Warning: invalid sign at line %d: %s\n", lineCount, signStr)
				continue
			}
			if s > 0 {
				sign = 1.0
			} else {
				sign = -1.0
			}
		}

		// Parse weight
		weight := 1.0
		if len(parts) >= 4 {
			w, err := strconv.ParseFloat(parts[3], 64)
			if err == nil {
				weight = w
			}
		}

		// Get or create vertex IDs
		fromID := sn.getOrCreateVertex(from)
		toID := sn.getOrCreateVertex(to)

		// Add edge
		sn.addEdge(fromID, toID, sign, weight)
		if undirected {
			sn.addEdge(toID, fromID, sign, weight)
		}

		lineCount++
		if lineCount%10000 == 0 {
			fmt.Printf("\r\t# of edges: %d", lineCount)
		}
	}

	fmt.Printf("\r\t# of edges: %d\n", lineCount)
	sn.TotalEdges = lineCount

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading file: %v", err)
	}

	sn.NumVertices = int64(len(sn.VertexKeys))

	fmt.Printf("Signed network loaded:\n")
	fmt.Printf("\t%d vertices\n", sn.NumVertices)
	fmt.Printf("\t%d positive edges\n", sn.NumPositiveEdges)
	fmt.Printf("\t%d negative edges\n", sn.NumNegativeEdges)
	fmt.Printf("\t%.2f%% positive, %.2f%% negative\n",
		float64(sn.NumPositiveEdges)/float64(sn.TotalEdges)*100,
		float64(sn.NumNegativeEdges)/float64(sn.TotalEdges)*100)

	return nil
}

// getOrCreateVertex gets or creates a vertex ID
func (sn *SignedNetwork) getOrCreateVertex(name string) int64 {
	if id, exists := sn.VertexHash[name]; exists {
		return id
	}

	id := int64(len(sn.VertexKeys))
	sn.VertexHash[name] = id
	sn.VertexKeys = append(sn.VertexKeys, name)
	return id
}

// addEdge adds a signed edge to the network
func (sn *SignedNetwork) addEdge(from, to int64, sign, weight float64) {
	if sign > 0 {
		// Positive edge
		sn.PositiveEdges[from] = append(sn.PositiveEdges[from], to)
		sn.PositiveWeights[from] = append(sn.PositiveWeights[from], weight)
		sn.PositiveDegree[from] += weight
		sn.NumPositiveEdges++
	} else {
		// Negative edge
		sn.NegativeEdges[from] = append(sn.NegativeEdges[from], to)
		sn.NegativeWeights[from] = append(sn.NegativeWeights[from], weight)
		sn.NegativeDegree[from] += weight
		sn.NumNegativeEdges++
	}
}

// GetVertexName returns the name of a vertex by ID
func (sn *SignedNetwork) GetVertexName(id int64) string {
	if id < 0 || id >= int64(len(sn.VertexKeys)) {
		return ""
	}
	return sn.VertexKeys[id]
}

// SamplePositiveNeighbor samples a positive neighbor
func (sn *SignedNetwork) SamplePositiveNeighbor(vid int64, rng *rand.Rand) int64 {
	neighbors := sn.PositiveEdges[vid]
	if len(neighbors) == 0 {
		return -1
	}

	weights := sn.PositiveWeights[vid]
	if len(weights) == 0 {
		return neighbors[rng.Intn(len(neighbors))]
	}

	// Weighted sampling
	totalWeight := 0.0
	for _, w := range weights {
		totalWeight += w
	}

	r := rng.Float64() * totalWeight
	cumWeight := 0.0
	for i, w := range weights {
		cumWeight += w
		if r <= cumWeight {
			return neighbors[i]
		}
	}

	return neighbors[len(neighbors)-1]
}

// SampleNegativeNeighbor samples a negative neighbor
func (sn *SignedNetwork) SampleNegativeNeighbor(vid int64, rng *rand.Rand) int64 {
	neighbors := sn.NegativeEdges[vid]
	if len(neighbors) == 0 {
		return -1
	}

	weights := sn.NegativeWeights[vid]
	if len(weights) == 0 {
		return neighbors[rng.Intn(len(neighbors))]
	}

	// Weighted sampling
	totalWeight := 0.0
	for _, w := range weights {
		totalWeight += w
	}

	r := rng.Float64() * totalWeight
	cumWeight := 0.0
	for i, w := range weights {
		cumWeight += w
		if r <= cumWeight {
			return neighbors[i]
		}
	}

	return neighbors[len(neighbors)-1]
}

// SampleVertex samples a random vertex
func (sn *SignedNetwork) SampleVertex(rng *rand.Rand) int64 {
	return rng.Int63n(sn.NumVertices)
}

// HasPositiveEdge checks if there's a positive edge
func (sn *SignedNetwork) HasPositiveEdge(from, to int64) bool {
	neighbors := sn.PositiveEdges[from]
	for _, n := range neighbors {
		if n == to {
			return true
		}
	}
	return false
}

// HasNegativeEdge checks if there's a negative edge
func (sn *SignedNetwork) HasNegativeEdge(from, to int64) bool {
	neighbors := sn.NegativeEdges[from]
	for _, n := range neighbors {
		if n == to {
			return true
		}
	}
	return false
}

// GetNumPositiveNeighbors returns the number of positive neighbors
func (sn *SignedNetwork) GetNumPositiveNeighbors(vid int64) int {
	return len(sn.PositiveEdges[vid])
}

// GetNumNegativeNeighbors returns the number of negative neighbors
func (sn *SignedNetwork) GetNumNegativeNeighbors(vid int64) int {
	return len(sn.NegativeEdges[vid])
}
