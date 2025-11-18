package pronet

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"sync"
)

const (
	Monitor           = 10000
	PowerSample       = 0.75
	SigmoidTableSize  = 1000
	MaxSigmoid        = 8.0
)

// Vertex represents a vertex in the network
type Vertex struct {
	Offset     int64
	Branch     int64
	OutDegree  float64
	InDegree   float64
}

// Context represents a context vertex
type Context struct {
	Vid      int64
	InDegree float64
}

// AliasTable for efficient weighted sampling
type AliasTable struct {
	Alias int64
	Prob  float64
}

// ProNet is the core network embedding framework
type ProNet struct {
	// Vertex and edge data
	Vertices      []Vertex
	Contexts      []Context
	VertexAT      []AliasTable
	ContextAT     []AliasTable
	NegativeAT    []AliasTable

	// Hash tables for vertex name mapping
	VertexHash    map[string]int64
	VertexKeys    []string

	// Cached sigmoid table for performance
	CachedSigmoid []float64

	// Graph structure (adjacency list)
	Graph         map[int64][]int64
	EdgeWeights   map[int64][]float64

	// Statistics
	MaxVid        int64
	MaxLine       int64

	mu            sync.RWMutex
}

// NewProNet creates a new ProNet instance
func NewProNet() *ProNet {
	pn := &ProNet{
		VertexHash:    make(map[string]int64),
		VertexKeys:    make([]string, 0),
		CachedSigmoid: make([]float64, SigmoidTableSize+1),
		Graph:         make(map[int64][]int64),
		EdgeWeights:   make(map[int64][]float64),
	}
	pn.initSigmoid()
	return pn
}

// initSigmoid initializes the sigmoid lookup table
func (pn *ProNet) initSigmoid() {
	for i := 0; i <= SigmoidTableSize; i++ {
		x := float64(i)*2.0*MaxSigmoid/float64(SigmoidTableSize) - MaxSigmoid
		pn.CachedSigmoid[i] = 1.0 / (1.0 + math.Exp(-x))
	}
}

// FastSigmoid returns sigmoid using lookup table for performance
func (pn *ProNet) FastSigmoid(x float64) float64 {
	if x < -MaxSigmoid {
		return 0.0
	} else if x > MaxSigmoid {
		return 1.0
	}
	idx := int((x + MaxSigmoid) * float64(SigmoidTableSize) / MaxSigmoid / 2.0)
	if idx >= len(pn.CachedSigmoid) {
		idx = len(pn.CachedSigmoid) - 1
	}
	return pn.CachedSigmoid[idx]
}

// LoadEdgeList loads the network from edge list file
func (pn *ProNet) LoadEdgeList(filename string, undirected bool) error {
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open file %s: %v", filename, err)
	}
	defer file.Close()

	fmt.Println("Loading network from:", filename)

	// Temporary structures to build the graph
	graph := make(map[int64][]int64)
	edge := make(map[int64][]float64)

	scanner := bufio.NewScanner(file)
	lineCount := int64(0)

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Fields(line)

		if len(parts) < 3 {
			continue
		}

		v1 := parts[0]
		v2 := parts[1]
		weight, err := strconv.ParseFloat(parts[2], 64)
		if err != nil {
			fmt.Printf("Warning: invalid weight at line %d\n", lineCount)
			continue
		}

		// Get or create vertex IDs
		vid1 := pn.getOrCreateVertex(v1)
		vid2 := pn.getOrCreateVertex(v2)

		// Add edges
		graph[vid1] = append(graph[vid1], vid2)
		edge[vid1] = append(edge[vid1], weight)

		if undirected {
			graph[vid2] = append(graph[vid2], vid1)
			edge[vid2] = append(edge[vid2], weight)
		}

		lineCount++
		if lineCount%Monitor == 0 {
			fmt.Printf("\r\t# of connections: %d", lineCount)
		}
	}

	fmt.Printf("\r\t# of connections: %d\n", lineCount)
	pn.MaxLine = lineCount

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading file: %v", err)
	}

	// Build vertex and context structures
	pn.buildGraph(graph, edge, undirected)

	return nil
}

// getOrCreateVertex gets or creates a vertex ID
func (pn *ProNet) getOrCreateVertex(name string) int64 {
	if vid, exists := pn.VertexHash[name]; exists {
		return vid
	}

	vid := int64(len(pn.VertexKeys))
	pn.VertexHash[name] = vid
	pn.VertexKeys = append(pn.VertexKeys, name)
	pn.MaxVid = vid + 1

	return vid
}

// buildGraph builds the graph structures and alias tables
func (pn *ProNet) buildGraph(graph map[int64][]int64, edge map[int64][]float64, undirected bool) {
	fmt.Println("Building graph structures...")

	pn.Graph = graph
	pn.EdgeWeights = edge

	// Initialize vertices
	pn.Vertices = make([]Vertex, pn.MaxVid)
	pn.Contexts = make([]Context, pn.MaxVid)

	// Calculate degrees
	for vid := int64(0); vid < pn.MaxVid; vid++ {
		if neighbors, exists := graph[vid]; exists {
			weights := edge[vid]
			for i, nid := range neighbors {
				w := weights[i]
				pn.Vertices[vid].OutDegree += w
				pn.Contexts[nid].InDegree += w
				pn.Vertices[nid].InDegree += w
			}
		}
		pn.Contexts[vid].Vid = vid
	}

	// Build alias tables for efficient sampling
	pn.buildVertexAliasTable()
	pn.buildContextAliasTable()
	pn.buildNegativeAliasTable()

	fmt.Printf("Graph loaded: %d vertices, %d edges\n", pn.MaxVid, pn.MaxLine)
}

// buildVertexAliasTable builds alias table for vertex sampling
func (pn *ProNet) buildVertexAliasTable() {
	distribution := make([]float64, pn.MaxVid)
	for i := int64(0); i < pn.MaxVid; i++ {
		distribution[i] = pn.Vertices[i].OutDegree
	}
	pn.VertexAT = BuildAliasMethod(distribution, 1.0)
}

// buildContextAliasTable builds alias table for context sampling
func (pn *ProNet) buildContextAliasTable() {
	distribution := make([]float64, pn.MaxVid)
	for i := int64(0); i < pn.MaxVid; i++ {
		distribution[i] = pn.Contexts[i].InDegree
	}
	pn.ContextAT = BuildAliasMethod(distribution, 1.0)
}

// buildNegativeAliasTable builds alias table for negative sampling
func (pn *ProNet) buildNegativeAliasTable() {
	distribution := make([]float64, pn.MaxVid)
	for i := int64(0); i < pn.MaxVid; i++ {
		degree := pn.Vertices[i].InDegree + pn.Vertices[i].OutDegree
		distribution[i] = degree
	}
	pn.NegativeAT = BuildAliasMethod(distribution, PowerSample)
}

// SourceSample samples a source vertex
func (pn *ProNet) SourceSample(rng *rand.Rand) int64 {
	return aliasSample(pn.VertexAT, rng)
}

// TargetSample samples a target vertex from a source vertex's neighbors
func (pn *ProNet) TargetSample(vid int64, rng *rand.Rand) int64 {
	neighbors := pn.Graph[vid]
	if len(neighbors) == 0 {
		return -1
	}

	weights := pn.EdgeWeights[vid]
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

// NegativeSample samples a negative vertex
func (pn *ProNet) NegativeSample(rng *rand.Rand) int64 {
	return aliasSample(pn.NegativeAT, rng)
}

// RandomWalk performs a random walk starting from vid
func (pn *ProNet) RandomWalk(vid int64, steps int, rng *rand.Rand) []int64 {
	walk := make([]int64, 0, steps+1)
	walk = append(walk, vid)

	current := vid
	for i := 0; i < steps; i++ {
		next := pn.TargetSample(current, rng)
		if next == -1 {
			break
		}
		walk = append(walk, next)
		current = next
	}

	return walk
}

// SkipGrams generates skip-gram training pairs from a walk
func (pn *ProNet) SkipGrams(walk []int64, windowSize int) ([]int64, []int64) {
	vertices := make([]int64, 0)
	contexts := make([]int64, 0)

	for i := 0; i < len(walk); i++ {
		start := i - windowSize
		if start < 0 {
			start = 0
		}
		end := i + windowSize + 1
		if end > len(walk) {
			end = len(walk)
		}

		for j := start; j < end; j++ {
			if i != j {
				vertices = append(vertices, walk[i])
				contexts = append(contexts, walk[j])
			}
		}
	}

	return vertices, contexts
}

// GetVertexName returns the name of a vertex by ID
func (pn *ProNet) GetVertexName(vid int64) string {
	if vid < 0 || vid >= int64(len(pn.VertexKeys)) {
		return ""
	}
	return pn.VertexKeys[vid]
}
