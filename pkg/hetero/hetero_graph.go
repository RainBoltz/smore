package hetero

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
)

// HeteroGraph represents a heterogeneous graph with typed nodes and edges
type HeteroGraph struct {
	// Node mapping: node_name -> node_id
	NodeHash map[string]int64
	NodeKeys []string

	// Node types: node_id -> type_name
	NodeTypes map[int64]string

	// Type mapping: type_name -> type_id
	TypeHash map[string]int64
	TypeKeys []string

	// Adjacency list by edge type: source_node -> list of (target_node, edge_type)
	Edges map[int64][]int64
	EdgeTypes map[int64][]string
	EdgeWeights map[int64][]float64

	// Statistics
	NumNodes int64
	NumEdges int64
	NumTypes int64

	// Type-specific indices for efficient meta-path walks
	NodesByType map[string][]int64           // type -> list of nodes
	NeighborsByType map[int64]map[string][]int64 // node -> type -> neighbors
}

// NewHeteroGraph creates a new heterogeneous graph
func NewHeteroGraph() *HeteroGraph {
	return &HeteroGraph{
		NodeHash:        make(map[string]int64),
		NodeKeys:        make([]string, 0),
		NodeTypes:       make(map[int64]string),
		TypeHash:        make(map[string]int64),
		TypeKeys:        make([]string, 0),
		Edges:           make(map[int64][]int64),
		EdgeTypes:       make(map[int64][]string),
		EdgeWeights:     make(map[int64][]float64),
		NodesByType:     make(map[string][]int64),
		NeighborsByType: make(map[int64]map[string][]int64),
	}
}

// LoadEdgeList loads heterogeneous graph from edge list file
// Format: source_node source_type target_node target_type edge_type [weight]
// Example: "Alice User Bob User friend 1.0"
// Example: "Alice User iPhone Product purchased 1.0"
func (hg *HeteroGraph) LoadEdgeList(filename string, undirected bool) error {
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open file %s: %v", filename, err)
	}
	defer file.Close()

	fmt.Println("Loading heterogeneous graph from:", filename)

	scanner := bufio.NewScanner(file)
	lineCount := int64(0)

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Fields(line)

		if len(parts) < 5 {
			continue
		}

		sourceName := parts[0]
		sourceType := parts[1]
		targetName := parts[2]
		targetType := parts[3]
		edgeType := parts[4]

		weight := 1.0
		if len(parts) >= 6 {
			w, err := strconv.ParseFloat(parts[5], 64)
			if err == nil {
				weight = w
			}
		}

		// Get or create nodes
		sourceID := hg.getOrCreateNode(sourceName, sourceType)
		targetID := hg.getOrCreateNode(targetName, targetType)

		// Add edge
		hg.addEdge(sourceID, targetID, edgeType, weight)

		if undirected {
			// For undirected edges, add reverse edge with same type
			hg.addEdge(targetID, sourceID, edgeType, weight)
		}

		lineCount++
		if lineCount%10000 == 0 {
			fmt.Printf("\r\t# of edges: %d", lineCount)
		}
	}

	fmt.Printf("\r\t# of edges: %d\n", lineCount)
	hg.NumEdges = lineCount
	hg.NumNodes = int64(len(hg.NodeKeys))
	hg.NumTypes = int64(len(hg.TypeKeys))

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading file: %v", err)
	}

	// Build type-specific indices
	hg.buildTypeIndices()

	fmt.Printf("Heterogeneous graph loaded:\n")
	fmt.Printf("\t%d nodes\n", hg.NumNodes)
	fmt.Printf("\t%d edges\n", hg.NumEdges)
	fmt.Printf("\t%d node types:", hg.NumTypes)
	for _, typeName := range hg.TypeKeys {
		count := len(hg.NodesByType[typeName])
		fmt.Printf(" %s(%d)", typeName, count)
	}
	fmt.Println()

	return nil
}

// getOrCreateNode gets or creates a node with given type
func (hg *HeteroGraph) getOrCreateNode(name, nodeType string) int64 {
	if id, exists := hg.NodeHash[name]; exists {
		return id
	}

	id := int64(len(hg.NodeKeys))
	hg.NodeHash[name] = id
	hg.NodeKeys = append(hg.NodeKeys, name)
	hg.NodeTypes[id] = nodeType

	// Register type
	if _, exists := hg.TypeHash[nodeType]; !exists {
		typeID := int64(len(hg.TypeKeys))
		hg.TypeHash[nodeType] = typeID
		hg.TypeKeys = append(hg.TypeKeys, nodeType)
		hg.NodesByType[nodeType] = make([]int64, 0)
	}

	hg.NodesByType[nodeType] = append(hg.NodesByType[nodeType], id)

	return id
}

// addEdge adds an edge with type
func (hg *HeteroGraph) addEdge(from, to int64, edgeType string, weight float64) {
	hg.Edges[from] = append(hg.Edges[from], to)
	hg.EdgeTypes[from] = append(hg.EdgeTypes[from], edgeType)
	hg.EdgeWeights[from] = append(hg.EdgeWeights[from], weight)
}

// buildTypeIndices builds type-specific neighbor indices
func (hg *HeteroGraph) buildTypeIndices() {
	for nodeID := int64(0); nodeID < hg.NumNodes; nodeID++ {
		neighbors := hg.Edges[nodeID]
		hg.NeighborsByType[nodeID] = make(map[string][]int64)

		for i, neighbor := range neighbors {
			neighborType := hg.NodeTypes[neighbor]
			hg.NeighborsByType[nodeID][neighborType] = append(
				hg.NeighborsByType[nodeID][neighborType],
				int64(i), // Store index into Edges array
			)
		}
	}
}

// GetNodeName returns the name of a node
func (hg *HeteroGraph) GetNodeName(id int64) string {
	if id < 0 || id >= int64(len(hg.NodeKeys)) {
		return ""
	}
	return hg.NodeKeys[id]
}

// GetNodeType returns the type of a node
func (hg *HeteroGraph) GetNodeType(id int64) string {
	return hg.NodeTypes[id]
}

// SampleNodeByType samples a random node of given type
func (hg *HeteroGraph) SampleNodeByType(nodeType string, rng *rand.Rand) int64 {
	nodes := hg.NodesByType[nodeType]
	if len(nodes) == 0 {
		return -1
	}
	return nodes[rng.Intn(len(nodes))]
}

// SampleNeighborByType samples a neighbor of given type
func (hg *HeteroGraph) SampleNeighborByType(nodeID int64, targetType string, rng *rand.Rand) int64 {
	// Get indices of neighbors with target type
	indices := hg.NeighborsByType[nodeID][targetType]
	if len(indices) == 0 {
		return -1
	}

	// Sample uniformly (could be extended to weighted sampling)
	idx := indices[rng.Intn(len(indices))]
	return hg.Edges[nodeID][idx]
}

// MetaPathWalk performs a meta-path-based random walk
// metaPath: sequence of node types (e.g., ["User", "Item", "Category", "Item", "User"])
func (hg *HeteroGraph) MetaPathWalk(startNode int64, metaPath []string, steps int, rng *rand.Rand) []int64 {
	if len(metaPath) < 2 {
		return []int64{startNode}
	}

	walk := make([]int64, 0, steps+1)
	walk = append(walk, startNode)

	current := startNode
	pathIdx := 0

	for len(walk) < steps+1 {
		// Current type in path
		currentType := metaPath[pathIdx%len(metaPath)]
		// Next type in path
		nextType := metaPath[(pathIdx+1)%len(metaPath)]

		// Verify current node has correct type
		if hg.GetNodeType(current) != currentType {
			// Type mismatch, stop walk
			break
		}

		// Sample neighbor of next type
		next := hg.SampleNeighborByType(current, nextType, rng)
		if next == -1 {
			// No neighbor of target type, stop walk
			break
		}

		walk = append(walk, next)
		current = next
		pathIdx++
	}

	return walk
}

// ValidateMetaPath checks if a meta-path is valid for the graph
func (hg *HeteroGraph) ValidateMetaPath(metaPath []string) error {
	if len(metaPath) < 2 {
		return fmt.Errorf("meta-path must have at least 2 types")
	}

	for _, typeName := range metaPath {
		if _, exists := hg.TypeHash[typeName]; !exists {
			return fmt.Errorf("unknown node type in meta-path: %s", typeName)
		}
	}

	return nil
}

// GetNeighborCount returns the number of neighbors of a given type
func (hg *HeteroGraph) GetNeighborCount(nodeID int64, nodeType string) int {
	indices := hg.NeighborsByType[nodeID][nodeType]
	return len(indices)
}
