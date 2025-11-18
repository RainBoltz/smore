package temporal

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
)

// TemporalEdge represents an edge with a timestamp
type TemporalEdge struct {
	From      int64
	To        int64
	Timestamp float64
}

// TemporalGraph represents a time-evolving graph
type TemporalGraph struct {
	// Node mapping
	NodeHash map[string]int64
	NodeKeys []string

	// Edges sorted by timestamp
	Edges []TemporalEdge

	// Adjacency list: node -> list of (neighbor, timestamp) pairs
	// Sorted by timestamp for efficient temporal queries
	OutEdges map[int64][]TemporalEdge
	InEdges  map[int64][]TemporalEdge

	// Statistics
	NumNodes int64
	NumEdges int64

	// Time range
	MinTime float64
	MaxTime float64
}

// NewTemporalGraph creates a new temporal graph
func NewTemporalGraph() *TemporalGraph {
	return &TemporalGraph{
		NodeHash: make(map[string]int64),
		NodeKeys: make([]string, 0),
		Edges:    make([]TemporalEdge, 0),
		OutEdges: make(map[int64][]TemporalEdge),
		InEdges:  make(map[int64][]TemporalEdge),
		MinTime:  math.MaxFloat64,
		MaxTime:  -math.MaxFloat64,
	}
}

// LoadEdgeList loads temporal graph from edge list file
// Format: source_node target_node timestamp
// Example: "Alice Bob 1234567890.5"
func (tg *TemporalGraph) LoadEdgeList(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open file %s: %v", filename, err)
	}
	defer file.Close()

	fmt.Println("Loading temporal graph from:", filename)

	scanner := bufio.NewScanner(file)
	lineCount := int64(0)

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Fields(line)

		if len(parts) < 3 {
			continue
		}

		sourceName := parts[0]
		targetName := parts[1]

		timestamp, err := strconv.ParseFloat(parts[2], 64)
		if err != nil {
			continue
		}

		// Get or create nodes
		sourceID := tg.getOrCreateNode(sourceName)
		targetID := tg.getOrCreateNode(targetName)

		// Add temporal edge
		edge := TemporalEdge{
			From:      sourceID,
			To:        targetID,
			Timestamp: timestamp,
		}

		tg.Edges = append(tg.Edges, edge)
		tg.OutEdges[sourceID] = append(tg.OutEdges[sourceID], edge)
		tg.InEdges[targetID] = append(tg.InEdges[targetID], edge)

		// Update time range
		if timestamp < tg.MinTime {
			tg.MinTime = timestamp
		}
		if timestamp > tg.MaxTime {
			tg.MaxTime = timestamp
		}

		lineCount++
		if lineCount%10000 == 0 {
			fmt.Printf("\r\t# of edges: %d", lineCount)
		}
	}

	fmt.Printf("\r\t# of edges: %d\n", lineCount)
	tg.NumEdges = lineCount
	tg.NumNodes = int64(len(tg.NodeKeys))

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading file: %v", err)
	}

	// Sort edges by timestamp
	tg.sortEdges()

	fmt.Printf("Temporal graph loaded:\n")
	fmt.Printf("\t%d nodes\n", tg.NumNodes)
	fmt.Printf("\t%d edges\n", tg.NumEdges)
	fmt.Printf("\tTime range: [%.2f, %.2f]\n", tg.MinTime, tg.MaxTime)

	return nil
}

// getOrCreateNode gets or creates a node
func (tg *TemporalGraph) getOrCreateNode(name string) int64 {
	if id, exists := tg.NodeHash[name]; exists {
		return id
	}

	id := int64(len(tg.NodeKeys))
	tg.NodeHash[name] = id
	tg.NodeKeys = append(tg.NodeKeys, name)

	return id
}

// sortEdges sorts all edge lists by timestamp
func (tg *TemporalGraph) sortEdges() {
	// Sort global edge list
	sort.Slice(tg.Edges, func(i, j int) bool {
		return tg.Edges[i].Timestamp < tg.Edges[j].Timestamp
	})

	// Sort outgoing edges for each node
	for nodeID := range tg.OutEdges {
		sort.Slice(tg.OutEdges[nodeID], func(i, j int) bool {
			return tg.OutEdges[nodeID][i].Timestamp < tg.OutEdges[nodeID][j].Timestamp
		})
	}

	// Sort incoming edges for each node
	for nodeID := range tg.InEdges {
		sort.Slice(tg.InEdges[nodeID], func(i, j int) bool {
			return tg.InEdges[nodeID][i].Timestamp < tg.InEdges[nodeID][j].Timestamp
		})
	}
}

// GetNodeName returns the name of a node
func (tg *TemporalGraph) GetNodeName(id int64) string {
	if id < 0 || id >= int64(len(tg.NodeKeys)) {
		return ""
	}
	return tg.NodeKeys[id]
}

// GetTemporalNeighbors returns neighbors within a time window
// Returns neighbors that connected after startTime but before endTime
func (tg *TemporalGraph) GetTemporalNeighbors(nodeID int64, startTime, endTime float64) []int64 {
	edges := tg.OutEdges[nodeID]
	neighbors := make([]int64, 0)

	for _, edge := range edges {
		if edge.Timestamp >= startTime && edge.Timestamp <= endTime {
			neighbors = append(neighbors, edge.To)
		}
		if edge.Timestamp > endTime {
			break // Edges are sorted by time
		}
	}

	return neighbors
}

// SampleTemporalNeighbor samples a random neighbor within time window
func (tg *TemporalGraph) SampleTemporalNeighbor(nodeID int64, startTime, endTime float64, rng *rand.Rand) (int64, float64) {
	neighbors := tg.GetTemporalNeighbors(nodeID, startTime, endTime)

	if len(neighbors) == 0 {
		return -1, 0
	}

	idx := rng.Intn(len(neighbors))
	return neighbors[idx], tg.OutEdges[nodeID][idx].Timestamp
}

// GetEdgeTime returns the timestamp of an edge (most recent if multiple exist)
func (tg *TemporalGraph) GetEdgeTime(from, to int64) float64 {
	edges := tg.OutEdges[from]

	// Search backwards for most recent edge
	for i := len(edges) - 1; i >= 0; i-- {
		if edges[i].To == to {
			return edges[i].Timestamp
		}
	}

	return -1
}

// TemporalRandomWalk performs a time-constrained random walk
// Each step must occur after the previous step's timestamp
func (tg *TemporalGraph) TemporalRandomWalk(startNode int64, startTime float64, steps int, timeWindow float64, rng *rand.Rand) []int64 {
	walk := make([]int64, 0, steps+1)
	walk = append(walk, startNode)

	current := startNode
	currentTime := startTime

	for len(walk) < steps+1 {
		// Get neighbors that appear after currentTime within the time window
		endTime := currentTime + timeWindow
		if endTime > tg.MaxTime {
			endTime = tg.MaxTime
		}

		neighbor, timestamp := tg.SampleTemporalNeighbor(current, currentTime, endTime, rng)
		if neighbor == -1 {
			// No valid neighbors, stop walk
			break
		}

		walk = append(walk, neighbor)
		current = neighbor
		currentTime = timestamp
	}

	return walk
}

// GetNodeActivity returns the activity level of a node (number of edges)
func (tg *TemporalGraph) GetNodeActivity(nodeID int64) int {
	return len(tg.OutEdges[nodeID]) + len(tg.InEdges[nodeID])
}

// GetActiveTimeRange returns the time range when a node was active
func (tg *TemporalGraph) GetActiveTimeRange(nodeID int64) (float64, float64) {
	minTime := math.MaxFloat64
	maxTime := -math.MaxFloat64

	// Check outgoing edges
	for _, edge := range tg.OutEdges[nodeID] {
		if edge.Timestamp < minTime {
			minTime = edge.Timestamp
		}
		if edge.Timestamp > maxTime {
			maxTime = edge.Timestamp
		}
	}

	// Check incoming edges
	for _, edge := range tg.InEdges[nodeID] {
		if edge.Timestamp < minTime {
			minTime = edge.Timestamp
		}
		if edge.Timestamp > maxTime {
			maxTime = edge.Timestamp
		}
	}

	if minTime == math.MaxFloat64 {
		return 0, 0
	}

	return minTime, maxTime
}

// ComputeTemporalStatistics computes temporal graph statistics
func (tg *TemporalGraph) ComputeTemporalStatistics() {
	fmt.Println("\nTemporal Graph Statistics:")

	// Compute average degree
	totalDegree := 0
	for i := int64(0); i < tg.NumNodes; i++ {
		totalDegree += tg.GetNodeActivity(i)
	}
	avgDegree := float64(totalDegree) / float64(tg.NumNodes)
	fmt.Printf("\tAverage degree: %.2f\n", avgDegree)

	// Compute time span
	timeSpan := tg.MaxTime - tg.MinTime
	fmt.Printf("\tTime span: %.2f units\n", timeSpan)

	// Compute edge density over time
	if timeSpan > 0 {
		edgesPerTime := float64(tg.NumEdges) / timeSpan
		fmt.Printf("\tEdges per time unit: %.2f\n", edgesPerTime)
	}

	// Sample some active nodes
	fmt.Println("\nSample Node Activity:")
	samples := 5
	if samples > int(tg.NumNodes) {
		samples = int(tg.NumNodes)
	}

	for i := 0; i < samples; i++ {
		nodeID := rand.Int63n(tg.NumNodes)
		name := tg.GetNodeName(nodeID)
		activity := tg.GetNodeActivity(nodeID)
		minTime, maxTime := tg.GetActiveTimeRange(nodeID)

		fmt.Printf("\tNode %s: %d edges, active [%.2f, %.2f]\n",
			name, activity, minTime, maxTime)
	}
}
