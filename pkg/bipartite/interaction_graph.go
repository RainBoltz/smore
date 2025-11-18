package bipartite

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
)

// Interaction represents a user-item interaction with timestamp and optional features
type Interaction struct {
	UserID    int64
	ItemID    int64
	Timestamp float64
	Features  []float64 // Optional interaction features
}

// InteractionGraph represents a bipartite user-item interaction network
type InteractionGraph struct {
	// User and item mappings
	UserHash map[string]int64
	UserKeys []string
	ItemHash map[string]int64
	ItemKeys []string

	// Interactions sorted by timestamp
	Interactions []Interaction

	// Interaction history per user/item
	UserInteractions map[int64][]Interaction
	ItemInteractions map[int64][]Interaction

	// Statistics
	NumUsers       int64
	NumItems       int64
	NumInteractions int64
	FeatureDim     int

	// Time range
	MinTime float64
	MaxTime float64
}

// NewInteractionGraph creates a new bipartite interaction graph
func NewInteractionGraph() *InteractionGraph {
	return &InteractionGraph{
		UserHash:         make(map[string]int64),
		UserKeys:         make([]string, 0),
		ItemHash:         make(map[string]int64),
		ItemKeys:         make([]string, 0),
		Interactions:     make([]Interaction, 0),
		UserInteractions: make(map[int64][]Interaction),
		ItemInteractions: make(map[int64][]Interaction),
		MinTime:          1e9,
		MaxTime:          -1e9,
	}
}

// LoadInteractions loads user-item interactions from file
// Format: user_id item_id timestamp [feature1 feature2 ...]
// Example: "user123 item456 1234567890.5 1.0 0.5"
func (ig *InteractionGraph) LoadInteractions(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open file %s: %v", filename, err)
	}
	defer file.Close()

	fmt.Println("Loading interaction graph from:", filename)

	scanner := bufio.NewScanner(file)
	lineCount := int64(0)
	firstLine := true

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Fields(line)

		if len(parts) < 3 {
			continue
		}

		userName := parts[0]
		itemName := parts[1]

		timestamp, err := strconv.ParseFloat(parts[2], 64)
		if err != nil {
			continue
		}

		// Parse optional features
		features := make([]float64, 0)
		for i := 3; i < len(parts); i++ {
			feat, err := strconv.ParseFloat(parts[i], 64)
			if err == nil {
				features = append(features, feat)
			}
		}

		// Set feature dimension from first interaction
		if firstLine {
			ig.FeatureDim = len(features)
			firstLine = false
		}

		// Get or create user and item
		userID := ig.getOrCreateUser(userName)
		itemID := ig.getOrCreateItem(itemName)

		// Create interaction
		interaction := Interaction{
			UserID:    userID,
			ItemID:    itemID,
			Timestamp: timestamp,
			Features:  features,
		}

		ig.Interactions = append(ig.Interactions, interaction)
		ig.UserInteractions[userID] = append(ig.UserInteractions[userID], interaction)
		ig.ItemInteractions[itemID] = append(ig.ItemInteractions[itemID], interaction)

		// Update time range
		if timestamp < ig.MinTime {
			ig.MinTime = timestamp
		}
		if timestamp > ig.MaxTime {
			ig.MaxTime = timestamp
		}

		lineCount++
		if lineCount%10000 == 0 {
			fmt.Printf("\r\t# of interactions: %d", lineCount)
		}
	}

	fmt.Printf("\r\t# of interactions: %d\n", lineCount)
	ig.NumInteractions = lineCount
	ig.NumUsers = int64(len(ig.UserKeys))
	ig.NumItems = int64(len(ig.ItemKeys))

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading file: %v", err)
	}

	// Sort interactions by timestamp
	ig.sortInteractions()

	fmt.Printf("Interaction graph loaded:\n")
	fmt.Printf("\t%d users\n", ig.NumUsers)
	fmt.Printf("\t%d items\n", ig.NumItems)
	fmt.Printf("\t%d interactions\n", ig.NumInteractions)
	fmt.Printf("\tFeature dimension: %d\n", ig.FeatureDim)
	fmt.Printf("\tTime range: [%.2f, %.2f]\n", ig.MinTime, ig.MaxTime)

	return nil
}

// getOrCreateUser gets or creates a user
func (ig *InteractionGraph) getOrCreateUser(name string) int64 {
	if id, exists := ig.UserHash[name]; exists {
		return id
	}

	id := int64(len(ig.UserKeys))
	ig.UserHash[name] = id
	ig.UserKeys = append(ig.UserKeys, name)

	return id
}

// getOrCreateItem gets or creates an item
func (ig *InteractionGraph) getOrCreateItem(name string) int64 {
	if id, exists := ig.ItemHash[name]; exists {
		return id
	}

	id := int64(len(ig.ItemKeys))
	ig.ItemHash[name] = id
	ig.ItemKeys = append(ig.ItemKeys, name)

	return id
}

// sortInteractions sorts all interactions by timestamp
func (ig *InteractionGraph) sortInteractions() {
	// Sort global interaction list
	sort.Slice(ig.Interactions, func(i, j int) bool {
		return ig.Interactions[i].Timestamp < ig.Interactions[j].Timestamp
	})

	// Sort per-user interactions
	for userID := range ig.UserInteractions {
		sort.Slice(ig.UserInteractions[userID], func(i, j int) bool {
			return ig.UserInteractions[userID][i].Timestamp < ig.UserInteractions[userID][j].Timestamp
		})
	}

	// Sort per-item interactions
	for itemID := range ig.ItemInteractions {
		sort.Slice(ig.ItemInteractions[itemID], func(i, j int) bool {
			return ig.ItemInteractions[itemID][i].Timestamp < ig.ItemInteractions[itemID][j].Timestamp
		})
	}
}

// GetUserName returns the name of a user
func (ig *InteractionGraph) GetUserName(id int64) string {
	if id < 0 || id >= int64(len(ig.UserKeys)) {
		return ""
	}
	return ig.UserKeys[id]
}

// GetItemName returns the name of an item
func (ig *InteractionGraph) GetItemName(id int64) string {
	if id < 0 || id >= int64(len(ig.ItemKeys)) {
		return ""
	}
	return ig.ItemKeys[id]
}

// GetUserInteractionsBefore returns interactions for a user before given time
func (ig *InteractionGraph) GetUserInteractionsBefore(userID int64, timestamp float64) []Interaction {
	interactions := ig.UserInteractions[userID]
	result := make([]Interaction, 0)

	for _, interaction := range interactions {
		if interaction.Timestamp < timestamp {
			result = append(result, interaction)
		} else {
			break // Sorted by timestamp
		}
	}

	return result
}

// GetItemInteractionsBefore returns interactions for an item before given time
func (ig *InteractionGraph) GetItemInteractionsBefore(itemID int64, timestamp float64) []Interaction {
	interactions := ig.ItemInteractions[itemID]
	result := make([]Interaction, 0)

	for _, interaction := range interactions {
		if interaction.Timestamp < timestamp {
			result = append(result, interaction)
		} else {
			break // Sorted by timestamp
		}
	}

	return result
}

// GetLastUserInteraction returns the most recent interaction for a user before given time
func (ig *InteractionGraph) GetLastUserInteraction(userID int64, timestamp float64) (Interaction, bool) {
	interactions := ig.GetUserInteractionsBefore(userID, timestamp)
	if len(interactions) == 0 {
		return Interaction{}, false
	}
	return interactions[len(interactions)-1], true
}

// GetLastItemInteraction returns the most recent interaction for an item before given time
func (ig *InteractionGraph) GetLastItemInteraction(itemID int64, timestamp float64) (Interaction, bool) {
	interactions := ig.GetItemInteractionsBefore(itemID, timestamp)
	if len(interactions) == 0 {
		return Interaction{}, false
	}
	return interactions[len(interactions)-1], true
}

// ComputeStatistics computes and displays interaction graph statistics
func (ig *InteractionGraph) ComputeStatistics() {
	fmt.Println("\nInteraction Graph Statistics:")

	// User activity
	totalUserInteractions := 0
	maxUserInteractions := 0
	for _, interactions := range ig.UserInteractions {
		count := len(interactions)
		totalUserInteractions += count
		if count > maxUserInteractions {
			maxUserInteractions = count
		}
	}
	avgUserInteractions := float64(totalUserInteractions) / float64(ig.NumUsers)

	// Item activity
	totalItemInteractions := 0
	maxItemInteractions := 0
	for _, interactions := range ig.ItemInteractions {
		count := len(interactions)
		totalItemInteractions += count
		if count > maxItemInteractions {
			maxItemInteractions = count
		}
	}
	avgItemInteractions := float64(totalItemInteractions) / float64(ig.NumItems)

	fmt.Printf("\tAvg interactions per user: %.2f (max: %d)\n", avgUserInteractions, maxUserInteractions)
	fmt.Printf("\tAvg interactions per item: %.2f (max: %d)\n", avgItemInteractions, maxItemInteractions)

	// Time statistics
	timeSpan := ig.MaxTime - ig.MinTime
	if timeSpan > 0 {
		interactionRate := float64(ig.NumInteractions) / timeSpan
		fmt.Printf("\tInteractions per time unit: %.2f\n", interactionRate)
	}
}
