package knowledge

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
)

// Triple represents a knowledge graph triple (head, relation, tail)
type Triple struct {
	Head     int64
	Relation int64
	Tail     int64
	Weight   float64
}

// KnowledgeGraph represents a knowledge graph with entities and relations
type KnowledgeGraph struct {
	// Entity and relation mappings
	EntityHash    map[string]int64
	RelationHash  map[string]int64
	EntityKeys    []string
	RelationKeys  []string

	// Triples
	Triples []Triple

	// Statistics
	NumEntities  int64
	NumRelations int64
	NumTriples   int64

	// Indexed structures for efficient sampling
	HeadIndex     map[int64][]int64 // entity -> list of triple indices where it's head
	TailIndex     map[int64][]int64 // entity -> list of triple indices where it's tail
	RelationIndex map[int64][]int64 // relation -> list of triple indices

	// For negative sampling: entities per relation
	EntitiesPerRelation map[int64]map[int64]bool // relation -> set of entities
}

// NewKnowledgeGraph creates a new knowledge graph instance
func NewKnowledgeGraph() *KnowledgeGraph {
	return &KnowledgeGraph{
		EntityHash:          make(map[string]int64),
		RelationHash:        make(map[string]int64),
		EntityKeys:          make([]string, 0),
		RelationKeys:        make([]string, 0),
		Triples:             make([]Triple, 0),
		HeadIndex:           make(map[int64][]int64),
		TailIndex:           make(map[int64][]int64),
		RelationIndex:       make(map[int64][]int64),
		EntitiesPerRelation: make(map[int64]map[int64]bool),
	}
}

// LoadTriples loads knowledge graph triples from a file
// Format: head relation tail [weight]
// Example: "Barack_Obama born_in Hawaii 1.0"
func (kg *KnowledgeGraph) LoadTriples(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open file %s: %v", filename, err)
	}
	defer file.Close()

	fmt.Println("Loading knowledge graph from:", filename)

	scanner := bufio.NewScanner(file)
	lineCount := int64(0)

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Fields(line)

		if len(parts) < 3 {
			continue
		}

		head := parts[0]
		relation := parts[1]
		tail := parts[2]

		weight := 1.0
		if len(parts) >= 4 {
			w, err := strconv.ParseFloat(parts[3], 64)
			if err == nil {
				weight = w
			}
		}

		// Get or create IDs
		headID := kg.getOrCreateEntity(head)
		relationID := kg.getOrCreateRelation(relation)
		tailID := kg.getOrCreateEntity(tail)

		// Add triple
		tripleIdx := int64(len(kg.Triples))
		kg.Triples = append(kg.Triples, Triple{
			Head:     headID,
			Relation: relationID,
			Tail:     tailID,
			Weight:   weight,
		})

		// Update indices
		kg.HeadIndex[headID] = append(kg.HeadIndex[headID], tripleIdx)
		kg.TailIndex[tailID] = append(kg.TailIndex[tailID], tripleIdx)
		kg.RelationIndex[relationID] = append(kg.RelationIndex[relationID], tripleIdx)

		// Track entities per relation
		if kg.EntitiesPerRelation[relationID] == nil {
			kg.EntitiesPerRelation[relationID] = make(map[int64]bool)
		}
		kg.EntitiesPerRelation[relationID][headID] = true
		kg.EntitiesPerRelation[relationID][tailID] = true

		lineCount++
		if lineCount%10000 == 0 {
			fmt.Printf("\r\t# of triples: %d", lineCount)
		}
	}

	fmt.Printf("\r\t# of triples: %d\n", lineCount)
	kg.NumTriples = lineCount
	kg.NumEntities = int64(len(kg.EntityKeys))
	kg.NumRelations = int64(len(kg.RelationKeys))

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading file: %v", err)
	}

	fmt.Printf("Knowledge graph loaded:\n")
	fmt.Printf("\t%d entities\n", kg.NumEntities)
	fmt.Printf("\t%d relations\n", kg.NumRelations)
	fmt.Printf("\t%d triples\n", kg.NumTriples)

	return nil
}

// getOrCreateEntity gets or creates an entity ID
func (kg *KnowledgeGraph) getOrCreateEntity(name string) int64 {
	if id, exists := kg.EntityHash[name]; exists {
		return id
	}

	id := int64(len(kg.EntityKeys))
	kg.EntityHash[name] = id
	kg.EntityKeys = append(kg.EntityKeys, name)
	return id
}

// getOrCreateRelation gets or creates a relation ID
func (kg *KnowledgeGraph) getOrCreateRelation(name string) int64 {
	if id, exists := kg.RelationHash[name]; exists {
		return id
	}

	id := int64(len(kg.RelationKeys))
	kg.RelationHash[name] = id
	kg.RelationKeys = append(kg.RelationKeys, name)
	return id
}

// GetEntityName returns the name of an entity by ID
func (kg *KnowledgeGraph) GetEntityName(id int64) string {
	if id < 0 || id >= int64(len(kg.EntityKeys)) {
		return ""
	}
	return kg.EntityKeys[id]
}

// GetRelationName returns the name of a relation by ID
func (kg *KnowledgeGraph) GetRelationName(id int64) string {
	if id < 0 || id >= int64(len(kg.RelationKeys)) {
		return ""
	}
	return kg.RelationKeys[id]
}

// SampleNegativeHead samples a random entity to replace the head
func (kg *KnowledgeGraph) SampleNegativeHead(triple Triple, rng *rand.Rand) int64 {
	// Sample from entities that appear in this relation
	if entities, ok := kg.EntitiesPerRelation[triple.Relation]; ok && len(entities) > 0 {
		// Convert set to slice
		entityList := make([]int64, 0, len(entities))
		for e := range entities {
			entityList = append(entityList, e)
		}
		return entityList[rng.Intn(len(entityList))]
	}
	// Fallback: random entity
	return rng.Int63n(kg.NumEntities)
}

// SampleNegativeTail samples a random entity to replace the tail
func (kg *KnowledgeGraph) SampleNegativeTail(triple Triple, rng *rand.Rand) int64 {
	// Sample from entities that appear in this relation
	if entities, ok := kg.EntitiesPerRelation[triple.Relation]; ok && len(entities) > 0 {
		// Convert set to slice
		entityList := make([]int64, 0, len(entities))
		for e := range entities {
			entityList = append(entityList, e)
		}
		return entityList[rng.Intn(len(entityList))]
	}
	// Fallback: random entity
	return rng.Int63n(kg.NumEntities)
}

// GetTriple returns the triple at the given index
func (kg *KnowledgeGraph) GetTriple(idx int64) Triple {
	if idx < 0 || idx >= int64(len(kg.Triples)) {
		return Triple{}
	}
	return kg.Triples[idx]
}
