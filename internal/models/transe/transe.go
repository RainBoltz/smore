package transe

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/cnclabs/smore/pkg/knowledge"
)

// TransE implements the TransE (Translating Embeddings) algorithm
// TransE models relations as translations in the embedding space: h + r ≈ t
// where h is head entity, r is relation, t is tail entity
type TransE struct {
	kg *knowledge.KnowledgeGraph
	dim int

	// Embeddings
	entityEmbeddings   [][]float64 // Entity embeddings
	relationEmbeddings [][]float64 // Relation embeddings

	// TransE parameters
	margin     float64 // Margin for ranking loss (default: 1.0)
	norm       int     // L1 or L2 norm (1 or 2, default: 2)
}

// New creates a new TransE instance
func New() *TransE {
	return &TransE{
		kg:     knowledge.NewKnowledgeGraph(),
		margin: 1.0,
		norm:   2,
	}
}

// LoadTriples loads the knowledge graph from a triples file
func (te *TransE) LoadTriples(filename string) error {
	return te.kg.LoadTriples(filename)
}

// Init initializes the model with given dimensions and parameters
func (te *TransE) Init(dim int, margin float64, norm int) {
	te.dim = dim
	te.margin = margin
	te.norm = norm

	fmt.Println("Model Setting:")
	fmt.Printf("\tdimension:\t\t%d\n", dim)
	fmt.Printf("\tmargin:\t\t\t%.2f\n", margin)
	if norm == 1 {
		fmt.Printf("\tnorm:\t\t\tL1 (Manhattan)\n")
	} else {
		fmt.Printf("\tnorm:\t\t\tL2 (Euclidean)\n")
	}
	fmt.Println()
	fmt.Println("TransE Principle:")
	fmt.Println("\th + r ≈ t")
	fmt.Println("\t(head + relation ≈ tail in embedding space)")

	// Initialize entity embeddings with random values
	te.entityEmbeddings = make([][]float64, te.kg.NumEntities)
	for i := int64(0); i < te.kg.NumEntities; i++ {
		te.entityEmbeddings[i] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			te.entityEmbeddings[i][d] = (rand.Float64() - 0.5) / float64(dim)
		}
		// L2 normalize entity embeddings
		te.normalizeEntity(i)
	}

	// Initialize relation embeddings with random values
	te.relationEmbeddings = make([][]float64, te.kg.NumRelations)
	for i := int64(0); i < te.kg.NumRelations; i++ {
		te.relationEmbeddings[i] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			te.relationEmbeddings[i][d] = (rand.Float64() - 0.5) / float64(dim)
		}
		// Relations are NOT normalized initially
	}
}

// normalizeEntity normalizes an entity embedding to unit length (L2 norm)
func (te *TransE) normalizeEntity(entityID int64) {
	norm := 0.0
	for d := 0; d < te.dim; d++ {
		norm += te.entityEmbeddings[entityID][d] * te.entityEmbeddings[entityID][d]
	}
	norm = math.Sqrt(norm)

	if norm > 1e-10 {
		for d := 0; d < te.dim; d++ {
			te.entityEmbeddings[entityID][d] /= norm
		}
	}
}

// score computes the TransE score for a triple: ||h + r - t||
// Lower score = better fit
func (te *TransE) score(head, relation, tail int64) float64 {
	distance := 0.0

	if te.norm == 1 {
		// L1 norm (Manhattan distance)
		for d := 0; d < te.dim; d++ {
			diff := te.entityEmbeddings[head][d] + te.relationEmbeddings[relation][d] - te.entityEmbeddings[tail][d]
			distance += math.Abs(diff)
		}
	} else {
		// L2 norm (Euclidean distance)
		for d := 0; d < te.dim; d++ {
			diff := te.entityEmbeddings[head][d] + te.relationEmbeddings[relation][d] - te.entityEmbeddings[tail][d]
			distance += diff * diff
		}
		distance = math.Sqrt(distance)
	}

	return distance
}

// Train trains the TransE model
func (te *TransE) Train(epochs int, batchSize int, alpha float64, workers int) {
	fmt.Println("Model:")
	fmt.Println("\t[TransE - Translating Embeddings]")
	fmt.Println()

	fmt.Println("Learning Parameters:")
	fmt.Printf("\tepochs:\t\t\t%d\n", epochs)
	fmt.Printf("\tbatch_size:\t\t%d\n", batchSize)
	fmt.Printf("\talpha:\t\t\t%.6f\n", alpha)
	fmt.Printf("\tworkers:\t\t%d\n", workers)
	fmt.Println()

	fmt.Println("Start Training:")

	totalBatches := int64(epochs) * ((te.kg.NumTriples + int64(batchSize) - 1) / int64(batchSize))
	batchCount := int64(0)
	alphaMin := alpha * 0.0001

	for epoch := 0; epoch < epochs; epoch++ {
		// Shuffle triples
		indices := make([]int64, te.kg.NumTriples)
		for i := int64(0); i < te.kg.NumTriples; i++ {
			indices[i] = i
		}
		// Fisher-Yates shuffle
		for i := int64(0); i < te.kg.NumTriples; i++ {
			j := i + rand.Int63n(te.kg.NumTriples-i)
			indices[i], indices[j] = indices[j], indices[i]
		}

		// Process in batches
		numBatches := (int(te.kg.NumTriples) + batchSize - 1) / batchSize

		for batchIdx := 0; batchIdx < numBatches; batchIdx++ {
			start := batchIdx * batchSize
			end := start + batchSize
			if end > int(te.kg.NumTriples) {
				end = int(te.kg.NumTriples)
			}

			// Process batch in parallel
			var wg sync.WaitGroup
			chunkSize := (end - start + workers - 1) / workers

			for w := 0; w < workers; w++ {
				wg.Add(1)
				chunkStart := start + w*chunkSize
				chunkEnd := chunkStart + chunkSize
				if chunkEnd > end {
					chunkEnd = end
				}

				go func(chunkStart, chunkEnd int) {
					defer wg.Done()

					rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(chunkStart)))

					for i := chunkStart; i < chunkEnd; i++ {
						tripleIdx := indices[i]
						triple := te.kg.GetTriple(tripleIdx)

						// Generate negative sample (corrupt head or tail)
						negTriple := triple
						if rng.Float64() < 0.5 {
							// Corrupt head
							negTriple.Head = te.kg.SampleNegativeHead(triple, rng)
						} else {
							// Corrupt tail
							negTriple.Tail = te.kg.SampleNegativeTail(triple, rng)
						}

						// Compute scores
						posScore := te.score(triple.Head, triple.Relation, triple.Tail)
						negScore := te.score(negTriple.Head, negTriple.Relation, negTriple.Tail)

						// Margin-based ranking loss: max(0, margin + posScore - negScore)
						loss := te.margin + posScore - negScore
						if loss > 0 {
							// Update gradients
							te.updateEmbeddings(triple, negTriple, alpha)
						}
					}
				}(chunkStart, chunkEnd)
			}

			wg.Wait()

			// Update progress
			batchCount++
			if batchCount%100 == 0 {
				currentAlpha := alpha * (1.0 - float64(batchCount)/float64(totalBatches))
				if currentAlpha < alphaMin {
					currentAlpha = alphaMin
				}
				progress := float64(batchCount) / float64(totalBatches) * 100
				fmt.Printf("\tEpoch: %d/%d\tAlpha: %.6f\tProgress: %.2f%%\r",
					epoch+1, epochs, currentAlpha, progress)
			}
		}

		// Normalize entity embeddings after each epoch
		for i := int64(0); i < te.kg.NumEntities; i++ {
			te.normalizeEntity(i)
		}
	}

	fmt.Printf("\tEpoch: %d/%d\tProgress: 100.00%%\n", epochs, epochs)
	fmt.Println("\nTraining Complete!")
}

// updateEmbeddings updates embeddings using gradient descent
func (te *TransE) updateEmbeddings(posTriple, negTriple knowledge.Triple, learningRate float64) {
	// Compute gradients for positive triple (h + r - t)
	posGrad := make([]float64, te.dim)
	for d := 0; d < te.dim; d++ {
		posGrad[d] = te.entityEmbeddings[posTriple.Head][d] +
			te.relationEmbeddings[posTriple.Relation][d] -
			te.entityEmbeddings[posTriple.Tail][d]
	}

	// Compute gradients for negative triple
	negGrad := make([]float64, te.dim)
	for d := 0; d < te.dim; d++ {
		negGrad[d] = te.entityEmbeddings[negTriple.Head][d] +
			te.relationEmbeddings[negTriple.Relation][d] -
			te.entityEmbeddings[negTriple.Tail][d]
	}

	// Normalize gradients based on norm type
	if te.norm == 1 {
		// L1: sign of gradient
		for d := 0; d < te.dim; d++ {
			if posGrad[d] > 0 {
				posGrad[d] = 1
			} else if posGrad[d] < 0 {
				posGrad[d] = -1
			}
			if negGrad[d] > 0 {
				negGrad[d] = 1
			} else if negGrad[d] < 0 {
				negGrad[d] = -1
			}
		}
	}

	// Update positive triple embeddings
	// Gradient: +posGrad for head and relation, -posGrad for tail
	for d := 0; d < te.dim; d++ {
		te.entityEmbeddings[posTriple.Head][d] -= learningRate * posGrad[d]
		te.relationEmbeddings[posTriple.Relation][d] -= learningRate * posGrad[d]
		te.entityEmbeddings[posTriple.Tail][d] += learningRate * posGrad[d]
	}

	// Update negative triple embeddings (opposite direction)
	// Gradient: -negGrad for head and relation, +negGrad for tail
	for d := 0; d < te.dim; d++ {
		te.entityEmbeddings[negTriple.Head][d] += learningRate * negGrad[d]
		te.relationEmbeddings[negTriple.Relation][d] += learningRate * negGrad[d]
		te.entityEmbeddings[negTriple.Tail][d] -= learningRate * negGrad[d]
	}
}

// SaveEmbeddings saves entity and relation embeddings to files
func (te *TransE) SaveEmbeddings(entityFile, relationFile string) error {
	fmt.Println("Save Model:")

	// Save entity embeddings
	entFile, err := os.Create(entityFile)
	if err != nil {
		return fmt.Errorf("failed to create entity file: %v", err)
	}
	defer entFile.Close()

	fmt.Fprintf(entFile, "%d %d\n", te.kg.NumEntities, te.dim)
	for i := int64(0); i < te.kg.NumEntities; i++ {
		name := te.kg.GetEntityName(i)
		fmt.Fprintf(entFile, "%s", name)
		for d := 0; d < te.dim; d++ {
			fmt.Fprintf(entFile, " %.6f", te.entityEmbeddings[i][d])
		}
		fmt.Fprintln(entFile)
	}
	fmt.Printf("\tEntities saved to <%s>\n", entityFile)

	// Save relation embeddings
	relFile, err := os.Create(relationFile)
	if err != nil {
		return fmt.Errorf("failed to create relation file: %v", err)
	}
	defer relFile.Close()

	fmt.Fprintf(relFile, "%d %d\n", te.kg.NumRelations, te.dim)
	for i := int64(0); i < te.kg.NumRelations; i++ {
		name := te.kg.GetRelationName(i)
		fmt.Fprintf(relFile, "%s", name)
		for d := 0; d < te.dim; d++ {
			fmt.Fprintf(relFile, " %.6f", te.relationEmbeddings[i][d])
		}
		fmt.Fprintln(relFile)
	}
	fmt.Printf("\tRelations saved to <%s>\n", relationFile)

	return nil
}

// Predict predicts the score for a given triple
// Lower score = more likely to be true
func (te *TransE) Predict(head, relation, tail string) (float64, error) {
	headID, exists := te.kg.EntityHash[head]
	if !exists {
		return 0, fmt.Errorf("entity not found: %s", head)
	}

	relID, exists := te.kg.RelationHash[relation]
	if !exists {
		return 0, fmt.Errorf("relation not found: %s", relation)
	}

	tailID, exists := te.kg.EntityHash[tail]
	if !exists {
		return 0, fmt.Errorf("entity not found: %s", tail)
	}

	return te.score(headID, relID, tailID), nil
}
