package rotate

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/cnclabs/smore/pkg/knowledge"
)

// RotatE implements the RotatE algorithm using complex-valued embeddings
// RotatE models relations as rotations in complex space: h ∘ r ≈ t
// where ∘ denotes element-wise complex multiplication (Hadamard product)
type RotatE struct {
	kg  *knowledge.KnowledgeGraph
	dim int // Actual embedding dimension (will be halved for complex representation)

	// Complex embeddings
	entityEmbeddings   [][]complex128 // Entity embeddings (complex-valued)
	relationEmbeddings [][]complex128 // Relation embeddings (unit complex numbers)

	// RotatE parameters
	margin          float64 // Margin for ranking loss (default: 6.0)
	adversarialTemp float64 // Temperature for self-adversarial negative sampling (default: 1.0)
}

// New creates a new RotatE instance
func New() *RotatE {
	return &RotatE{
		kg:              knowledge.NewKnowledgeGraph(),
		margin:          6.0,
		adversarialTemp: 1.0,
	}
}

// LoadTriples loads the knowledge graph from a triples file
func (re *RotatE) LoadTriples(filename string) error {
	return re.kg.LoadTriples(filename)
}

// Init initializes the model with given dimensions and parameters
func (re *RotatE) Init(dim int, margin float64, adversarialTemp float64) {
	if dim%2 != 0 {
		dim++ // Ensure even dimension for complex embeddings
	}
	re.dim = dim / 2 // Complex dimension (each complex number = 2 real dimensions)
	re.margin = margin
	re.adversarialTemp = adversarialTemp

	fmt.Println("Model Setting:")
	fmt.Printf("\tdimension:\t\t%d (complex: %d)\n", dim, re.dim)
	fmt.Printf("\tmargin:\t\t\t%.2f\n", margin)
	fmt.Printf("\tadversarial temp:\t%.2f\n", adversarialTemp)
	fmt.Println()
	fmt.Println("RotatE Principle:")
	fmt.Println("\th ∘ r ≈ t")
	fmt.Println("\t(head ∘ relation ≈ tail in complex space)")
	fmt.Println("\t∘ = element-wise complex multiplication (rotation)")
	fmt.Println()
	fmt.Println("Advantages over TransE:")
	fmt.Println("\t✓ Models symmetry, antisymmetry, inversion, composition")
	fmt.Println("\t✓ Better for 1-to-N, N-to-1, N-to-N relations")
	fmt.Println("\t✓ State-of-the-art on many benchmarks")

	// Initialize entity embeddings with random complex values
	re.entityEmbeddings = make([][]complex128, re.kg.NumEntities)
	for i := int64(0); i < re.kg.NumEntities; i++ {
		re.entityEmbeddings[i] = make([]complex128, re.dim)
		for d := 0; d < re.dim; d++ {
			// Random complex number with uniform phase
			phase := rand.Float64() * 2.0 * math.Pi
			magnitude := (rand.Float64()*0.5 + 0.5) / float64(re.dim)
			re.entityEmbeddings[i][d] = complex(
				magnitude*math.Cos(phase),
				magnitude*math.Sin(phase),
			)
		}
	}

	// Initialize relation embeddings as unit complex numbers (rotations)
	re.relationEmbeddings = make([][]complex128, re.kg.NumRelations)
	for i := int64(0); i < re.kg.NumRelations; i++ {
		re.relationEmbeddings[i] = make([]complex128, re.dim)
		for d := 0; d < re.dim; d++ {
			// Random phase for rotation (unit complex number)
			phase := rand.Float64() * 2.0 * math.Pi
			re.relationEmbeddings[i][d] = cmplx.Exp(complex(0, phase))
		}
	}
}

// score computes the RotatE score: ||h ∘ r - t||
// Lower score = better fit
func (re *RotatE) score(head, relation, tail int64) float64 {
	distance := 0.0

	for d := 0; d < re.dim; d++ {
		// Element-wise complex multiplication: h ∘ r
		rotated := re.entityEmbeddings[head][d] * re.relationEmbeddings[relation][d]

		// Compute difference: (h ∘ r) - t
		diff := rotated - re.entityEmbeddings[tail][d]

		// L2 distance (squared)
		distance += real(diff)*real(diff) + imag(diff)*imag(diff)
	}

	return math.Sqrt(distance)
}

// Train trains the RotatE model with self-adversarial negative sampling
func (re *RotatE) Train(epochs int, batchSize int, alpha float64, workers int) {
	fmt.Println()
	fmt.Println("Model:")
	fmt.Println("\t[RotatE - Rotation-based Knowledge Graph Embedding]")
	fmt.Println()

	fmt.Println("Learning Parameters:")
	fmt.Printf("\tepochs:\t\t\t%d\n", epochs)
	fmt.Printf("\tbatch_size:\t\t%d\n", batchSize)
	fmt.Printf("\talpha:\t\t\t%.6f\n", alpha)
	fmt.Printf("\tworkers:\t\t%d\n", workers)
	fmt.Println()

	fmt.Println("Start Training:")

	totalBatches := int64(epochs) * ((re.kg.NumTriples + int64(batchSize) - 1) / int64(batchSize))
	batchCount := int64(0)
	alphaMin := alpha * 0.0001
	currentAlpha := alpha

	for epoch := 0; epoch < epochs; epoch++ {
		// Shuffle triples
		indices := make([]int64, re.kg.NumTriples)
		for i := int64(0); i < re.kg.NumTriples; i++ {
			indices[i] = i
		}
		// Fisher-Yates shuffle
		for i := int64(0); i < re.kg.NumTriples; i++ {
			j := i + rand.Int63n(re.kg.NumTriples-i)
			indices[i], indices[j] = indices[j], indices[i]
		}

		// Process in batches
		numBatches := (int(re.kg.NumTriples) + batchSize - 1) / batchSize

		for batchIdx := 0; batchIdx < numBatches; batchIdx++ {
			start := batchIdx * batchSize
			end := start + batchSize
			if end > int(re.kg.NumTriples) {
				end = int(re.kg.NumTriples)
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
						triple := re.kg.GetTriple(tripleIdx)

						// Generate negative sample
						negTriple := triple
						if rng.Float64() < 0.5 {
							negTriple.Head = re.kg.SampleNegativeHead(triple, rng)
						} else {
							negTriple.Tail = re.kg.SampleNegativeTail(triple, rng)
						}

						// Compute scores
						posScore := re.score(triple.Head, triple.Relation, triple.Tail)
						negScore := re.score(negTriple.Head, negTriple.Relation, negTriple.Tail)

						// Self-adversarial negative sampling weight
						weight := 1.0
						if re.adversarialTemp > 0 {
							weight = math.Exp(-negScore / re.adversarialTemp)
						}

						// Margin-based ranking loss
						loss := re.margin + posScore - negScore
						if loss > 0 {
							re.updateEmbeddings(triple, negTriple, currentAlpha, weight)
						}
					}
				}(chunkStart, chunkEnd)
			}

			wg.Wait()

			// Update progress and learning rate
			batchCount++
			if batchCount%100 == 0 {
				currentAlpha = alpha * (1.0 - float64(batchCount)/float64(totalBatches))
				if currentAlpha < alphaMin {
					currentAlpha = alphaMin
				}
				progress := float64(batchCount) / float64(totalBatches) * 100
				fmt.Printf("\tEpoch: %d/%d\tAlpha: %.6f\tProgress: %.2f%%\r",
					epoch+1, epochs, currentAlpha, progress)
			}
		}

		// Normalize relation embeddings to unit complex numbers after each epoch
		re.normalizeRelations()
	}

	fmt.Printf("\tEpoch: %d/%d\tProgress: 100.00%%\n", epochs, epochs)
	fmt.Println("\nTraining Complete!")
}

// updateEmbeddings updates embeddings using gradient descent
func (re *RotatE) updateEmbeddings(posTriple, negTriple knowledge.Triple, learningRate, weight float64) {
	// Compute gradients for positive triple
	for d := 0; d < re.dim; d++ {
		// h ∘ r - t
		rotated := re.entityEmbeddings[posTriple.Head][d] * re.relationEmbeddings[posTriple.Relation][d]
		diff := rotated - re.entityEmbeddings[posTriple.Tail][d]

		// Normalize gradient
		norm := cmplx.Abs(diff)
		if norm > 1e-10 {
			grad := diff / complex(norm, 0)

			// Update embeddings
			re.entityEmbeddings[posTriple.Head][d] -= complex(learningRate, 0) * grad * cmplx.Conj(re.relationEmbeddings[posTriple.Relation][d])
			re.relationEmbeddings[posTriple.Relation][d] -= complex(learningRate, 0) * grad * cmplx.Conj(re.entityEmbeddings[posTriple.Head][d])
			re.entityEmbeddings[posTriple.Tail][d] += complex(learningRate, 0) * grad
		}
	}

	// Compute gradients for negative triple (with adversarial weight)
	for d := 0; d < re.dim; d++ {
		rotated := re.entityEmbeddings[negTriple.Head][d] * re.relationEmbeddings[negTriple.Relation][d]
		diff := rotated - re.entityEmbeddings[negTriple.Tail][d]

		norm := cmplx.Abs(diff)
		if norm > 1e-10 {
			grad := diff / complex(norm, 0) * complex(weight, 0)

			// Update embeddings (opposite direction)
			re.entityEmbeddings[negTriple.Head][d] += complex(learningRate, 0) * grad * cmplx.Conj(re.relationEmbeddings[negTriple.Relation][d])
			re.relationEmbeddings[negTriple.Relation][d] += complex(learningRate, 0) * grad * cmplx.Conj(re.entityEmbeddings[negTriple.Head][d])
			re.entityEmbeddings[negTriple.Tail][d] -= complex(learningRate, 0) * grad
		}
	}
}

// normalizeRelations normalizes relation embeddings to unit complex numbers
func (re *RotatE) normalizeRelations() {
	for i := int64(0); i < re.kg.NumRelations; i++ {
		for d := 0; d < re.dim; d++ {
			// Normalize to unit circle (magnitude = 1)
			magnitude := cmplx.Abs(re.relationEmbeddings[i][d])
			if magnitude > 1e-10 {
				re.relationEmbeddings[i][d] /= complex(magnitude, 0)
			}
		}
	}
}

// SaveEmbeddings saves entity and relation embeddings to files
func (re *RotatE) SaveEmbeddings(entityFile, relationFile string) error {
	fmt.Println("Save Model:")

	// Save entity embeddings (convert complex to real: [real1, imag1, real2, imag2, ...])
	entFile, err := os.Create(entityFile)
	if err != nil {
		return fmt.Errorf("failed to create entity file: %v", err)
	}
	defer entFile.Close()

	fmt.Fprintf(entFile, "%d %d\n", re.kg.NumEntities, re.dim*2)
	for i := int64(0); i < re.kg.NumEntities; i++ {
		name := re.kg.GetEntityName(i)
		fmt.Fprintf(entFile, "%s", name)
		for d := 0; d < re.dim; d++ {
			fmt.Fprintf(entFile, " %.6f %.6f", real(re.entityEmbeddings[i][d]), imag(re.entityEmbeddings[i][d]))
		}
		fmt.Fprintln(entFile)
	}
	fmt.Printf("\tEntities saved to <%s>\n", entityFile)

	// Save relation embeddings (as phases in radians)
	relFile, err := os.Create(relationFile)
	if err != nil {
		return fmt.Errorf("failed to create relation file: %v", err)
	}
	defer relFile.Close()

	fmt.Fprintf(relFile, "%d %d\n", re.kg.NumRelations, re.dim)
	for i := int64(0); i < re.kg.NumRelations; i++ {
		name := re.kg.GetRelationName(i)
		fmt.Fprintf(relFile, "%s", name)
		for d := 0; d < re.dim; d++ {
			// Save as phase angle
			phase := cmplx.Phase(re.relationEmbeddings[i][d])
			fmt.Fprintf(relFile, " %.6f", phase)
		}
		fmt.Fprintln(relFile)
	}
	fmt.Printf("\tRelations saved to <%s> (as rotation phases)\n", relationFile)

	return nil
}

// Predict predicts the score for a given triple
func (re *RotatE) Predict(head, relation, tail string) (float64, error) {
	headID, exists := re.kg.EntityHash[head]
	if !exists {
		return 0, fmt.Errorf("entity not found: %s", head)
	}

	relID, exists := re.kg.RelationHash[relation]
	if !exists {
		return 0, fmt.Errorf("relation not found: %s", relation)
	}

	tailID, exists := re.kg.EntityHash[tail]
	if !exists {
		return 0, fmt.Errorf("entity not found: %s", tail)
	}

	return re.score(headID, relID, tailID), nil
}
