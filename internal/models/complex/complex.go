package complex_embeddings

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/cnclabs/smore/pkg/knowledge"
)

// ComplEx implements Complex Embeddings for Knowledge Graphs
// Uses complex-valued embeddings to model symmetric and asymmetric relations
type ComplEx struct {
	kg  *knowledge.KnowledgeGraph
	dim int

	// Complex-valued embeddings
	entityEmbeddings   [][]complex128
	relationEmbeddings [][]complex128

	// Training parameters
	learningRate float64
	margin       float64
}

// New creates a new ComplEx instance
func New() *ComplEx {
	return &ComplEx{
		kg: knowledge.NewKnowledgeGraph(),
	}
}

// LoadTriples loads the knowledge graph
func (cx *ComplEx) LoadTriples(filename string) error {
	return cx.kg.LoadTriples(filename)
}

// Init initializes the model
func (cx *ComplEx) Init(dim int, learningRate, margin float64) {
	cx.dim = dim
	cx.learningRate = learningRate
	cx.margin = margin

	fmt.Println("Model Setting:")
	fmt.Printf("\tdimension:\t\t%d\n", dim)
	fmt.Printf("\tlearning_rate:\t\t%.6f\n", learningRate)
	fmt.Printf("\tmargin:\t\t\t%.2f\n", margin)
	fmt.Println()

	fmt.Println("ComplEx Principle:")
	fmt.Println("\t✓ Complex-valued embeddings for entities and relations")
	fmt.Println("\t✓ Score: Re(<h, r, conj(t)>) = Re(Σ h_i * r_i * conj(t_i))")
	fmt.Println("\t✓ Can model symmetric, antisymmetric, and inverse relations")
	fmt.Println("\t✓ More expressive than TransE")

	// Initialize entity embeddings with complex values
	cx.entityEmbeddings = make([][]complex128, cx.kg.NumEntities)
	for i := int64(0); i < cx.kg.NumEntities; i++ {
		cx.entityEmbeddings[i] = make([]complex128, dim)
		for d := 0; d < dim; d++ {
			// Initialize with small random complex numbers
			real := (rand.Float64() - 0.5) / float64(dim)
			imag := (rand.Float64() - 0.5) / float64(dim)
			cx.entityEmbeddings[i][d] = complex(real, imag)
		}
		// Normalize
		cx.normalizeEntity(i)
	}

	// Initialize relation embeddings with complex values
	cx.relationEmbeddings = make([][]complex128, cx.kg.NumRelations)
	for i := int64(0); i < cx.kg.NumRelations; i++ {
		cx.relationEmbeddings[i] = make([]complex128, dim)
		for d := 0; d < dim; d++ {
			real := (rand.Float64() - 0.5) / float64(dim)
			imag := (rand.Float64() - 0.5) / float64(dim)
			cx.relationEmbeddings[i][d] = complex(real, imag)
		}
	}
}

// score computes the ComplEx score for a triple (h, r, t)
// Score = Re(<h, r, conj(t)>) = Re(Σ h_i * r_i * conj(t_i))
func (cx *ComplEx) score(head, relation, tail int64) float64 {
	var sum complex128 = 0

	for d := 0; d < cx.dim; d++ {
		h := cx.entityEmbeddings[head][d]
		r := cx.relationEmbeddings[relation][d]
		t := cx.entityEmbeddings[tail][d]

		// Trilinear product: h * r * conj(t)
		sum += h * r * cmplxConj(t)
	}

	// Return real part
	return real(sum)
}

// normalizeEntity normalizes an entity embedding to unit length
func (cx *ComplEx) normalizeEntity(entityID int64) {
	norm := 0.0
	for d := 0; d < cx.dim; d++ {
		val := cx.entityEmbeddings[entityID][d]
		norm += real(val)*real(val) + imag(val)*imag(val)
	}
	norm = math.Sqrt(norm)

	if norm > 0 {
		for d := 0; d < cx.dim; d++ {
			cx.entityEmbeddings[entityID][d] /= complex(norm, 0)
		}
	}
}

// updateTriple updates embeddings for a triple using gradient descent
func (cx *ComplEx) updateTriple(head, relation, tail int64, isPositive bool, learningRate float64) {
	// Compute loss gradient
	var lossGradient float64
	if isPositive {
		// Positive triple: maximize score (minimize -score)
		lossGradient = -1.0
	} else {
		// Negative triple: minimize score (minimize score)
		lossGradient = 1.0
	}

	// Gradient of score with respect to embeddings
	// ∂score/∂h = r * conj(t)
	// ∂score/∂r = h * conj(t)
	// ∂score/∂t = conj(h * r)

	for d := 0; d < cx.dim; d++ {
		h := cx.entityEmbeddings[head][d]
		r := cx.relationEmbeddings[relation][d]
		t := cx.entityEmbeddings[tail][d]

		// Compute gradients
		gradH := r * cmplxConj(t)
		gradR := h * cmplxConj(t)
		gradT := cmplxConj(h * r)

		// Update with gradient descent (only real part of gradient for real loss)
		cx.entityEmbeddings[head][d] -= complex(learningRate*lossGradient*real(gradH), learningRate*lossGradient*imag(gradH))
		cx.relationEmbeddings[relation][d] -= complex(learningRate*lossGradient*real(gradR), learningRate*lossGradient*imag(gradR))
		cx.entityEmbeddings[tail][d] -= complex(learningRate*lossGradient*real(gradT), learningRate*lossGradient*imag(gradT))
	}

	// Normalize entity embeddings
	cx.normalizeEntity(head)
	cx.normalizeEntity(tail)
}

// Train trains the ComplEx model
func (cx *ComplEx) Train(epochs, batchSize, negativeSamples int, workers int) {
	fmt.Println()
	fmt.Println("Model:")
	fmt.Println("\t[ComplEx - Complex Embeddings]")
	fmt.Println()

	fmt.Println("Learning Parameters:")
	fmt.Printf("\tepochs:\t\t\t%d\n", epochs)
	fmt.Printf("\tbatch_size:\t\t%d\n", batchSize)
	fmt.Printf("\tnegative_samples:\t%d\n", negativeSamples)
	fmt.Printf("\tworkers:\t\t%d\n", workers)
	fmt.Println()

	fmt.Println("Start Training:")

	totalTriples := len(cx.kg.Triples)

	for epoch := 0; epoch < epochs; epoch++ {
		fmt.Printf("\nEpoch %d/%d:\n", epoch+1, epochs)

		// Shuffle triples
		indices := make([]int, totalTriples)
		for i := range indices {
			indices[i] = i
		}
		rand.Shuffle(totalTriples, func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})

		totalLoss := 0.0
		count := 0
		var lossMu sync.Mutex

		// Process in batches with parallel workers
		numBatches := (totalTriples + batchSize - 1) / batchSize

		for batchIdx := 0; batchIdx < numBatches; batchIdx++ {
			start := batchIdx * batchSize
			end := start + batchSize
			if end > totalTriples {
				end = totalTriples
			}

			var wg sync.WaitGroup
			batchIndices := indices[start:end]
			chunkSize := (len(batchIndices) + workers - 1) / workers

			for w := 0; w < workers; w++ {
				wg.Add(1)
				chunkStart := w * chunkSize
				chunkEnd := chunkStart + chunkSize
				if chunkEnd > len(batchIndices) {
					chunkEnd = len(batchIndices)
				}

				go func(chunk []int) {
					defer wg.Done()

					rng := rand.New(rand.NewSource(time.Now().UnixNano()))
					localLoss := 0.0

					for _, idx := range chunk {
						triple := cx.kg.Triples[idx]

						// Positive score
						posScore := cx.score(triple.Head, triple.Relation, triple.Tail)

						// Sample negative triples
						for n := 0; n < negativeSamples; n++ {
							// Randomly corrupt head or tail
							var negHead, negTail int64
							if rng.Float64() < 0.5 {
								// Corrupt head
								negHead = cx.kg.SampleNegativeHead(triple, rng)
								negTail = triple.Tail
							} else {
								// Corrupt tail
								negHead = triple.Head
								negTail = cx.kg.SampleNegativeTail(triple, rng)
							}

							negScore := cx.score(negHead, triple.Relation, negTail)

							// Margin-based ranking loss: max(0, margin + negScore - posScore)
							loss := math.Max(0, cx.margin+negScore-posScore)
							localLoss += loss

							// Update if loss > 0
							if loss > 0 {
								cx.updateTriple(triple.Head, triple.Relation, triple.Tail, true, cx.learningRate)
								cx.updateTriple(negHead, triple.Relation, negTail, false, cx.learningRate)
							}
						}
					}

					lossMu.Lock()
					totalLoss += localLoss
					count += len(chunk)
					lossMu.Unlock()
				}(batchIndices[chunkStart:chunkEnd])
			}

			wg.Wait()

			// Progress reporting
			if (batchIdx+1)%10 == 0 || batchIdx == numBatches-1 {
				progress := float64(count) / float64(totalTriples) * 100
				avgLoss := totalLoss / float64(count*negativeSamples)
				fmt.Printf("\tProgress: %.2f%% - Avg Loss: %.4f\r", progress, avgLoss)
			}
		}

		avgLoss := totalLoss / float64(totalTriples*negativeSamples)
		fmt.Printf("\tEpoch %d completed - Avg Loss: %.4f\n", epoch+1, avgLoss)
	}

	fmt.Println("\nTraining Complete!")
}

// SaveEmbeddings saves the learned embeddings
func (cx *ComplEx) SaveEmbeddings(filename string) error {
	fmt.Println("Save Model:")

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer file.Close()

	// Write header
	fmt.Fprintf(file, "%d %d %d\n", cx.kg.NumEntities, cx.kg.NumRelations, cx.dim)

	// Write entity embeddings
	fmt.Fprintln(file, "# Entities")
	for i := int64(0); i < cx.kg.NumEntities; i++ {
		name := cx.kg.GetEntityName(i)
		fmt.Fprintf(file, "E\t%s", name)

		for d := 0; d < cx.dim; d++ {
			val := cx.entityEmbeddings[i][d]
			fmt.Fprintf(file, " %.6f %.6fi", real(val), imag(val))
		}
		fmt.Fprintln(file)
	}

	// Write relation embeddings
	fmt.Fprintln(file, "# Relations")
	for i := int64(0); i < cx.kg.NumRelations; i++ {
		name := cx.kg.GetRelationName(i)
		fmt.Fprintf(file, "R\t%s", name)

		for d := 0; d < cx.dim; d++ {
			val := cx.relationEmbeddings[i][d]
			fmt.Fprintf(file, " %.6f %.6fi", real(val), imag(val))
		}
		fmt.Fprintln(file)
	}

	fmt.Printf("\tSave to <%s>\n", filename)
	return nil
}

// EvaluateLinkPrediction evaluates link prediction performance
func (cx *ComplEx) EvaluateLinkPrediction(testTriples int) {
	fmt.Println("\nLink Prediction Evaluation:")

	if testTriples > len(cx.kg.Triples) {
		testTriples = len(cx.kg.Triples)
	}

	// Use last testTriples for evaluation
	startIdx := len(cx.kg.Triples) - testTriples

	hits := 0
	totalRank := 0.0

	for i := startIdx; i < len(cx.kg.Triples); i++ {
		triple := cx.kg.Triples[i]
		correctScore := cx.score(triple.Head, triple.Relation, triple.Tail)

		// Rank against random negative samples
		betterCount := 0
		numSamples := 10

		for n := 0; n < numSamples; n++ {
			negHead := rand.Int63n(cx.kg.NumEntities)
			negScore := cx.score(negHead, triple.Relation, triple.Tail)

			if correctScore > negScore {
				betterCount++
			}
		}

		rank := numSamples - betterCount + 1
		totalRank += float64(rank)

		if rank <= 3 {
			hits++
		}
	}

	mrr := 1.0 / (totalRank / float64(testTriples))
	hits3 := float64(hits) / float64(testTriples) * 100

	fmt.Printf("\tMean Reciprocal Rank (MRR): %.4f\n", mrr)
	fmt.Printf("\tHits@3: %.2f%%\n", hits3)
}

// cmplxConj returns the complex conjugate
func cmplxConj(c complex128) complex128 {
	return complex(real(c), -imag(c))
}
