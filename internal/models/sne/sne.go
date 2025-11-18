package sne

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/cnclabs/smore/pkg/signed"
)

// SNE implements Signed Network Embedding
// SNE models signed networks with both positive and negative edges
// using two embedding vectors per node (source and target)
type SNE struct {
	sn  *signed.SignedNetwork
	dim int

	// Embeddings: two vectors per node
	sourceEmbeddings [][]float64 // Source (from) embeddings
	targetEmbeddings [][]float64 // Target (to) embeddings

	// SNE parameters
	negativeSamples int     // Number of negative samples
	beta            float64 // Weight for negative edges (default: 1.0)
}

// New creates a new SNE instance
func New() *SNE {
	return &SNE{
		sn:              signed.NewSignedNetwork(),
		negativeSamples: 5,
		beta:            1.0,
	}
}

// LoadEdgeList loads the signed network from an edge list file
func (sne *SNE) LoadEdgeList(filename string, undirected bool) error {
	return sne.sn.LoadEdgeList(filename, undirected)
}

// Init initializes the model with given dimensions and parameters
func (sne *SNE) Init(dim int, negativeSamples int, beta float64) {
	sne.dim = dim
	sne.negativeSamples = negativeSamples
	sne.beta = beta

	fmt.Println("Model Setting:")
	fmt.Printf("\tdimension:\t\t%d\n", dim)
	fmt.Printf("\tnegative_samples:\t%d\n", negativeSamples)
	fmt.Printf("\tbeta (neg weight):\t%.2f\n", beta)
	fmt.Println()
	fmt.Println("SNE Principle:")
	fmt.Println("\t✓ Positive edges → similar embeddings")
	fmt.Println("\t✓ Negative edges → dissimilar embeddings")
	fmt.Println("\t✓ Balance theory: enemy of enemy = friend")
	fmt.Println()
	fmt.Println("Use Cases:")
	fmt.Println("\t• Social networks (friend/enemy)")
	fmt.Println("\t• Trust networks (trust/distrust)")
	fmt.Println("\t• Review systems (upvote/downvote)")
	fmt.Println("\t• Signed collaboration networks")

	maxVid := sne.sn.NumVertices

	// Initialize source embeddings
	sne.sourceEmbeddings = make([][]float64, maxVid)
	for vid := int64(0); vid < maxVid; vid++ {
		sne.sourceEmbeddings[vid] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			sne.sourceEmbeddings[vid][d] = (rand.Float64() - 0.5) / float64(dim)
		}
	}

	// Initialize target embeddings
	sne.targetEmbeddings = make([][]float64, maxVid)
	for vid := int64(0); vid < maxVid; vid++ {
		sne.targetEmbeddings[vid] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			sne.targetEmbeddings[vid][d] = (rand.Float64() - 0.5) / float64(dim)
		}
	}
}

// sigmoid computes the sigmoid function
func sigmoid(x float64) float64 {
	if x > 6 {
		return 1.0
	} else if x < -6 {
		return 0.0
	}
	return 1.0 / (1.0 + math.Exp(-x))
}

// dotProduct computes the dot product of two vectors
func dotProduct(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// Train trains the SNE model
func (sne *SNE) Train(epochs int, samplesPerEdge int, alpha float64, workers int) {
	fmt.Println()
	fmt.Println("Model:")
	fmt.Println("\t[SNE - Signed Network Embedding]")
	fmt.Println()

	fmt.Println("Learning Parameters:")
	fmt.Printf("\tepochs:\t\t\t%d\n", epochs)
	fmt.Printf("\tsamples_per_edge:\t%d\n", samplesPerEdge)
	fmt.Printf("\talpha:\t\t\t%.6f\n", alpha)
	fmt.Printf("\tworkers:\t\t%d\n", workers)
	fmt.Println()

	fmt.Println("Start Training:")

	totalSamples := int64(epochs) * int64(samplesPerEdge) * (sne.sn.NumPositiveEdges + sne.sn.NumNegativeEdges)
	sampleCount := int64(0)
	alphaMin := alpha * 0.0001
	currentAlpha := alpha

	var countMu sync.Mutex

	for epoch := 0; epoch < epochs; epoch++ {
		// Process positive edges
		sne.processEdges(true, samplesPerEdge, &currentAlpha, alpha, &sampleCount, totalSamples, &countMu, workers)

		// Process negative edges
		sne.processEdges(false, samplesPerEdge, &currentAlpha, alpha, &sampleCount, totalSamples, &countMu, workers)

		// Update learning rate
		currentAlpha = alpha * (1.0 - float64(sampleCount)/float64(totalSamples))
		if currentAlpha < alphaMin {
			currentAlpha = alphaMin
		}

		progress := float64(sampleCount) / float64(totalSamples) * 100
		fmt.Printf("\tEpoch: %d/%d\tAlpha: %.6f\tProgress: %.2f%%\r",
			epoch+1, epochs, currentAlpha, progress)
	}

	fmt.Printf("\tEpoch: %d/%d\tProgress: 100.00%%\n", epochs, epochs)
	fmt.Println("\nTraining Complete!")
}

// processEdges processes positive or negative edges
func (sne *SNE) processEdges(positive bool, samplesPerEdge int, currentAlpha *float64, baseAlpha float64,
	sampleCount *int64, totalSamples int64, countMu *sync.Mutex, workers int) {

	// Get all edges of this type
	var edges []struct {
		from int64
		to   int64
	}

	if positive {
		for from, neighbors := range sne.sn.PositiveEdges {
			for _, to := range neighbors {
				edges = append(edges, struct {
					from int64
					to   int64
				}{from, to})
			}
		}
	} else {
		for from, neighbors := range sne.sn.NegativeEdges {
			for _, to := range neighbors {
				edges = append(edges, struct {
					from int64
					to   int64
				}{from, to})
			}
		}
	}

	if len(edges) == 0 {
		return
	}

	// Shuffle edges
	for i := range edges {
		j := rand.Intn(i + 1)
		edges[i], edges[j] = edges[j], edges[i]
	}

	// Process in parallel
	var wg sync.WaitGroup
	chunkSize := (len(edges) + workers - 1) / workers

	for w := 0; w < workers; w++ {
		wg.Add(1)
		start := w * chunkSize
		end := start + chunkSize
		if end > len(edges) {
			end = len(edges)
		}

		go func(start, end int) {
			defer wg.Done()

			rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(start)))

			for i := start; i < end; i++ {
				from := edges[i].from
				to := edges[i].to

				for s := 0; s < samplesPerEdge; s++ {
					if positive {
						sne.updatePositiveEdge(from, to, *currentAlpha, rng)
					} else {
						sne.updateNegativeEdge(from, to, *currentAlpha, rng)
					}

					// Update progress
					countMu.Lock()
					*sampleCount++
					if *sampleCount%10000 == 0 {
						*currentAlpha = baseAlpha * (1.0 - float64(*sampleCount)/float64(totalSamples))
						if *currentAlpha < baseAlpha*0.0001 {
							*currentAlpha = baseAlpha * 0.0001
						}
					}
					countMu.Unlock()
				}
			}
		}(start, end)
	}

	wg.Wait()
}

// updatePositiveEdge updates embeddings for a positive edge
func (sne *SNE) updatePositiveEdge(from, to int64, learningRate float64, rng *rand.Rand) {
	// Positive edge: maximize similarity
	// Loss: -log(sigmoid(u^T v))

	// Compute current score
	score := dotProduct(sne.sourceEmbeddings[from], sne.targetEmbeddings[to])
	label := 1.0
	pred := sigmoid(score)
	grad := learningRate * (label - pred)

	// Update embeddings
	for d := 0; d < sne.dim; d++ {
		sourceGrad := grad * sne.targetEmbeddings[to][d]
		targetGrad := grad * sne.sourceEmbeddings[from][d]

		sne.sourceEmbeddings[from][d] += sourceGrad
		sne.targetEmbeddings[to][d] += targetGrad
	}

	// Negative sampling
	for i := 0; i < sne.negativeSamples; i++ {
		neg := sne.sn.SampleVertex(rng)
		if neg == to {
			continue
		}

		negScore := dotProduct(sne.sourceEmbeddings[from], sne.targetEmbeddings[neg])
		negLabel := 0.0
		negPred := sigmoid(negScore)
		negGrad := learningRate * (negLabel - negPred)

		for d := 0; d < sne.dim; d++ {
			sourceGrad := negGrad * sne.targetEmbeddings[neg][d]
			targetGrad := negGrad * sne.sourceEmbeddings[from][d]

			sne.sourceEmbeddings[from][d] += sourceGrad
			sne.targetEmbeddings[neg][d] += targetGrad
		}
	}
}

// updateNegativeEdge updates embeddings for a negative edge
func (sne *SNE) updateNegativeEdge(from, to int64, learningRate float64, rng *rand.Rand) {
	// Negative edge: minimize similarity
	// Loss: -log(1 - sigmoid(u^T v)) = -log(sigmoid(-u^T v))

	score := dotProduct(sne.sourceEmbeddings[from], sne.targetEmbeddings[to])
	label := 0.0
	pred := sigmoid(score)
	grad := learningRate * sne.beta * (label - pred)

	// Update embeddings
	for d := 0; d < sne.dim; d++ {
		sourceGrad := grad * sne.targetEmbeddings[to][d]
		targetGrad := grad * sne.sourceEmbeddings[from][d]

		sne.sourceEmbeddings[from][d] += sourceGrad
		sne.targetEmbeddings[to][d] += targetGrad
	}

	// Negative sampling (sample positive edges as negatives for negative edges)
	for i := 0; i < sne.negativeSamples; i++ {
		neg := sne.sn.SampleVertex(rng)
		if neg == to || sne.sn.HasNegativeEdge(from, neg) {
			continue
		}

		negScore := dotProduct(sne.sourceEmbeddings[from], sne.targetEmbeddings[neg])
		negLabel := 1.0
		negPred := sigmoid(negScore)
		negGrad := learningRate * sne.beta * (negLabel - negPred)

		for d := 0; d < sne.dim; d++ {
			sourceGrad := negGrad * sne.targetEmbeddings[neg][d]
			targetGrad := negGrad * sne.sourceEmbeddings[from][d]

			sne.sourceEmbeddings[from][d] += sourceGrad
			sne.targetEmbeddings[neg][d] += targetGrad
		}
	}
}

// SaveEmbeddings saves the learned embeddings
func (sne *SNE) SaveEmbeddings(filename string) error {
	fmt.Println("Save Model:")

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer file.Close()

	// Save as concatenated [source, target] vectors
	fmt.Fprintf(file, "%d %d\n", sne.sn.NumVertices, sne.dim*2)

	for vid := int64(0); vid < sne.sn.NumVertices; vid++ {
		name := sne.sn.GetVertexName(vid)
		fmt.Fprintf(file, "%s", name)

		// Write source embedding
		for d := 0; d < sne.dim; d++ {
			fmt.Fprintf(file, " %.6f", sne.sourceEmbeddings[vid][d])
		}

		// Write target embedding
		for d := 0; d < sne.dim; d++ {
			fmt.Fprintf(file, " %.6f", sne.targetEmbeddings[vid][d])
		}

		fmt.Fprintln(file)
	}

	fmt.Printf("\tSave to <%s>\n", filename)
	return nil
}

// PredictSign predicts the sign of an edge
func (sne *SNE) PredictSign(from, to int64) float64 {
	score := dotProduct(sne.sourceEmbeddings[from], sne.targetEmbeddings[to])
	return score // Positive score → positive edge, negative score → negative edge
}

// ComputeBalanceRatio computes how well the embeddings preserve balance theory
func (sne *SNE) ComputeBalanceRatio() float64 {
	correct := 0
	total := 0

	// Check positive edges
	for from, neighbors := range sne.sn.PositiveEdges {
		for _, to := range neighbors {
			score := sne.PredictSign(from, to)
			if score > 0 {
				correct++
			}
			total++
		}
	}

	// Check negative edges
	for from, neighbors := range sne.sn.NegativeEdges {
		for _, to := range neighbors {
			score := sne.PredictSign(from, to)
			if score < 0 {
				correct++
			}
			total++
		}
	}

	if total == 0 {
		return 0.0
	}

	return float64(correct) / float64(total)
}
