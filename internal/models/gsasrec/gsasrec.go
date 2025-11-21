package gsasrec

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/cnclabs/smore/pkg/pronet"
)

// gSASRec implements Generalized Self-Attentive Sequential Recommendation
// Paper: "gSASRec: Reducing Overconfidence in Sequential Recommendation
// Trained with Negative Sampling" (RecSys 2023 Best Paper)
// By Aleksandr V. Petrov and Craig Macdonald
type gSASRec struct {
	pnet *pronet.ProNet
	dim  int

	// Model parameters
	maxSeqLen   int
	numBlocks   int
	numHeads    int
	dropoutRate float64

	// gBCE loss parameters (key innovation)
	beta float64 // Generalization parameter for BCE loss

	// Embeddings
	itemEmbed [][]float64 // [numItems x dim]
	posEmbed  [][]float64 // [maxSeqLen x dim]

	// Attention layers
	attnWeights [][][]float64 // [numBlocks x dim x dim]
	ffnWeights1 [][][]float64 // [numBlocks x dim x dim*4]
	ffnWeights2 [][][]float64 // [numBlocks x dim*4 x dim]

	// User sequences
	userSeqs map[int64][]int64
}

// New creates a new gSASRec model
func New() *gSASRec {
	return &gSASRec{
		pnet:     pronet.NewProNet(),
		userSeqs: make(map[int64][]int64),
		beta:     0.5, // Default beta value from paper
	}
}

// LoadEdgeList loads user-item interaction sequences
func (g *gSASRec) LoadEdgeList(filename string, undirected bool) error {
	return g.pnet.LoadEdgeList(filename, undirected)
}

// BuildSequencesFromGraph builds user sequences from loaded graph
func (g *gSASRec) BuildSequencesFromGraph() {
	fmt.Println("Building sequences from graph...")
	g.userSeqs = make(map[int64][]int64)

	for user := int64(0); user < g.pnet.MaxVid; user++ {
		if neighbors, exists := g.pnet.Graph[user]; exists {
			g.userSeqs[user] = neighbors
		}
	}

	fmt.Printf("\tBuilt sequences for %d users\n", len(g.userSeqs))
}

// Init initializes the model with given parameters
func (g *gSASRec) Init(dim, maxSeqLen, numBlocks, numHeads int, dropoutRate, beta float64) {
	g.dim = dim
	g.maxSeqLen = maxSeqLen
	g.numBlocks = numBlocks
	g.numHeads = numHeads
	g.dropoutRate = dropoutRate
	g.beta = beta

	numItems := g.pnet.MaxVid

	fmt.Println("Model Setting:")
	fmt.Printf("\tdimension:\t\t%d\n", dim)
	fmt.Printf("\tmax_seq_len:\t\t%d\n", maxSeqLen)
	fmt.Printf("\tnum_blocks:\t\t%d\n", numBlocks)
	fmt.Printf("\tnum_heads:\t\t%d\n", numHeads)
	fmt.Printf("\tdropout_rate:\t\t%.2f\n", dropoutRate)
	fmt.Printf("\tbeta (gBCE):\t\t%.2f\n", beta)

	// Initialize item embeddings
	g.itemEmbed = make([][]float64, numItems)
	for i := int64(0); i < numItems; i++ {
		g.itemEmbed[i] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			g.itemEmbed[i][d] = (rand.Float64() - 0.5) / float64(dim)
		}
	}

	// Initialize positional embeddings
	g.posEmbed = make([][]float64, maxSeqLen)
	for pos := 0; pos < maxSeqLen; pos++ {
		g.posEmbed[pos] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			// Sinusoidal positional encoding
			if d%2 == 0 {
				g.posEmbed[pos][d] = math.Sin(float64(pos) / math.Pow(10000, float64(d)/float64(dim)))
			} else {
				g.posEmbed[pos][d] = math.Cos(float64(pos) / math.Pow(10000, float64(d-1)/float64(dim)))
			}
		}
	}

	// Initialize attention weights
	g.attnWeights = make([][][]float64, numBlocks)
	for b := 0; b < numBlocks; b++ {
		g.attnWeights[b] = make([][]float64, dim)
		for i := 0; i < dim; i++ {
			g.attnWeights[b][i] = make([]float64, dim)
			for j := 0; j < dim; j++ {
				g.attnWeights[b][i][j] = (rand.Float64() - 0.5) / math.Sqrt(float64(dim))
			}
		}
	}

	// Initialize FFN weights
	hiddenDim := dim * 4
	g.ffnWeights1 = make([][][]float64, numBlocks)
	g.ffnWeights2 = make([][][]float64, numBlocks)
	for b := 0; b < numBlocks; b++ {
		g.ffnWeights1[b] = make([][]float64, dim)
		for i := 0; i < dim; i++ {
			g.ffnWeights1[b][i] = make([]float64, hiddenDim)
			for j := 0; j < hiddenDim; j++ {
				g.ffnWeights1[b][i][j] = (rand.Float64() - 0.5) / math.Sqrt(float64(dim))
			}
		}

		g.ffnWeights2[b] = make([][]float64, hiddenDim)
		for i := 0; i < hiddenDim; i++ {
			g.ffnWeights2[b][i] = make([]float64, dim)
			for j := 0; j < dim; j++ {
				g.ffnWeights2[b][i][j] = (rand.Float64() - 0.5) / math.Sqrt(float64(hiddenDim))
			}
		}
	}
}

// selfAttention computes simplified self-attention with causal masking
func (g *gSASRec) selfAttention(seq [][]float64, block int) [][]float64 {
	seqLen := len(seq)
	output := make([][]float64, seqLen)

	for i := 0; i < seqLen; i++ {
		output[i] = make([]float64, g.dim)

		// Compute attention scores
		scores := make([]float64, seqLen)
		sumScores := 0.0
		for j := 0; j <= i; j++ { // Causal masking
			score := 0.0
			for d := 0; d < g.dim; d++ {
				for k := 0; k < g.dim; k++ {
					score += seq[i][d] * g.attnWeights[block][d][k] * seq[j][k]
				}
			}
			score /= math.Sqrt(float64(g.dim))
			scores[j] = math.Exp(score)
			sumScores += scores[j]
		}

		// Normalize and apply attention
		for j := 0; j <= i; j++ {
			weight := scores[j] / sumScores
			for d := 0; d < g.dim; d++ {
				output[i][d] += weight * seq[j][d]
			}
		}
	}

	return output
}

// feedForward applies position-wise feed-forward network
func (g *gSASRec) feedForward(seq [][]float64, block int) [][]float64 {
	seqLen := len(seq)
	output := make([][]float64, seqLen)
	hiddenDim := g.dim * 4

	for i := 0; i < seqLen; i++ {
		// First layer with ReLU
		hidden := make([]float64, hiddenDim)
		for h := 0; h < hiddenDim; h++ {
			for d := 0; d < g.dim; d++ {
				hidden[h] += seq[i][d] * g.ffnWeights1[block][d][h]
			}
			if hidden[h] < 0 {
				hidden[h] = 0 // ReLU
			}
		}

		// Second layer
		output[i] = make([]float64, g.dim)
		for d := 0; d < g.dim; d++ {
			for h := 0; h < hiddenDim; h++ {
				output[i][d] += hidden[h] * g.ffnWeights2[block][h][d]
			}
		}
	}

	return output
}

// forward performs forward pass through the model
func (g *gSASRec) forward(sequence []int64) [][]float64 {
	seqLen := len(sequence)
	if seqLen > g.maxSeqLen {
		sequence = sequence[seqLen-g.maxSeqLen:]
		seqLen = g.maxSeqLen
	}

	// Create input embeddings (item + positional)
	hidden := make([][]float64, seqLen)
	for i := 0; i < seqLen; i++ {
		hidden[i] = make([]float64, g.dim)
		itemID := sequence[i]
		for d := 0; d < g.dim; d++ {
			hidden[i][d] = g.itemEmbed[itemID][d] + g.posEmbed[i][d]
		}
	}

	// Apply transformer blocks
	for b := 0; b < g.numBlocks; b++ {
		// Self-attention with residual
		attnOut := g.selfAttention(hidden, b)
		for i := 0; i < seqLen; i++ {
			for d := 0; d < g.dim; d++ {
				attnOut[i][d] += hidden[i][d] // Residual connection
			}
		}

		// Feed-forward with residual
		ffnOut := g.feedForward(attnOut, b)
		for i := 0; i < seqLen; i++ {
			for d := 0; d < g.dim; d++ {
				hidden[i][d] = ffnOut[i][d] + attnOut[i][d] // Residual connection
			}
		}
	}

	return hidden
}

// generalizedBCE computes the generalized binary cross-entropy loss
// This is the KEY INNOVATION of gSASRec to reduce overconfidence
// Formula: gBCE = beta * BCE_pos + (1-beta) * BCE_neg
func (g *gSASRec) generalizedBCE(score float64, label float64) (float64, float64) {
	prob := g.pnet.FastSigmoid(score)

	var loss, grad float64
	if label > 0.5 { // Positive sample
		// BCE for positive: -log(prob)
		// Weight by beta
		loss = -g.beta * math.Log(prob + 1e-10)
		grad = g.beta * (prob - 1.0)
	} else { // Negative sample
		// BCE for negative: -log(1-prob)
		// Weight by (1-beta)
		loss = -(1.0 - g.beta) * math.Log(1.0 - prob + 1e-10)
		grad = (1.0 - g.beta) * prob
	}

	return loss, grad
}

// Train trains the gSASRec model with generalized BCE loss
func (g *gSASRec) Train(epochs, batchSize, negativeSamples int, alpha float64, workers int) {
	fmt.Println("Model:")
	fmt.Println("\t[gSASRec - RecSys 2023 Best Paper]")

	fmt.Println("Learning Parameters:")
	fmt.Printf("\tepochs:\t\t\t%d\n", epochs)
	fmt.Printf("\tbatch_size:\t\t%d\n", batchSize)
	fmt.Printf("\tnegative_samples:\t%d\n", negativeSamples)
	fmt.Printf("\talpha:\t\t\t%.6f\n", alpha)
	fmt.Printf("\tworkers:\t\t%d\n", workers)
	fmt.Printf("\tbeta (gBCE):\t\t%.2f\n", g.beta)

	fmt.Println("Start Training:")

	totalSeqs := int64(len(g.userSeqs))
	alphaMin := alpha * 0.0001

	for epoch := 0; epoch < epochs; epoch++ {
		currentAlpha := alpha * (1.0 - float64(epoch)/float64(epochs))
		if currentAlpha < alphaMin {
			currentAlpha = alphaMin
		}

		count := int64(0)
		totalLoss := 0.0
		var countMu sync.Mutex
		var wg sync.WaitGroup

		// Convert map to slice for parallel processing
		users := make([]int64, 0, len(g.userSeqs))
		for user := range g.userSeqs {
			users = append(users, user)
		}

		chunkSize := (len(users) + workers - 1) / workers

		for w := 0; w < workers; w++ {
			wg.Add(1)
			start := w * chunkSize
			end := start + chunkSize
			if end > len(users) {
				end = len(users)
			}

			go func(start, end int) {
				defer wg.Done()
				rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(start)))
				localLoss := 0.0

				for i := start; i < end; i++ {
					user := users[i]
					seq := g.userSeqs[user]
					if len(seq) < 2 {
						continue
					}

					// Sample positions to predict
					seqLen := len(seq)
					for pos := 1; pos < seqLen; pos++ {
						inputSeq := seq[:pos]
						target := seq[pos]

						// Forward pass
						hiddenStates := g.forward(inputSeq)
						lastHidden := hiddenStates[len(hiddenStates)-1]

						// Positive sample with gBCE
						posScore := 0.0
						for d := 0; d < g.dim; d++ {
							posScore += lastHidden[d] * g.itemEmbed[target][d]
						}
						posLoss, posGrad := g.generalizedBCE(posScore, 1.0)
						localLoss += posLoss

						// Update positive item
						for d := 0; d < g.dim; d++ {
							g.itemEmbed[target][d] -= currentAlpha * posGrad * lastHidden[d]
						}

						// Negative samples with gBCE
						for n := 0; n < negativeSamples; n++ {
							negItem := g.pnet.NegativeSample(rng)
							if negItem == target {
								continue
							}

							negScore := 0.0
							for d := 0; d < g.dim; d++ {
								negScore += lastHidden[d] * g.itemEmbed[negItem][d]
							}
							negLoss, negGrad := g.generalizedBCE(negScore, 0.0)
							localLoss += negLoss

							// Update negative item
							for d := 0; d < g.dim; d++ {
								g.itemEmbed[negItem][d] -= currentAlpha * negGrad * lastHidden[d]
							}
						}
					}

					countMu.Lock()
					count++
					totalLoss += localLoss
					if count%pronet.Monitor == 0 {
						avgLoss := totalLoss / float64(count)
						fmt.Printf("\tEpoch: %d\tLoss: %.4f\tAlpha: %.6f\tProgress: %.3f %%\r",
							epoch+1, avgLoss, currentAlpha, float64(count)/float64(totalSeqs)*100)
					}
					countMu.Unlock()
				}
			}(start, end)
		}

		wg.Wait()
		avgLoss := totalLoss / float64(count)
		fmt.Printf("\tEpoch: %d\tLoss: %.4f\tAlpha: %.6f\tProgress: 100.00 %%\n",
			epoch+1, avgLoss, currentAlpha)
	}
}

// SaveWeights saves the learned embeddings to a file
func (g *gSASRec) SaveWeights(filename string) error {
	fmt.Println("Save Model:")

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer file.Close()

	fmt.Fprintf(file, "%d %d\n", g.pnet.MaxVid, g.dim)

	for vid := int64(0); vid < g.pnet.MaxVid; vid++ {
		name := g.pnet.GetVertexName(vid)
		fmt.Fprintf(file, "%s", name)
		for d := 0; d < g.dim; d++ {
			fmt.Fprintf(file, " %.6f", g.itemEmbed[vid][d])
		}
		fmt.Fprintln(file)
	}

	fmt.Printf("\tSave to <%s>\n", filename)
	return nil
}
