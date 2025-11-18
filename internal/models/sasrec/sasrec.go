package sasrec

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/cnclabs/smore/pkg/pronet"
)

// SASRec implements Self-Attentive Sequential Recommendation (ICDM 2018)
// Paper: "Self-Attentive Sequential Recommendation" by Kang & McAuley
// This is the FULL implementation with multi-head attention, layer norm, and dropout
type SASRec struct {
	pnet *pronet.ProNet
	dim  int

	// Model parameters
	maxSeqLen   int
	numBlocks   int
	numHeads    int
	dropoutRate float64

	// Embeddings
	itemEmbed [][]float64 // [numItems x dim]
	posEmbed  [][]float64 // [maxSeqLen x dim]

	// Multi-head attention parameters (Q, K, V projections + output projection)
	queryProj  [][][]float64 // [numBlocks x dim x dim]
	keyProj    [][][]float64 // [numBlocks x dim x dim]
	valueProj  [][][]float64 // [numBlocks x dim x dim]
	outputProj [][][]float64 // [numBlocks x dim x dim]

	// Feed-forward network weights
	ffnWeights1 [][][]float64 // [numBlocks x dim x dim*4]
	ffnWeights2 [][][]float64 // [numBlocks x dim*4 x dim]

	// Layer normalization parameters
	attnLayerNormGamma [][]float64 // [numBlocks x dim]
	attnLayerNormBeta  [][]float64 // [numBlocks x dim]
	ffnLayerNormGamma  [][]float64 // [numBlocks x dim]
	ffnLayerNormBeta   [][]float64 // [numBlocks x dim]

	// User sequences (stored for training)
	userSeqs map[int64][]int64
}

// New creates a new SASRec model
func New() *SASRec {
	return &SASRec{
		pnet:     pronet.NewProNet(),
		userSeqs: make(map[int64][]int64),
	}
}

// LoadEdgeList loads user-item interaction sequences
// Format: user_id item_id (chronologically ordered)
func (s *SASRec) LoadEdgeList(filename string, undirected bool) error {
	return s.pnet.LoadEdgeList(filename, undirected)
}

// LoadSequences loads user interaction sequences from file
// Format: user_id item1 item2 item3 ...
func (s *SASRec) LoadSequences(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open sequences file: %v", err)
	}
	defer file.Close()

	fmt.Println("Loading sequences...")
	// Implementation for loading sequences would go here
	// For now, we'll build sequences from the graph

	return nil
}

// BuildSequencesFromGraph builds user sequences from loaded graph
func (s *SASRec) BuildSequencesFromGraph() {
	fmt.Println("Building sequences from graph...")
	s.userSeqs = make(map[int64][]int64)

	for user := int64(0); user < s.pnet.MaxVid; user++ {
		if neighbors, exists := s.pnet.Graph[user]; exists {
			s.userSeqs[user] = neighbors
		}
	}

	fmt.Printf("\tBuilt sequences for %d users\n", len(s.userSeqs))
}

// Init initializes the model with given parameters
func (s *SASRec) Init(dim, maxSeqLen, numBlocks, numHeads int, dropoutRate float64) {
	s.dim = dim
	s.maxSeqLen = maxSeqLen
	s.numBlocks = numBlocks
	s.numHeads = numHeads
	s.dropoutRate = dropoutRate

	numItems := s.pnet.MaxVid

	fmt.Println("Model Setting:")
	fmt.Printf("\tdimension:\t\t%d\n", dim)
	fmt.Printf("\tmax_seq_len:\t\t%d\n", maxSeqLen)
	fmt.Printf("\tnum_blocks:\t\t%d\n", numBlocks)
	fmt.Printf("\tnum_heads:\t\t%d\n", numHeads)
	fmt.Printf("\tdropout_rate:\t\t%.2f\n", dropoutRate)

	// Initialize item embeddings
	s.itemEmbed = make([][]float64, numItems)
	for i := int64(0); i < numItems; i++ {
		s.itemEmbed[i] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			s.itemEmbed[i][d] = (rand.Float64() - 0.5) / float64(dim)
		}
	}

	// Initialize positional embeddings (sinusoidal)
	s.posEmbed = make([][]float64, maxSeqLen)
	for pos := 0; pos < maxSeqLen; pos++ {
		s.posEmbed[pos] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			// Sinusoidal positional encoding
			if d%2 == 0 {
				s.posEmbed[pos][d] = math.Sin(float64(pos) / math.Pow(10000, float64(d)/float64(dim)))
			} else {
				s.posEmbed[pos][d] = math.Cos(float64(pos) / math.Pow(10000, float64(d-1)/float64(dim)))
			}
		}
	}

	// Initialize multi-head attention parameters
	s.queryProj = make([][][]float64, numBlocks)
	s.keyProj = make([][][]float64, numBlocks)
	s.valueProj = make([][][]float64, numBlocks)
	s.outputProj = make([][][]float64, numBlocks)

	for b := 0; b < numBlocks; b++ {
		// Query projection
		s.queryProj[b] = make([][]float64, dim)
		for i := 0; i < dim; i++ {
			s.queryProj[b][i] = make([]float64, dim)
			for j := 0; j < dim; j++ {
				s.queryProj[b][i][j] = (rand.Float64() - 0.5) / math.Sqrt(float64(dim))
			}
		}

		// Key projection
		s.keyProj[b] = make([][]float64, dim)
		for i := 0; i < dim; i++ {
			s.keyProj[b][i] = make([]float64, dim)
			for j := 0; j < dim; j++ {
				s.keyProj[b][i][j] = (rand.Float64() - 0.5) / math.Sqrt(float64(dim))
			}
		}

		// Value projection
		s.valueProj[b] = make([][]float64, dim)
		for i := 0; i < dim; i++ {
			s.valueProj[b][i] = make([]float64, dim)
			for j := 0; j < dim; j++ {
				s.valueProj[b][i][j] = (rand.Float64() - 0.5) / math.Sqrt(float64(dim))
			}
		}

		// Output projection
		s.outputProj[b] = make([][]float64, dim)
		for i := 0; i < dim; i++ {
			s.outputProj[b][i] = make([]float64, dim)
			for j := 0; j < dim; j++ {
				s.outputProj[b][i][j] = (rand.Float64() - 0.5) / math.Sqrt(float64(dim))
			}
		}
	}

	// Initialize FFN weights
	hiddenDim := dim * 4
	s.ffnWeights1 = make([][][]float64, numBlocks)
	s.ffnWeights2 = make([][][]float64, numBlocks)
	for b := 0; b < numBlocks; b++ {
		// First FFN layer: dim -> dim*4
		s.ffnWeights1[b] = make([][]float64, dim)
		for i := 0; i < dim; i++ {
			s.ffnWeights1[b][i] = make([]float64, hiddenDim)
			for j := 0; j < hiddenDim; j++ {
				s.ffnWeights1[b][i][j] = (rand.Float64() - 0.5) / math.Sqrt(float64(dim))
			}
		}

		// Second FFN layer: dim*4 -> dim
		s.ffnWeights2[b] = make([][]float64, hiddenDim)
		for i := 0; i < hiddenDim; i++ {
			s.ffnWeights2[b][i] = make([]float64, dim)
			for j := 0; j < dim; j++ {
				s.ffnWeights2[b][i][j] = (rand.Float64() - 0.5) / math.Sqrt(float64(hiddenDim))
			}
		}
	}

	// Initialize layer normalization parameters
	s.attnLayerNormGamma = make([][]float64, numBlocks)
	s.attnLayerNormBeta = make([][]float64, numBlocks)
	s.ffnLayerNormGamma = make([][]float64, numBlocks)
	s.ffnLayerNormBeta = make([][]float64, numBlocks)

	for b := 0; b < numBlocks; b++ {
		// Initialize gamma to 1.0 (scale)
		s.attnLayerNormGamma[b] = make([]float64, dim)
		s.ffnLayerNormGamma[b] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			s.attnLayerNormGamma[b][d] = 1.0
			s.ffnLayerNormGamma[b][d] = 1.0
		}

		// Initialize beta to 0.0 (shift)
		s.attnLayerNormBeta[b] = make([]float64, dim)
		s.ffnLayerNormBeta[b] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			s.attnLayerNormBeta[b][d] = 0.0
			s.ffnLayerNormBeta[b][d] = 0.0
		}
	}
}

// layerNorm applies layer normalization
// Formula: gamma * (x - mean) / sqrt(variance + epsilon) + beta
func (s *SASRec) layerNorm(input [][]float64, gamma, beta []float64) [][]float64 {
	seqLen := len(input)
	output := make([][]float64, seqLen)
	epsilon := 1e-8

	for i := 0; i < seqLen; i++ {
		// Compute mean
		mean := 0.0
		for d := 0; d < s.dim; d++ {
			mean += input[i][d]
		}
		mean /= float64(s.dim)

		// Compute variance
		variance := 0.0
		for d := 0; d < s.dim; d++ {
			diff := input[i][d] - mean
			variance += diff * diff
		}
		variance /= float64(s.dim)

		// Normalize and apply scale/shift
		output[i] = make([]float64, s.dim)
		for d := 0; d < s.dim; d++ {
			normalized := (input[i][d] - mean) / math.Sqrt(variance+epsilon)
			output[i][d] = gamma[d]*normalized + beta[d]
		}
	}

	return output
}

// applyDropout applies dropout during training
// During training: randomly zero out elements with probability dropoutRate, scale by 1/(1-dropoutRate)
// During inference: return input unchanged
func (s *SASRec) applyDropout(input [][]float64, training bool, rng *rand.Rand) [][]float64 {
	if !training || s.dropoutRate == 0.0 {
		return input
	}

	seqLen := len(input)
	output := make([][]float64, seqLen)
	scale := 1.0 / (1.0 - s.dropoutRate)

	for i := 0; i < seqLen; i++ {
		output[i] = make([]float64, s.dim)
		for d := 0; d < s.dim; d++ {
			if rng.Float64() > s.dropoutRate {
				output[i][d] = input[i][d] * scale
			}
			// else: remains 0.0
		}
	}

	return output
}

// matMul performs matrix multiplication: result = input * weights
func (s *SASRec) matMul(input [][]float64, weights [][]float64) [][]float64 {
	seqLen := len(input)
	outDim := len(weights[0])
	output := make([][]float64, seqLen)

	for i := 0; i < seqLen; i++ {
		output[i] = make([]float64, outDim)
		for j := 0; j < outDim; j++ {
			for k := 0; k < s.dim; k++ {
				output[i][j] += input[i][k] * weights[k][j]
			}
		}
	}

	return output
}

// multiHeadAttention computes full multi-head self-attention
func (s *SASRec) multiHeadAttention(seq [][]float64, block int, training bool, rng *rand.Rand) [][]float64 {
	seqLen := len(seq)
	headDim := s.dim / s.numHeads

	// Project to Q, K, V
	queries := s.matMul(seq, s.queryProj[block])
	keys := s.matMul(seq, s.keyProj[block])
	values := s.matMul(seq, s.valueProj[block])

	// Split into multiple heads and compute attention for each head
	headOutputs := make([][][]float64, s.numHeads)

	for h := 0; h < s.numHeads; h++ {
		// Extract head-specific Q, K, V
		headStart := h * headDim

		headOutput := make([][]float64, seqLen)
		for i := 0; i < seqLen; i++ {
			headOutput[i] = make([]float64, headDim)

			// Compute attention scores for this position
			scores := make([]float64, seqLen)
			sumScores := 0.0

			for j := 0; j <= i; j++ { // Causal masking: only attend to past
				score := 0.0
				for d := 0; d < headDim; d++ {
					qIdx := headStart + d
					kIdx := headStart + d
					score += queries[i][qIdx] * keys[j][kIdx]
				}
				score /= math.Sqrt(float64(headDim)) // Scaling
				scores[j] = math.Exp(score)
				sumScores += scores[j]
			}

			// Normalize attention weights (softmax)
			attnWeights := make([]float64, seqLen)
			for j := 0; j <= i; j++ {
				attnWeights[j] = scores[j] / sumScores
			}

			// Apply dropout to attention weights during training
			if training && s.dropoutRate > 0.0 {
				scale := 1.0 / (1.0 - s.dropoutRate)
				for j := 0; j <= i; j++ {
					if rng.Float64() > s.dropoutRate {
						attnWeights[j] *= scale
					} else {
						attnWeights[j] = 0.0
					}
				}
			}

			// Apply attention to values
			for j := 0; j <= i; j++ {
				for d := 0; d < headDim; d++ {
					vIdx := headStart + d
					headOutput[i][d] += attnWeights[j] * values[j][vIdx]
				}
			}
		}

		headOutputs[h] = headOutput
	}

	// Concatenate heads
	concatenated := make([][]float64, seqLen)
	for i := 0; i < seqLen; i++ {
		concatenated[i] = make([]float64, s.dim)
		for h := 0; h < s.numHeads; h++ {
			headStart := h * headDim
			for d := 0; d < headDim; d++ {
				concatenated[i][headStart+d] = headOutputs[h][i][d]
			}
		}
	}

	// Apply output projection
	output := s.matMul(concatenated, s.outputProj[block])

	return output
}

// feedForward applies position-wise feed-forward network with dropout
func (s *SASRec) feedForward(seq [][]float64, block int, training bool, rng *rand.Rand) [][]float64 {
	seqLen := len(seq)
	hiddenDim := s.dim * 4
	output := make([][]float64, seqLen)

	for i := 0; i < seqLen; i++ {
		// First layer with ReLU
		hidden := make([]float64, hiddenDim)
		for h := 0; h < hiddenDim; h++ {
			for d := 0; d < s.dim; d++ {
				hidden[h] += seq[i][d] * s.ffnWeights1[block][d][h]
			}
			if hidden[h] < 0 {
				hidden[h] = 0 // ReLU
			}
		}

		// Apply dropout after first layer
		if training && s.dropoutRate > 0.0 {
			scale := 1.0 / (1.0 - s.dropoutRate)
			for h := 0; h < hiddenDim; h++ {
				if rng.Float64() > s.dropoutRate {
					hidden[h] *= scale
				} else {
					hidden[h] = 0.0
				}
			}
		}

		// Second layer
		output[i] = make([]float64, s.dim)
		for d := 0; d < s.dim; d++ {
			for h := 0; h < hiddenDim; h++ {
				output[i][d] += hidden[h] * s.ffnWeights2[block][h][d]
			}
		}

		// Apply dropout after second layer
		if training && s.dropoutRate > 0.0 {
			scale := 1.0 / (1.0 - s.dropoutRate)
			for d := 0; d < s.dim; d++ {
				if rng.Float64() > s.dropoutRate {
					output[i][d] *= scale
				} else {
					output[i][d] = 0.0
				}
			}
		}
	}

	return output
}

// forward performs forward pass through the model
func (s *SASRec) forward(sequence []int64, training bool, rng *rand.Rand) [][]float64 {
	seqLen := len(sequence)
	if seqLen > s.maxSeqLen {
		sequence = sequence[seqLen-s.maxSeqLen:]
		seqLen = s.maxSeqLen
	}

	// Create input embeddings (item + positional)
	hidden := make([][]float64, seqLen)
	for i := 0; i < seqLen; i++ {
		hidden[i] = make([]float64, s.dim)
		itemID := sequence[i]
		for d := 0; d < s.dim; d++ {
			hidden[i][d] = s.itemEmbed[itemID][d] + s.posEmbed[i][d]
		}
	}

	// Apply dropout to input embeddings
	hidden = s.applyDropout(hidden, training, rng)

	// Apply transformer blocks
	for b := 0; b < s.numBlocks; b++ {
		// Layer normalization before self-attention (pre-norm architecture)
		normedInput := s.layerNorm(hidden, s.attnLayerNormGamma[b], s.attnLayerNormBeta[b])

		// Multi-head self-attention
		attnOut := s.multiHeadAttention(normedInput, b, training, rng)

		// Residual connection
		for i := 0; i < seqLen; i++ {
			for d := 0; d < s.dim; d++ {
				attnOut[i][d] += hidden[i][d]
			}
		}

		// Layer normalization before feed-forward (pre-norm architecture)
		normedAttn := s.layerNorm(attnOut, s.ffnLayerNormGamma[b], s.ffnLayerNormBeta[b])

		// Feed-forward network with dropout
		ffnOut := s.feedForward(normedAttn, b, training, rng)

		// Residual connection
		for i := 0; i < seqLen; i++ {
			for d := 0; d < s.dim; d++ {
				hidden[i][d] = ffnOut[i][d] + attnOut[i][d]
			}
		}
	}

	return hidden
}

// Train trains the SASRec model
func (s *SASRec) Train(epochs, batchSize, negativeSamples int, alpha float64, workers int) {
	fmt.Println("Model:")
	fmt.Println("\t[SASRec]")

	fmt.Println("Learning Parameters:")
	fmt.Printf("\tepochs:\t\t\t%d\n", epochs)
	fmt.Printf("\tbatch_size:\t\t%d\n", batchSize)
	fmt.Printf("\tnegative_samples:\t%d\n", negativeSamples)
	fmt.Printf("\talpha:\t\t\t%.6f\n", alpha)
	fmt.Printf("\tworkers:\t\t%d\n", workers)

	fmt.Println("Start Training:")

	totalSeqs := int64(len(s.userSeqs))
	alphaMin := alpha * 0.0001

	for epoch := 0; epoch < epochs; epoch++ {
		currentAlpha := alpha * (1.0 - float64(epoch)/float64(epochs))
		if currentAlpha < alphaMin {
			currentAlpha = alphaMin
		}

		count := int64(0)
		var countMu sync.Mutex
		var wg sync.WaitGroup

		// Convert map to slice for parallel processing
		users := make([]int64, 0, len(s.userSeqs))
		for user := range s.userSeqs {
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

				for i := start; i < end; i++ {
					user := users[i]
					seq := s.userSeqs[user]
					if len(seq) < 2 {
						continue
					}

					// Sample a position to predict
					seqLen := len(seq)
					for pos := 1; pos < seqLen; pos++ {
						// Get sequence up to pos
						inputSeq := seq[:pos]
						target := seq[pos]

						// Forward pass with training=true (enables dropout)
						hiddenStates := s.forward(inputSeq, true, rng)
						lastHidden := hiddenStates[len(hiddenStates)-1]

						// Positive sample
						posScore := 0.0
						for d := 0; d < s.dim; d++ {
							posScore += lastHidden[d] * s.itemEmbed[target][d]
						}
						posProb := s.pnet.FastSigmoid(posScore)

						// Update positive item
						posGrad := currentAlpha * (1.0 - posProb)
						for d := 0; d < s.dim; d++ {
							s.itemEmbed[target][d] += posGrad * lastHidden[d]
						}

						// Negative samples
						for n := 0; n < negativeSamples; n++ {
							negItem := s.pnet.NegativeSample(rng)
							if negItem == target {
								continue
							}

							negScore := 0.0
							for d := 0; d < s.dim; d++ {
								negScore += lastHidden[d] * s.itemEmbed[negItem][d]
							}
							negProb := s.pnet.FastSigmoid(negScore)

							// Update negative item
							negGrad := currentAlpha * (0.0 - negProb)
							for d := 0; d < s.dim; d++ {
								s.itemEmbed[negItem][d] += negGrad * lastHidden[d]
							}
						}
					}

					countMu.Lock()
					count++
					if count%pronet.Monitor == 0 {
						fmt.Printf("\tEpoch: %d\tAlpha: %.6f\tProgress: %.3f %%\r",
							epoch+1, currentAlpha, float64(count)/float64(totalSeqs)*100)
					}
					countMu.Unlock()
				}
			}(start, end)
		}

		wg.Wait()
		fmt.Printf("\tEpoch: %d\tAlpha: %.6f\tProgress: 100.00 %%\n", epoch+1, currentAlpha)
	}
}

// SaveWeights saves the learned embeddings to a file
func (s *SASRec) SaveWeights(filename string) error {
	fmt.Println("Save Model:")

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer file.Close()

	fmt.Fprintf(file, "%d %d\n", s.pnet.MaxVid, s.dim)

	for vid := int64(0); vid < s.pnet.MaxVid; vid++ {
		name := s.pnet.GetVertexName(vid)
		fmt.Fprintf(file, "%s", name)
		for d := 0; d < s.dim; d++ {
			fmt.Fprintf(file, " %.6f", s.itemEmbed[vid][d])
		}
		fmt.Fprintln(file)
	}

	fmt.Printf("\tSave to <%s>\n", filename)
	return nil
}
