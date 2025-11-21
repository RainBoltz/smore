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

	// Attention layers (simplified: using single weight matrix per block)
	attnWeights [][][]float64 // [numBlocks x dim x dim]
	ffnWeights1 [][][]float64 // [numBlocks x dim x dim*4]
	ffnWeights2 [][][]float64 // [numBlocks x dim*4 x dim]

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

	// Initialize positional embeddings
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

	// Initialize attention weights
	s.attnWeights = make([][][]float64, numBlocks)
	for b := 0; b < numBlocks; b++ {
		s.attnWeights[b] = make([][]float64, dim)
		for i := 0; i < dim; i++ {
			s.attnWeights[b][i] = make([]float64, dim)
			for j := 0; j < dim; j++ {
				s.attnWeights[b][i][j] = (rand.Float64() - 0.5) / math.Sqrt(float64(dim))
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
}

// selfAttention computes simplified self-attention
func (s *SASRec) selfAttention(seq [][]float64, block int) [][]float64 {
	seqLen := len(seq)
	output := make([][]float64, seqLen)

	for i := 0; i < seqLen; i++ {
		output[i] = make([]float64, s.dim)

		// Compute attention scores
		scores := make([]float64, seqLen)
		sumScores := 0.0
		for j := 0; j <= i; j++ { // Causal masking
			score := 0.0
			for d := 0; d < s.dim; d++ {
				for k := 0; k < s.dim; k++ {
					score += seq[i][d] * s.attnWeights[block][d][k] * seq[j][k]
				}
			}
			score /= math.Sqrt(float64(s.dim))
			scores[j] = math.Exp(score)
			sumScores += scores[j]
		}

		// Normalize and apply attention
		for j := 0; j <= i; j++ {
			weight := scores[j] / sumScores
			for d := 0; d < s.dim; d++ {
				output[i][d] += weight * seq[j][d]
			}
		}
	}

	return output
}

// feedForward applies position-wise feed-forward network
func (s *SASRec) feedForward(seq [][]float64, block int) [][]float64 {
	seqLen := len(seq)
	output := make([][]float64, seqLen)
	hiddenDim := s.dim * 4

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

		// Second layer
		output[i] = make([]float64, s.dim)
		for d := 0; d < s.dim; d++ {
			for h := 0; h < hiddenDim; h++ {
				output[i][d] += hidden[h] * s.ffnWeights2[block][h][d]
			}
		}
	}

	return output
}

// forward performs forward pass through the model
func (s *SASRec) forward(sequence []int64) [][]float64 {
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

	// Apply transformer blocks
	for b := 0; b < s.numBlocks; b++ {
		// Self-attention with residual
		attnOut := s.selfAttention(hidden, b)
		for i := 0; i < seqLen; i++ {
			for d := 0; d < s.dim; d++ {
				attnOut[i][d] += hidden[i][d] // Residual connection
			}
		}

		// Feed-forward with residual
		ffnOut := s.feedForward(attnOut, b)
		for i := 0; i < seqLen; i++ {
			for d := 0; d < s.dim; d++ {
				hidden[i][d] = ffnOut[i][d] + attnOut[i][d] // Residual connection
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

						// Forward pass
						hiddenStates := s.forward(inputSeq)
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
