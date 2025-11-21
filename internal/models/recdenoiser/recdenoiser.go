package recdenoiser

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/cnclabs/smore/pkg/pronet"
)

// RecDenoiser implements Denoising Self-Attentive Sequential Recommendation
// Paper: "Denoising Self-Attentive Sequential Recommendation" (RecSys 2022 Best Paper)
// By Huiyuan Chen, Yusan Lin, Menghai Pan, et al.
//
// Key Innovation: Trainable binary masks to prune noisy attentions,
// resulting in sparse and clean attention distributions.
type RecDenoiser struct {
	pnet *pronet.ProNet
	dim  int

	// Model parameters
	maxSeqLen   int
	numBlocks   int
	numHeads    int
	dropoutRate float64

	// Denoising parameters
	maskTemp      float64 // Temperature for mask training
	maskThreshold float64 // Threshold for binary mask (0.5 typical)
	sparsityRate  float64 // Target sparsity rate (e.g., 0.3 = 30% pruned)

	// Embeddings
	itemEmbed [][]float64 // [numItems x dim]
	posEmbed  [][]float64 // [maxSeqLen x dim]

	// Attention layers with denoising masks
	attnWeights    [][][]float64   // [numBlocks x dim x dim]
	attnMasks      [][][]float64   // [numBlocks x maxSeqLen x maxSeqLen] - Trainable masks!
	attnMaskLogits [][][]float64   // [numBlocks x maxSeqLen x maxSeqLen] - Logits for Gumbel
	ffnWeights1    [][][]float64   // [numBlocks x dim x dim*4]
	ffnWeights2    [][][]float64   // [numBlocks x dim*4 x dim]

	// User sequences
	userSeqs map[int64][]int64
}

// New creates a new RecDenoiser model
func New() *RecDenoiser {
	return &RecDenoiser{
		pnet:          pronet.NewProNet(),
		userSeqs:      make(map[int64][]int64),
		maskTemp:      1.0,  // Temperature for Gumbel-Softmax
		maskThreshold: 0.5,  // Binary threshold
		sparsityRate:  0.3,  // 30% sparsity
	}
}

// LoadEdgeList loads user-item interaction sequences
func (r *RecDenoiser) LoadEdgeList(filename string, undirected bool) error {
	return r.pnet.LoadEdgeList(filename, undirected)
}

// BuildSequencesFromGraph builds user sequences from loaded graph
func (r *RecDenoiser) BuildSequencesFromGraph() {
	fmt.Println("Building sequences from graph...")
	r.userSeqs = make(map[int64][]int64)

	for user := int64(0); user < r.pnet.MaxVid; user++ {
		if neighbors, exists := r.pnet.Graph[user]; exists {
			r.userSeqs[user] = neighbors
		}
	}

	fmt.Printf("\tBuilt sequences for %d users\n", len(r.userSeqs))
}

// Init initializes the model with given parameters
func (r *RecDenoiser) Init(dim, maxSeqLen, numBlocks, numHeads int, dropoutRate, sparsityRate float64) {
	r.dim = dim
	r.maxSeqLen = maxSeqLen
	r.numBlocks = numBlocks
	r.numHeads = numHeads
	r.dropoutRate = dropoutRate
	r.sparsityRate = sparsityRate

	numItems := r.pnet.MaxVid

	fmt.Println("Model Setting:")
	fmt.Printf("\tdimension:\t\t%d\n", dim)
	fmt.Printf("\tmax_seq_len:\t\t%d\n", maxSeqLen)
	fmt.Printf("\tnum_blocks:\t\t%d\n", numBlocks)
	fmt.Printf("\tnum_heads:\t\t%d\n", numHeads)
	fmt.Printf("\tdropout_rate:\t\t%.2f\n", dropoutRate)
	fmt.Printf("\tsparsity_rate:\t\t%.2f (denoising)\n", sparsityRate)

	// Initialize item embeddings
	r.itemEmbed = make([][]float64, numItems)
	for i := int64(0); i < numItems; i++ {
		r.itemEmbed[i] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			r.itemEmbed[i][d] = (rand.Float64() - 0.5) / float64(dim)
		}
	}

	// Initialize positional embeddings
	r.posEmbed = make([][]float64, maxSeqLen)
	for pos := 0; pos < maxSeqLen; pos++ {
		r.posEmbed[pos] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			if d%2 == 0 {
				r.posEmbed[pos][d] = math.Sin(float64(pos) / math.Pow(10000, float64(d)/float64(dim)))
			} else {
				r.posEmbed[pos][d] = math.Cos(float64(pos) / math.Pow(10000, float64(d-1)/float64(dim)))
			}
		}
	}

	// Initialize attention weights
	r.attnWeights = make([][][]float64, numBlocks)
	for b := 0; b < numBlocks; b++ {
		r.attnWeights[b] = make([][]float64, dim)
		for i := 0; i < dim; i++ {
			r.attnWeights[b][i] = make([]float64, dim)
			for j := 0; j < dim; j++ {
				r.attnWeights[b][i][j] = (rand.Float64() - 0.5) / math.Sqrt(float64(dim))
			}
		}
	}

	// Initialize trainable attention masks (KEY INNOVATION!)
	r.attnMasks = make([][][]float64, numBlocks)
	r.attnMaskLogits = make([][][]float64, numBlocks)
	for b := 0; b < numBlocks; b++ {
		r.attnMasks[b] = make([][]float64, maxSeqLen)
		r.attnMaskLogits[b] = make([][]float64, maxSeqLen)
		for i := 0; i < maxSeqLen; i++ {
			r.attnMasks[b][i] = make([]float64, maxSeqLen)
			r.attnMaskLogits[b][i] = make([]float64, maxSeqLen)
			for j := 0; j < maxSeqLen; j++ {
				// Initialize mask logits (will be learned)
				r.attnMaskLogits[b][i][j] = rand.Float64()*2 - 1 // [-1, 1]
				r.attnMasks[b][i][j] = 1.0                        // Start with no pruning
			}
		}
	}

	// Initialize FFN weights
	hiddenDim := dim * 4
	r.ffnWeights1 = make([][][]float64, numBlocks)
	r.ffnWeights2 = make([][][]float64, numBlocks)
	for b := 0; b < numBlocks; b++ {
		r.ffnWeights1[b] = make([][]float64, dim)
		for i := 0; i < dim; i++ {
			r.ffnWeights1[b][i] = make([]float64, hiddenDim)
			for j := 0; j < hiddenDim; j++ {
				r.ffnWeights1[b][i][j] = (rand.Float64() - 0.5) / math.Sqrt(float64(dim))
			}
		}

		r.ffnWeights2[b] = make([][]float64, hiddenDim)
		for i := 0; i < hiddenDim; i++ {
			r.ffnWeights2[b][i] = make([]float64, dim)
			for j := 0; j < dim; j++ {
				r.ffnWeights2[b][i][j] = (rand.Float64() - 0.5) / math.Sqrt(float64(hiddenDim))
			}
		}
	}
}

// gumbelSigmoid applies Gumbel-Sigmoid for differentiable binary mask training
func (r *RecDenoiser) gumbelSigmoid(logit float64, rng *rand.Rand) float64 {
	// Sample Gumbel noise
	u1 := rng.Float64()
	u2 := rng.Float64()
	gumbel1 := -math.Log(-math.Log(u1 + 1e-10) + 1e-10)
	gumbel2 := -math.Log(-math.Log(u2 + 1e-10) + 1e-10)

	// Apply Gumbel-Softmax trick
	y := (logit + gumbel1 - gumbel2) / r.maskTemp
	return 1.0 / (1.0 + math.Exp(-y))
}

// denoisedSelfAttention computes self-attention with trainable mask pruning
func (r *RecDenoiser) denoisedSelfAttention(seq [][]float64, block int, rng *rand.Rand, training bool) [][]float64 {
	seqLen := len(seq)
	output := make([][]float64, seqLen)

	// Update masks during training
	if training {
		for i := 0; i < seqLen && i < r.maxSeqLen; i++ {
			for j := 0; j <= i && j < r.maxSeqLen; j++ {
				// Use Gumbel-Sigmoid for differentiable binary sampling
				r.attnMasks[block][i][j] = r.gumbelSigmoid(r.attnMaskLogits[block][i][j], rng)
			}
		}
	}

	for i := 0; i < seqLen; i++ {
		output[i] = make([]float64, r.dim)

		// Compute attention scores with denoising mask
		scores := make([]float64, seqLen)
		sumScores := 0.0
		for j := 0; j <= i; j++ {
			// Compute attention score
			score := 0.0
			for d := 0; d < r.dim; d++ {
				for k := 0; k < r.dim; k++ {
					score += seq[i][d] * r.attnWeights[block][d][k] * seq[j][k]
				}
			}
			score /= math.Sqrt(float64(r.dim))

			// Apply trainable mask (KEY: prune noisy attentions!)
			maskIdx := i
			if maskIdx >= r.maxSeqLen {
				maskIdx = r.maxSeqLen - 1
			}
			maskValue := r.attnMasks[block][maskIdx][j]

			// Masked attention
			scores[j] = math.Exp(score) * maskValue
			sumScores += scores[j]
		}

		// Normalize and apply attention
		if sumScores > 0 {
			for j := 0; j <= i; j++ {
				weight := scores[j] / sumScores
				for d := 0; d < r.dim; d++ {
					output[i][d] += weight * seq[j][d]
				}
			}
		} else {
			// Fallback if all masked out
			for d := 0; d < r.dim; d++ {
				output[i][d] = seq[i][d]
			}
		}
	}

	return output
}

// feedForward applies position-wise feed-forward network
func (r *RecDenoiser) feedForward(seq [][]float64, block int) [][]float64 {
	seqLen := len(seq)
	output := make([][]float64, seqLen)
	hiddenDim := r.dim * 4

	for i := 0; i < seqLen; i++ {
		hidden := make([]float64, hiddenDim)
		for h := 0; h < hiddenDim; h++ {
			for d := 0; d < r.dim; d++ {
				hidden[h] += seq[i][d] * r.ffnWeights1[block][d][h]
			}
			if hidden[h] < 0 {
				hidden[h] = 0 // ReLU
			}
		}

		output[i] = make([]float64, r.dim)
		for d := 0; d < r.dim; d++ {
			for h := 0; h < hiddenDim; h++ {
				output[i][d] += hidden[h] * r.ffnWeights2[block][h][d]
			}
		}
	}

	return output
}

// forward performs forward pass with denoising
func (r *RecDenoiser) forward(sequence []int64, rng *rand.Rand, training bool) [][]float64 {
	seqLen := len(sequence)
	if seqLen > r.maxSeqLen {
		sequence = sequence[seqLen-r.maxSeqLen:]
		seqLen = r.maxSeqLen
	}

	// Create input embeddings
	hidden := make([][]float64, seqLen)
	for i := 0; i < seqLen; i++ {
		hidden[i] = make([]float64, r.dim)
		itemID := sequence[i]
		for d := 0; d < r.dim; d++ {
			hidden[i][d] = r.itemEmbed[itemID][d] + r.posEmbed[i][d]
		}
	}

	// Apply denoising transformer blocks
	for b := 0; b < r.numBlocks; b++ {
		// Denoised self-attention with trainable masks
		attnOut := r.denoisedSelfAttention(hidden, b, rng, training)
		for i := 0; i < seqLen; i++ {
			for d := 0; d < r.dim; d++ {
				attnOut[i][d] += hidden[i][d] // Residual
			}
		}

		// Feed-forward
		ffnOut := r.feedForward(attnOut, b)
		for i := 0; i < seqLen; i++ {
			for d := 0; d < r.dim; d++ {
				hidden[i][d] = ffnOut[i][d] + attnOut[i][d] // Residual
			}
		}
	}

	return hidden
}

// computeSparsityLoss computes L1 regularization on masks to encourage sparsity
func (r *RecDenoiser) computeSparsityLoss() float64 {
	loss := 0.0
	count := 0

	for b := 0; b < r.numBlocks; b++ {
		for i := 0; i < r.maxSeqLen; i++ {
			for j := 0; j < r.maxSeqLen; j++ {
				loss += math.Abs(r.attnMasks[b][i][j])
				count++
			}
		}
	}

	return loss / float64(count)
}

// Train trains the RecDenoiser model with denoising
func (r *RecDenoiser) Train(epochs, batchSize, negativeSamples int, alpha, lambdaSparsity float64, workers int) {
	fmt.Println("Model:")
	fmt.Println("\t[Rec-Denoiser - RecSys 2022 Best Paper]")

	fmt.Println("Learning Parameters:")
	fmt.Printf("\tepochs:\t\t\t%d\n", epochs)
	fmt.Printf("\tbatch_size:\t\t%d\n", batchSize)
	fmt.Printf("\tnegative_samples:\t%d\n", negativeSamples)
	fmt.Printf("\talpha:\t\t\t%.6f\n", alpha)
	fmt.Printf("\tlambda_sparsity:\t%.6f\n", lambdaSparsity)
	fmt.Printf("\tworkers:\t\t%d\n", workers)

	fmt.Println("Start Training:")

	totalSeqs := int64(len(r.userSeqs))
	alphaMin := alpha * 0.0001

	for epoch := 0; epoch < epochs; epoch++ {
		currentAlpha := alpha * (1.0 - float64(epoch)/float64(epochs))
		if currentAlpha < alphaMin {
			currentAlpha = alphaMin
		}

		count := int64(0)
		totalLoss := 0.0
		totalSparsity := 0.0
		var countMu sync.Mutex
		var wg sync.WaitGroup

		users := make([]int64, 0, len(r.userSeqs))
		for user := range r.userSeqs {
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
				localSparsity := 0.0

				for i := start; i < end; i++ {
					user := users[i]
					seq := r.userSeqs[user]
					if len(seq) < 2 {
						continue
					}

					seqLen := len(seq)
					for pos := 1; pos < seqLen; pos++ {
						inputSeq := seq[:pos]
						target := seq[pos]

						// Forward with denoising (training mode)
						hiddenStates := r.forward(inputSeq, rng, true)
						lastHidden := hiddenStates[len(hiddenStates)-1]

						// Positive sample
						posScore := 0.0
						for d := 0; d < r.dim; d++ {
							posScore += lastHidden[d] * r.itemEmbed[target][d]
						}
						posProb := r.pnet.FastSigmoid(posScore)
						posLoss := -math.Log(posProb + 1e-10)
						localLoss += posLoss

						// Update
						posGrad := currentAlpha * (1.0 - posProb)
						for d := 0; d < r.dim; d++ {
							r.itemEmbed[target][d] += posGrad * lastHidden[d]
						}

						// Negative samples
						for n := 0; n < negativeSamples; n++ {
							negItem := r.pnet.NegativeSample(rng)
							if negItem == target {
								continue
							}

							negScore := 0.0
							for d := 0; d < r.dim; d++ {
								negScore += lastHidden[d] * r.itemEmbed[negItem][d]
							}
							negProb := r.pnet.FastSigmoid(negScore)
							negLoss := -math.Log(1.0 - negProb + 1e-10)
							localLoss += negLoss

							negGrad := currentAlpha * (0.0 - negProb)
							for d := 0; d < r.dim; d++ {
								r.itemEmbed[negItem][d] += negGrad * lastHidden[d]
							}
						}

						// Update mask logits (encourage sparsity)
						sparsityLoss := r.computeSparsityLoss()
						localSparsity += sparsityLoss

						// Apply sparsity regularization to mask logits
						for b := 0; b < r.numBlocks; b++ {
							for ii := 0; ii < r.maxSeqLen; ii++ {
								for jj := 0; jj < r.maxSeqLen; jj++ {
									// L1 penalty on mask values
									maskGrad := lambdaSparsity * currentAlpha
									if r.attnMasks[b][ii][jj] > 0 {
										r.attnMaskLogits[b][ii][jj] -= maskGrad
									} else {
										r.attnMaskLogits[b][ii][jj] += maskGrad
									}
								}
							}
						}
					}

					countMu.Lock()
					count++
					totalLoss += localLoss
					totalSparsity += localSparsity
					if count%pronet.Monitor == 0 {
						avgLoss := totalLoss / float64(count)
						avgSparsity := totalSparsity / float64(count)
						fmt.Printf("\tEpoch: %d\tLoss: %.4f\tSparsity: %.4f\tAlpha: %.6f\tProgress: %.3f %%\r",
							epoch+1, avgLoss, avgSparsity, currentAlpha, float64(count)/float64(totalSeqs)*100)
					}
					countMu.Unlock()
				}
			}(start, end)
		}

		wg.Wait()
		avgLoss := totalLoss / float64(count)
		avgSparsity := totalSparsity / float64(count)
		fmt.Printf("\tEpoch: %d\tLoss: %.4f\tSparsity: %.4f\tAlpha: %.6f\tProgress: 100.00 %%\n",
			epoch+1, avgLoss, avgSparsity, currentAlpha)
	}
}

// SaveWeights saves the learned embeddings
func (r *RecDenoiser) SaveWeights(filename string) error {
	fmt.Println("Save Model:")

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer file.Close()

	fmt.Fprintf(file, "%d %d\n", r.pnet.MaxVid, r.dim)

	for vid := int64(0); vid < r.pnet.MaxVid; vid++ {
		name := r.pnet.GetVertexName(vid)
		fmt.Fprintf(file, "%s", name)
		for d := 0; d < r.dim; d++ {
			fmt.Fprintf(file, " %.6f", r.itemEmbed[vid][d])
		}
		fmt.Fprintln(file)
	}

	fmt.Printf("\tSave to <%s>\n", filename)
	return nil
}
