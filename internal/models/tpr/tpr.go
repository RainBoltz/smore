package tpr

import (
	"fmt"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/cnclabs/smore/pkg/pronet"
)

// TPR implements Text-aware Preference Ranking
// Paper: "TPR: Text-aware Preference Ranking for Recommender Systems" (CIKM 2020)
// Combines collaborative filtering (user-item) with content-based features (item-word)
type TPR struct {
	// User-item graph
	uiGraph *pronet.ProNet

	// Item-word graph (text features)
	iwGraph *pronet.ProNet

	dim int

	// Embeddings
	userEmbed [][]float64 // User embeddings
	itemEmbed [][]float64 // Item embeddings (shared between UI and IW graphs)
	wordEmbed [][]float64 // Word embeddings

	// Text integration weight
	textWeight float64 // Weight for text-aware component
}

// New creates a new TPR model
func New() *TPR {
	return &TPR{
		uiGraph:    pronet.NewProNet(),
		iwGraph:    pronet.NewProNet(),
		textWeight: 0.5, // Balance collaborative and content signals
	}
}

// LoadUserItemGraph loads the user-item interaction graph
func (t *TPR) LoadUserItemGraph(filename string, undirected bool) error {
	fmt.Println("Loading user-item graph...")
	return t.uiGraph.LoadEdgeList(filename, undirected)
}

// LoadItemWordGraph loads the item-word feature graph
func (t *TPR) LoadItemWordGraph(filename string, undirected bool) error {
	fmt.Println("Loading item-word graph...")
	return t.iwGraph.LoadEdgeList(filename, undirected)
}

// Init initializes the TPR model
func (t *TPR) Init(dim int, textWeight float64) {
	t.dim = dim
	t.textWeight = textWeight

	numUsers := t.uiGraph.MaxVid
	numItems := t.uiGraph.MaxVid // Items are in both graphs
	numWords := t.iwGraph.MaxVid

	fmt.Println("Model Setting:")
	fmt.Printf("\tdimension:\t\t%d\n", dim)
	fmt.Printf("\ttext_weight:\t\t%.2f\n", textWeight)
	fmt.Printf("\tnum_users:\t\t%d\n", numUsers)
	fmt.Printf("\tnum_items:\t\t%d\n", numItems)
	fmt.Printf("\tnum_words:\t\t%d\n", numWords)

	// Initialize user embeddings
	t.userEmbed = make([][]float64, numUsers)
	for uid := int64(0); uid < numUsers; uid++ {
		t.userEmbed[uid] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			t.userEmbed[uid][d] = (rand.Float64() - 0.5) / float64(dim)
		}
	}

	// Initialize item embeddings (shared between UI and IW)
	t.itemEmbed = make([][]float64, numItems)
	for iid := int64(0); iid < numItems; iid++ {
		t.itemEmbed[iid] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			t.itemEmbed[iid][d] = (rand.Float64() - 0.5) / float64(dim)
		}
	}

	// Initialize word embeddings
	t.wordEmbed = make([][]float64, numWords)
	for wid := int64(0); wid < numWords; wid++ {
		t.wordEmbed[wid] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			t.wordEmbed[wid][d] = (rand.Float64() - 0.5) / float64(dim)
		}
	}
}

// getTextEnrichedItemEmbedding computes text-enriched item representation
// Combines base item embedding with aggregated word embeddings
func (t *TPR) getTextEnrichedItemEmbedding(itemID int64, output []float64) {
	// Start with base item embedding
	for d := 0; d < t.dim; d++ {
		output[d] = (1.0 - t.textWeight) * t.itemEmbed[itemID][d]
	}

	// Add text component: aggregate word embeddings
	if words, exists := t.iwGraph.Graph[itemID]; exists && len(words) > 0 {
		for _, wordID := range words {
			for d := 0; d < t.dim; d++ {
				output[d] += (t.textWeight / float64(len(words))) * t.wordEmbed[wordID][d]
			}
		}
	} else {
		// No text features: use only item embedding
		for d := 0; d < t.dim; d++ {
			output[d] = t.itemEmbed[itemID][d]
		}
	}
}

// Train trains the TPR model using BPR-style ranking with text-aware features
func (t *TPR) Train(sampleTimes int, alpha, lambda float64, workers int) {
	fmt.Println("Model:")
	fmt.Println("\t[TPR: Text-aware Preference Ranking]")

	fmt.Println("Learning Parameters:")
	fmt.Printf("\tsample_times:\t\t%d\n", sampleTimes)
	fmt.Printf("\talpha:\t\t\t%.6f\n", alpha)
	fmt.Printf("\tlambda:\t\t\t%.6f\n", lambda)
	fmt.Printf("\tworkers:\t\t%d\n", workers)

	fmt.Println("Start Training:")

	total := int64(sampleTimes) * t.uiGraph.MaxLine
	alphaMin := alpha * 0.0001
	currentAlpha := alpha
	count := int64(0)

	var countMu sync.Mutex
	var wg sync.WaitGroup
	chunkSize := (sampleTimes + workers - 1) / workers

	for w := 0; w < workers; w++ {
		wg.Add(1)
		start := w * chunkSize
		end := start + chunkSize
		if end > sampleTimes {
			end = sampleTimes
		}

		go func(start, end int) {
			defer wg.Done()

			rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(start)))

			// Buffers for text-enriched embeddings
			posItemVec := make([]float64, t.dim)
			negItemVec := make([]float64, t.dim)
			userGrad := make([]float64, t.dim)
			posGrad := make([]float64, t.dim)
			negGrad := make([]float64, t.dim)

			for i := start; i < end; i++ {
				for j := int64(0); j < t.uiGraph.MaxLine; j++ {
					// Sample user
					user := t.uiGraph.SourceSample(rng)
					if user == -1 {
						continue
					}

					// Sample positive item
					posItem := t.uiGraph.TargetSample(user, rng)
					if posItem == -1 {
						continue
					}

					// Sample negative item
					negItem := t.uiGraph.NegativeSample(rng)

					// Get text-enriched item representations
					t.getTextEnrichedItemEmbedding(posItem, posItemVec)
					t.getTextEnrichedItemEmbedding(negItem, negItemVec)

					// Compute scores
					posScore := 0.0
					negScore := 0.0
					for d := 0; d < t.dim; d++ {
						posScore += t.userEmbed[user][d] * posItemVec[d]
						negScore += t.userEmbed[user][d] * negItemVec[d]
					}

					// BPR gradient
					diff := negScore - posScore
					gradCoef := currentAlpha * t.uiGraph.FastSigmoid(diff)

					// Reset gradients
					for d := 0; d < t.dim; d++ {
						userGrad[d] = 0.0
						posGrad[d] = 0.0
						negGrad[d] = 0.0
					}

					// Compute gradients
					for d := 0; d < t.dim; d++ {
						userGrad[d] = gradCoef * (posItemVec[d] - negItemVec[d])
						posGrad[d] = gradCoef * t.userEmbed[user][d]
						negGrad[d] = -gradCoef * t.userEmbed[user][d]
					}

					// Update user embedding with L2 regularization
					for d := 0; d < t.dim; d++ {
						t.userEmbed[user][d] += userGrad[d] - lambda*currentAlpha*t.userEmbed[user][d]
					}

					// Update item embeddings (base component)
					for d := 0; d < t.dim; d++ {
						t.itemEmbed[posItem][d] += (1.0-t.textWeight)*posGrad[d] - lambda*currentAlpha*t.itemEmbed[posItem][d]
						t.itemEmbed[negItem][d] += (1.0-t.textWeight)*negGrad[d] - lambda*currentAlpha*t.itemEmbed[negItem][d]
					}

					// Update word embeddings (text component)
					if posWords, exists := t.iwGraph.Graph[posItem]; exists && len(posWords) > 0 {
						wordWeight := t.textWeight / float64(len(posWords))
						for _, wordID := range posWords {
							for d := 0; d < t.dim; d++ {
								t.wordEmbed[wordID][d] += wordWeight*posGrad[d] - lambda*currentAlpha*t.wordEmbed[wordID][d]
							}
						}
					}

					if negWords, exists := t.iwGraph.Graph[negItem]; exists && len(negWords) > 0 {
						wordWeight := t.textWeight / float64(len(negWords))
						for _, wordID := range negWords {
							for d := 0; d < t.dim; d++ {
								t.wordEmbed[wordID][d] += wordWeight*negGrad[d] - lambda*currentAlpha*t.wordEmbed[wordID][d]
							}
						}
					}

					// Update progress
					countMu.Lock()
					count++
					if count%pronet.Monitor == 0 {
						currentAlpha = alpha * (1.0 - float64(count)/float64(total))
						if currentAlpha < alphaMin {
							currentAlpha = alphaMin
						}
						fmt.Printf("\tAlpha: %.6f\tProgress: %.3f %%\r", currentAlpha, float64(count)/float64(total)*100)
					}
					countMu.Unlock()
				}
			}
		}(start, end)
	}

	wg.Wait()
	fmt.Printf("\tAlpha: %.6f\tProgress: 100.00 %%\n", currentAlpha)
}

// SaveWeights saves the learned embeddings to files
func (t *TPR) SaveWeights(userFile, itemFile, wordFile string) error {
	fmt.Println("Save Model:")

	// Save user embeddings
	if userFile != "" {
		file, err := os.Create(userFile)
		if err != nil {
			return fmt.Errorf("failed to create user file: %v", err)
		}
		defer file.Close()

		fmt.Fprintf(file, "%d %d\n", len(t.userEmbed), t.dim)
		for uid, emb := range t.userEmbed {
			name := t.uiGraph.GetVertexName(int64(uid))
			fmt.Fprintf(file, "%s", name)
			for d := 0; d < t.dim; d++ {
				fmt.Fprintf(file, " %.6f", emb[d])
			}
			fmt.Fprintln(file)
		}
		fmt.Printf("\tUsers saved to <%s>\n", userFile)
	}

	// Save item embeddings
	if itemFile != "" {
		file, err := os.Create(itemFile)
		if err != nil {
			return fmt.Errorf("failed to create item file: %v", err)
		}
		defer file.Close()

		fmt.Fprintf(file, "%d %d\n", len(t.itemEmbed), t.dim)
		for iid, emb := range t.itemEmbed {
			name := t.uiGraph.GetVertexName(int64(iid))
			fmt.Fprintf(file, "%s", name)
			for d := 0; d < t.dim; d++ {
				fmt.Fprintf(file, " %.6f", emb[d])
			}
			fmt.Fprintln(file)
		}
		fmt.Printf("\tItems saved to <%s>\n", itemFile)
	}

	// Save word embeddings
	if wordFile != "" {
		file, err := os.Create(wordFile)
		if err != nil {
			return fmt.Errorf("failed to create word file: %v", err)
		}
		defer file.Close()

		fmt.Fprintf(file, "%d %d\n", len(t.wordEmbed), t.dim)
		for wid, emb := range t.wordEmbed {
			name := t.iwGraph.GetVertexName(int64(wid))
			fmt.Fprintf(file, "%s", name)
			for d := 0; d < t.dim; d++ {
				fmt.Fprintf(file, " %.6f", emb[d])
			}
			fmt.Fprintln(file)
		}
		fmt.Printf("\tWords saved to <%s>\n", wordFile)
	}

	return nil
}
