package cpr

import (
	"fmt"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/cnclabs/smore/pkg/pronet"
)

// CPR implements Cross-Domain Preference Ranking with User Transformation
// This model learns user preferences across two domains (source and target)
// by transforming user representations
type CPR struct {
	// Dual graphs for cross-domain learning
	targetGraph *pronet.ProNet // Target domain (main prediction domain)
	sourceGraph *pronet.ProNet // Source domain (auxiliary domain)

	// Embeddings
	dim          int
	userEmbed    [][]float64 // User embeddings (shared across domains)
	targetEmbed  [][]float64 // Target domain item embeddings
	sourceEmbed  [][]float64 // Source domain item embeddings

	// User to items mapping for transformation
	userToTarget map[int64][]int64 // User's target domain items
	userToSource map[int64][]int64 // User's source domain items

	// User ID mapping (unified across domains)
	userIDs      []int64
	maxUserID    int64
}

// New creates a new CPR model
func New() *CPR {
	return &CPR{
		targetGraph:  pronet.NewProNet(),
		sourceGraph:  pronet.NewProNet(),
		userToTarget: make(map[int64][]int64),
		userToSource: make(map[int64][]int64),
	}
}

// LoadTargetDomain loads the target domain graph
func (cpr *CPR) LoadTargetDomain(filename string, undirected bool) error {
	fmt.Println("Loading target domain graph...")
	if err := cpr.targetGraph.LoadEdgeList(filename, undirected); err != nil {
		return err
	}

	// Build user to target items mapping
	for vid := int64(0); vid < cpr.targetGraph.MaxVid; vid++ {
		neighbors := cpr.targetGraph.GetNeighbors(vid)
		if len(neighbors) > 0 {
			cpr.userToTarget[vid] = neighbors
		}
	}

	return nil
}

// LoadSourceDomain loads the source domain graph
func (cpr *CPR) LoadSourceDomain(filename string, undirected bool) error {
	fmt.Println("Loading source domain graph...")
	if err := cpr.sourceGraph.LoadEdgeList(filename, undirected); err != nil {
		return err
	}

	// Build user to source items mapping
	for vid := int64(0); vid < cpr.sourceGraph.MaxVid; vid++ {
		neighbors := cpr.sourceGraph.GetNeighbors(vid)
		if len(neighbors) > 0 {
			cpr.userToSource[vid] = neighbors
		}
	}

	return nil
}

// Init initializes the CPR model embeddings
func (cpr *CPR) Init(dim int) {
	cpr.dim = dim

	// Determine max user ID (shared across domains)
	cpr.maxUserID = cpr.targetGraph.MaxVid
	if cpr.sourceGraph.MaxVid > cpr.maxUserID {
		cpr.maxUserID = cpr.sourceGraph.MaxVid
	}

	fmt.Println("Model Setting:")
	fmt.Printf("\tdimension:\t\t%d\n", dim)
	fmt.Printf("\tmax_user_id:\t\t%d\n", cpr.maxUserID)
	fmt.Printf("\ttarget_items:\t\t%d\n", cpr.targetGraph.MaxVid)
	fmt.Printf("\tsource_items:\t\t%d\n", cpr.sourceGraph.MaxVid)

	// Initialize user embeddings
	cpr.userEmbed = make([][]float64, cpr.maxUserID)
	for uid := int64(0); uid < cpr.maxUserID; uid++ {
		cpr.userEmbed[uid] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			cpr.userEmbed[uid][d] = (rand.Float64() - 0.5) / float64(dim)
		}
	}

	// Initialize target domain item embeddings
	cpr.targetEmbed = make([][]float64, cpr.targetGraph.MaxVid)
	for iid := int64(0); iid < cpr.targetGraph.MaxVid; iid++ {
		cpr.targetEmbed[iid] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			cpr.targetEmbed[iid][d] = (rand.Float64() - 0.5) / float64(dim)
		}
	}

	// Initialize source domain item embeddings
	cpr.sourceEmbed = make([][]float64, cpr.sourceGraph.MaxVid)
	for iid := int64(0); iid < cpr.sourceGraph.MaxVid; iid++ {
		cpr.sourceEmbed[iid] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			cpr.sourceEmbed[iid][d] = (rand.Float64() - 0.5) / float64(dim)
		}
	}
}

// transformUser computes transformed user embedding by aggregating
// information from both target and source domains
func (cpr *CPR) transformUser(userID int64, transformed []float64) {
	// Initialize to zero
	for d := 0; d < cpr.dim; d++ {
		transformed[d] = 0.0
	}

	count := 0.0

	// Add base user embedding
	for d := 0; d < cpr.dim; d++ {
		transformed[d] += cpr.userEmbed[userID][d]
	}
	count += 1.0

	// Aggregate from target domain items (if any)
	if targets, ok := cpr.userToTarget[userID]; ok {
		for _, itemID := range targets {
			if itemID < int64(len(cpr.targetEmbed)) {
				for d := 0; d < cpr.dim; d++ {
					transformed[d] += cpr.targetEmbed[itemID][d]
				}
				count += 1.0
			}
		}
	}

	// Aggregate from source domain items (if any)
	if sources, ok := cpr.userToSource[userID]; ok {
		for _, itemID := range sources {
			if itemID < int64(len(cpr.sourceEmbed)) {
				for d := 0; d < cpr.dim; d++ {
					transformed[d] += cpr.sourceEmbed[itemID][d]
				}
				count += 1.0
			}
		}
	}

	// Average the aggregated embeddings
	if count > 0 {
		for d := 0; d < cpr.dim; d++ {
			transformed[d] /= count
		}
	}
}

// Train trains the CPR model using margin-based BPR loss
func (cpr *CPR) Train(updateTimes, negativeSamples int, alpha, userReg, itemReg, margin float64, workers int) {
	fmt.Println("Model:")
	fmt.Println("\t[CPR: Cross-Domain Preference Ranking]")

	fmt.Println("Learning Parameters:")
	fmt.Printf("\tupdate_times:\t\t%d\n", updateTimes)
	fmt.Printf("\tnegative_samples:\t%d\n", negativeSamples)
	fmt.Printf("\talpha:\t\t\t%.6f\n", alpha)
	fmt.Printf("\tuser_reg:\t\t%.6f\n", userReg)
	fmt.Printf("\titem_reg:\t\t%.6f\n", itemReg)
	fmt.Printf("\tmargin:\t\t\t%.6f\n", margin)
	fmt.Printf("\tworkers:\t\t%d\n", workers)

	fmt.Println("Start Training:")

	total := int64(updateTimes) * 1000000
	alphaMin := alpha * 0.0001
	currentAlpha := alpha
	count := int64(0)

	var countMu sync.Mutex
	var wg sync.WaitGroup
	chunkSize := total / int64(workers)

	for w := 0; w < workers; w++ {
		wg.Add(1)
		start := int64(w) * chunkSize
		end := start + chunkSize
		if w == workers-1 {
			end = total
		}

		go func(start, end int64) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(time.Now().UnixNano() + start))

			// Buffers for user transformation
			userVec := make([]float64, cpr.dim)
			userGrad := make([]float64, cpr.dim)
			posGrad := make([]float64, cpr.dim)
			negGrad := make([]float64, cpr.dim)

			for i := start; i < end; i++ {
				// Sample a user and positive item from target domain
				userID := cpr.targetGraph.SourceSample(rng)
				posItem := cpr.targetGraph.TargetSample(userID, rng)
				if posItem == -1 {
					continue
				}

				// Transform user embedding
				cpr.transformUser(userID, userVec)

				// Sample negative item
				negItem := cpr.targetGraph.NegativeSample(rng)

				// Compute scores
				posScore := 0.0
				negScore := 0.0
				for d := 0; d < cpr.dim; d++ {
					posScore += userVec[d] * cpr.targetEmbed[posItem][d]
					negScore += userVec[d] * cpr.targetEmbed[negItem][d]
				}

				// Margin-based BPR loss: max(0, margin - (pos_score - neg_score))
				diff := posScore - negScore
				if diff < margin {
					// Compute gradient: sigmoid(-diff) for standard BPR
					// For margin-based: gradient when diff < margin
					g := currentAlpha * cpr.targetGraph.FastSigmoid(-(diff - margin))

					// Reset gradients
					for d := 0; d < cpr.dim; d++ {
						userGrad[d] = 0.0
						posGrad[d] = 0.0
						negGrad[d] = 0.0
					}

					// Compute gradients
					for d := 0; d < cpr.dim; d++ {
						posGrad[d] = g * userVec[d]
						negGrad[d] = -g * userVec[d]
						userGrad[d] = g * (cpr.targetEmbed[posItem][d] - cpr.targetEmbed[negItem][d])
					}

					// Apply L2 regularization and update embeddings
					for d := 0; d < cpr.dim; d++ {
						// Update user embedding
						cpr.userEmbed[userID][d] -= currentAlpha * userReg * cpr.userEmbed[userID][d]
						cpr.userEmbed[userID][d] += userGrad[d]

						// Update positive item
						cpr.targetEmbed[posItem][d] -= currentAlpha * itemReg * cpr.targetEmbed[posItem][d]
						cpr.targetEmbed[posItem][d] += posGrad[d]

						// Update negative item
						cpr.targetEmbed[negItem][d] -= currentAlpha * itemReg * cpr.targetEmbed[negItem][d]
						cpr.targetEmbed[negItem][d] += negGrad[d]
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
		}(start, end)
	}

	wg.Wait()
	fmt.Printf("\tAlpha: %.6f\tProgress: 100.00 %%\n", currentAlpha)
}

// SaveWeights saves the learned embeddings to files
func (cpr *CPR) SaveWeights(userFile, targetFile, sourceFile string) error {
	fmt.Println("Save Model:")

	// Save user embeddings
	if userFile != "" {
		file, err := os.Create(userFile)
		if err != nil {
			return fmt.Errorf("failed to create user file: %v", err)
		}
		defer file.Close()

		fmt.Fprintf(file, "%d %d\n", cpr.maxUserID, cpr.dim)
		for uid := int64(0); uid < cpr.maxUserID; uid++ {
			name := cpr.targetGraph.GetVertexName(uid)
			fmt.Fprintf(file, "%s", name)
			for d := 0; d < cpr.dim; d++ {
				fmt.Fprintf(file, " %.6f", cpr.userEmbed[uid][d])
			}
			fmt.Fprintln(file)
		}
		fmt.Printf("\tSave users to <%s>\n", userFile)
	}

	// Save target domain item embeddings
	if targetFile != "" {
		file, err := os.Create(targetFile)
		if err != nil {
			return fmt.Errorf("failed to create target file: %v", err)
		}
		defer file.Close()

		fmt.Fprintf(file, "%d %d\n", cpr.targetGraph.MaxVid, cpr.dim)
		for iid := int64(0); iid < cpr.targetGraph.MaxVid; iid++ {
			name := cpr.targetGraph.GetVertexName(iid)
			fmt.Fprintf(file, "%s", name)
			for d := 0; d < cpr.dim; d++ {
				fmt.Fprintf(file, " %.6f", cpr.targetEmbed[iid][d])
			}
			fmt.Fprintln(file)
		}
		fmt.Printf("\tSave target items to <%s>\n", targetFile)
	}

	// Save source domain item embeddings
	if sourceFile != "" {
		file, err := os.Create(sourceFile)
		if err != nil {
			return fmt.Errorf("failed to create source file: %v", err)
		}
		defer file.Close()

		fmt.Fprintf(file, "%d %d\n", cpr.sourceGraph.MaxVid, cpr.dim)
		for iid := int64(0); iid < cpr.sourceGraph.MaxVid; iid++ {
			name := cpr.sourceGraph.GetVertexName(iid)
			fmt.Fprintf(file, "%s", name)
			for d := 0; d < cpr.dim; d++ {
				fmt.Fprintf(file, " %.6f", cpr.sourceEmbed[iid][d])
			}
			fmt.Fprintln(file)
		}
		fmt.Printf("\tSave source items to <%s>\n", sourceFile)
	}

	return nil
}
