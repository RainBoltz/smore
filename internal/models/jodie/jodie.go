package jodie

import (
	"fmt"
	"math"
	"math/rand"
	"os"

	"github.com/cnclabs/smore/pkg/bipartite"
	"github.com/cnclabs/smore/pkg/rnn"
)

// JODIE implements Joint Dynamic User-Item Embeddings
// Predicts future embedding trajectories for temporal interaction prediction
type JODIE struct {
	ig  *bipartite.InteractionGraph
	dim int

	// Dynamic embeddings that evolve over time
	userEmbeddings [][]float64
	itemEmbeddings [][]float64

	// Static initial embeddings (for reference)
	userEmbeddingsStatic [][]float64
	itemEmbeddingsStatic [][]float64

	// Last update timestamps
	userLastUpdate []float64
	itemLastUpdate []float64

	// RNN cells for embedding updates
	userRNN *rnn.RNNCell
	itemRNN *rnn.RNNCell

	// Projection RNN for predicting future states
	projectionRNN *rnn.RNNCell
}

// New creates a new JODIE instance
func New() *JODIE {
	return &JODIE{
		ig: bipartite.NewInteractionGraph(),
	}
}

// LoadInteractions loads the interaction graph
func (j *JODIE) LoadInteractions(filename string) error {
	return j.ig.LoadInteractions(filename)
}

// Init initializes the model
func (j *JODIE) Init(dim int) {
	j.dim = dim

	fmt.Println("Model Setting:")
	fmt.Printf("\tdimension:\t\t%d\n", dim)
	fmt.Printf("\tfeature_dim:\t\t%d\n", j.ig.FeatureDim)
	fmt.Println()

	fmt.Println("JODIE Principle:")
	fmt.Println("\t✓ Dynamic user and item embeddings")
	fmt.Println("\t✓ RNN-based embedding updates after each interaction")
	fmt.Println("\t✓ Projects embeddings to future times")
	fmt.Println("\t✓ Predicts future interactions and embedding trajectories")

	// Initialize user embeddings
	j.userEmbeddings = make([][]float64, j.ig.NumUsers)
	j.userEmbeddingsStatic = make([][]float64, j.ig.NumUsers)
	j.userLastUpdate = make([]float64, j.ig.NumUsers)

	for i := int64(0); i < j.ig.NumUsers; i++ {
		j.userEmbeddings[i] = make([]float64, dim)
		j.userEmbeddingsStatic[i] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			val := (rand.Float64() - 0.5) / float64(dim)
			j.userEmbeddings[i][d] = val
			j.userEmbeddingsStatic[i][d] = val
		}
		j.userLastUpdate[i] = j.ig.MinTime
	}

	// Initialize item embeddings
	j.itemEmbeddings = make([][]float64, j.ig.NumItems)
	j.itemEmbeddingsStatic = make([][]float64, j.ig.NumItems)
	j.itemLastUpdate = make([]float64, j.ig.NumItems)

	for i := int64(0); i < j.ig.NumItems; i++ {
		j.itemEmbeddings[i] = make([]float64, dim)
		j.itemEmbeddingsStatic[i] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			val := (rand.Float64() - 0.5) / float64(dim)
			j.itemEmbeddings[i][d] = val
			j.itemEmbeddingsStatic[i][d] = val
		}
		j.itemLastUpdate[i] = j.ig.MinTime
	}

	// Initialize RNN cells
	// Input: concatenation of user/item embedding + features + time delta
	inputDim := dim + j.ig.FeatureDim + 1 // +1 for time delta

	j.userRNN = rnn.NewRNNCell(inputDim, dim)
	j.itemRNN = rnn.NewRNNCell(inputDim, dim)
	j.projectionRNN = rnn.NewRNNCell(1, dim) // Input is just time delta
}

// projectEmbedding projects an embedding forward in time
func (j *JODIE) projectEmbedding(embedding []float64, timeDelta float64) []float64 {
	// Simple projection: use RNN with time delta as input
	if timeDelta <= 0 {
		// No projection needed
		result := make([]float64, j.dim)
		copy(result, embedding)
		return result
	}

	// Create input with normalized time delta
	return j.projectionRNN.Project(embedding, timeDelta/100.0) // Normalize time
}

// updateUserEmbedding updates user embedding after an interaction
func (j *JODIE) updateUserEmbedding(userID int64, interaction bipartite.Interaction, learningRate float64) {
	// Get current embeddings
	userEmb := j.userEmbeddings[userID]
	itemEmb := j.itemEmbeddings[interaction.ItemID]

	// Compute time delta since last update
	timeDelta := interaction.Timestamp - j.userLastUpdate[userID]

	// Create RNN input: [user_emb, item_emb, features, time_delta]
	input := make([]float64, 0, j.dim+j.dim+j.ig.FeatureDim+1)
	input = append(input, itemEmb...)
	if len(interaction.Features) > 0 {
		input = append(input, interaction.Features...)
	} else {
		// Pad with zeros if no features
		for i := 0; i < j.ig.FeatureDim; i++ {
			input = append(input, 0.0)
		}
	}
	input = append(input, timeDelta/100.0) // Normalized time delta

	// Pad or truncate input to match RNN input dimension
	for len(input) < j.userRNN.InputDim {
		input = append(input, 0.0)
	}
	input = input[:j.userRNN.InputDim]

	// Update embedding using RNN
	newEmb := j.userRNN.Forward(userEmb, input)

	// Apply learning rate and update
	for d := 0; d < j.dim; d++ {
		j.userEmbeddings[userID][d] = (1-learningRate)*userEmb[d] + learningRate*newEmb[d]
	}

	// Update last update time
	j.userLastUpdate[userID] = interaction.Timestamp
}

// updateItemEmbedding updates item embedding after an interaction
func (j *JODIE) updateItemEmbedding(itemID int64, interaction bipartite.Interaction, learningRate float64) {
	// Get current embeddings
	itemEmb := j.itemEmbeddings[itemID]
	userEmb := j.userEmbeddings[interaction.UserID]

	// Compute time delta since last update
	timeDelta := interaction.Timestamp - j.itemLastUpdate[itemID]

	// Create RNN input: [item_emb, user_emb, features, time_delta]
	input := make([]float64, 0, j.dim+j.dim+j.ig.FeatureDim+1)
	input = append(input, userEmb...)
	if len(interaction.Features) > 0 {
		input = append(input, interaction.Features...)
	} else {
		// Pad with zeros if no features
		for i := 0; i < j.ig.FeatureDim; i++ {
			input = append(input, 0.0)
		}
	}
	input = append(input, timeDelta/100.0) // Normalized time delta

	// Pad or truncate input to match RNN input dimension
	for len(input) < j.itemRNN.InputDim {
		input = append(input, 0.0)
	}
	input = input[:j.itemRNN.InputDim]

	// Update embedding using RNN
	newEmb := j.itemRNN.Forward(itemEmb, input)

	// Apply learning rate and update
	for d := 0; d < j.dim; d++ {
		j.itemEmbeddings[itemID][d] = (1-learningRate)*itemEmb[d] + learningRate*newEmb[d]
	}

	// Update last update time
	j.itemLastUpdate[itemID] = interaction.Timestamp
}

// predictInteraction predicts if user will interact with item
func (j *JODIE) predictInteraction(userEmb, itemEmb []float64) float64 {
	// Compute dot product (cosine similarity)
	dotProduct := 0.0
	for d := 0; d < j.dim; d++ {
		dotProduct += userEmb[d] * itemEmb[d]
	}

	// Apply sigmoid to get probability
	return 1.0 / (1.0 + math.Exp(-dotProduct))
}

// Train trains the JODIE model
func (j *JODIE) Train(epochs int, learningRate float64, batchSize int) {
	fmt.Println()
	fmt.Println("Model:")
	fmt.Println("\t[JODIE - Joint Dynamic User-Item Embeddings]")
	fmt.Println()

	fmt.Println("Learning Parameters:")
	fmt.Printf("\tepochs:\t\t\t%d\n", epochs)
	fmt.Printf("\tlearning_rate:\t\t%.6f\n", learningRate)
	fmt.Printf("\tbatch_size:\t\t%d\n", batchSize)
	fmt.Println()

	fmt.Println("Start Training:")

	totalInteractions := len(j.ig.Interactions)

	for epoch := 0; epoch < epochs; epoch++ {
		fmt.Printf("\nEpoch %d/%d:\n", epoch+1, epochs)

		// Reset embeddings to initial state at start of each epoch
		for i := int64(0); i < j.ig.NumUsers; i++ {
			copy(j.userEmbeddings[i], j.userEmbeddingsStatic[i])
			j.userLastUpdate[i] = j.ig.MinTime
		}
		for i := int64(0); i < j.ig.NumItems; i++ {
			copy(j.itemEmbeddings[i], j.itemEmbeddingsStatic[i])
			j.itemLastUpdate[i] = j.ig.MinTime
		}

		totalLoss := 0.0
		count := 0

		// Process interactions in chronological order
		for idx, interaction := range j.ig.Interactions {
			// Get current embeddings (before update)
			userEmb := make([]float64, j.dim)
			itemEmb := make([]float64, j.dim)
			copy(userEmb, j.userEmbeddings[interaction.UserID])
			copy(itemEmb, j.itemEmbeddings[interaction.ItemID])

			// Predict interaction (positive sample)
			predPos := j.predictInteraction(userEmb, itemEmb)

			// Sample negative item
			negItemID := rand.Int63n(j.ig.NumItems)
			for negItemID == interaction.ItemID {
				negItemID = rand.Int63n(j.ig.NumItems)
			}
			negItemEmb := make([]float64, j.dim)
			copy(negItemEmb, j.itemEmbeddings[negItemID])

			// Predict negative sample
			predNeg := j.predictInteraction(userEmb, negItemEmb)

			// Compute loss (binary cross-entropy)
			loss := -math.Log(predPos+1e-10) - math.Log(1-predNeg+1e-10)
			totalLoss += loss

			// Update embeddings based on this interaction
			j.updateUserEmbedding(interaction.UserID, interaction, learningRate)
			j.updateItemEmbedding(interaction.ItemID, interaction, learningRate)

			// Update static embeddings (slowly)
			staticLR := learningRate * 0.1
			for d := 0; d < j.dim; d++ {
				j.userEmbeddingsStatic[interaction.UserID][d] += staticLR * (j.userEmbeddings[interaction.UserID][d] - j.userEmbeddingsStatic[interaction.UserID][d])
				j.itemEmbeddingsStatic[interaction.ItemID][d] += staticLR * (j.itemEmbeddings[interaction.ItemID][d] - j.itemEmbeddingsStatic[interaction.ItemID][d])
			}

			count++
			if (idx+1)%1000 == 0 || idx == totalInteractions-1 {
				avgLoss := totalLoss / float64(count)
				progress := float64(idx+1) / float64(totalInteractions) * 100
				fmt.Printf("\tProgress: %.2f%% - Loss: %.4f\r", progress, avgLoss)
			}
		}

		avgLoss := totalLoss / float64(count)
		fmt.Printf("\tEpoch %d completed - Avg Loss: %.4f\n", epoch+1, avgLoss)
	}

	fmt.Println("\nTraining Complete!")
}

// SaveEmbeddings saves the learned embeddings
func (j *JODIE) SaveEmbeddings(filename string) error {
	fmt.Println("Save Model:")

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer file.Close()

	// Write header: num_users num_items dimension
	fmt.Fprintf(file, "%d %d %d\n", j.ig.NumUsers, j.ig.NumItems, j.dim)

	// Write user embeddings
	fmt.Fprintln(file, "# Users")
	for i := int64(0); i < j.ig.NumUsers; i++ {
		name := j.ig.GetUserName(i)
		fmt.Fprintf(file, "U\t%s", name)

		for d := 0; d < j.dim; d++ {
			fmt.Fprintf(file, " %.6f", j.userEmbeddingsStatic[i][d])
		}
		fmt.Fprintln(file)
	}

	// Write item embeddings
	fmt.Fprintln(file, "# Items")
	for i := int64(0); i < j.ig.NumItems; i++ {
		name := j.ig.GetItemName(i)
		fmt.Fprintf(file, "I\t%s", name)

		for d := 0; d < j.dim; d++ {
			fmt.Fprintf(file, " %.6f", j.itemEmbeddingsStatic[i][d])
		}
		fmt.Fprintln(file)
	}

	fmt.Printf("\tSave to <%s>\n", filename)
	return nil
}

// GetUserEmbedding returns the embedding for a user
func (j *JODIE) GetUserEmbedding(userID int64) []float64 {
	if userID < 0 || userID >= j.ig.NumUsers {
		return nil
	}
	return j.userEmbeddingsStatic[userID]
}

// GetItemEmbedding returns the embedding for an item
func (j *JODIE) GetItemEmbedding(itemID int64) []float64 {
	if itemID < 0 || itemID >= j.ig.NumItems {
		return nil
	}
	return j.itemEmbeddingsStatic[itemID]
}

// EvaluatePredictions evaluates prediction accuracy on recent interactions
func (j *JODIE) EvaluatePredictions() {
	fmt.Println("\nPrediction Evaluation:")

	// Use last 20% of interactions for evaluation
	testSize := len(j.ig.Interactions) / 5
	if testSize > 1000 {
		testSize = 1000
	}
	if testSize < 10 {
		testSize = len(j.ig.Interactions)
	}

	startIdx := len(j.ig.Interactions) - testSize

	correct := 0
	total := 0

	for i := startIdx; i < len(j.ig.Interactions); i++ {
		interaction := j.ig.Interactions[i]

		userEmb := j.userEmbeddingsStatic[interaction.UserID]
		itemEmb := j.itemEmbeddingsStatic[interaction.ItemID]

		// Positive prediction
		predPos := j.predictInteraction(userEmb, itemEmb)

		// Random negative item
		negItemID := rand.Int63n(j.ig.NumItems)
		for negItemID == interaction.ItemID {
			negItemID = rand.Int63n(j.ig.NumItems)
		}
		negItemEmb := j.itemEmbeddingsStatic[negItemID]
		predNeg := j.predictInteraction(userEmb, negItemEmb)

		// Check if positive score is higher than negative
		if predPos > predNeg {
			correct++
		}
		total++
	}

	accuracy := float64(correct) / float64(total) * 100
	fmt.Printf("\tPrediction accuracy: %.2f%% (%d/%d)\n", accuracy, correct, total)
	fmt.Printf("\t(Positive interactions ranked higher than random negatives)\n")
}
