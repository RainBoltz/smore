package rnn

import (
	"math"
	"math/rand"
)

// RNNCell represents a simple RNN cell for temporal embedding updates
type RNNCell struct {
	InputDim  int
	HiddenDim int

	// Weight matrices
	Wh [][]float64 // hidden-to-hidden weights [hiddenDim x hiddenDim]
	Wx [][]float64 // input-to-hidden weights [hiddenDim x inputDim]
	B  []float64   // bias [hiddenDim]
}

// NewRNNCell creates a new RNN cell
func NewRNNCell(inputDim, hiddenDim int) *RNNCell {
	rnn := &RNNCell{
		InputDim:  inputDim,
		HiddenDim: hiddenDim,
	}

	// Initialize weight matrices with small random values
	scale := 1.0 / math.Sqrt(float64(hiddenDim))

	// Wh: hidden-to-hidden
	rnn.Wh = make([][]float64, hiddenDim)
	for i := 0; i < hiddenDim; i++ {
		rnn.Wh[i] = make([]float64, hiddenDim)
		for j := 0; j < hiddenDim; j++ {
			rnn.Wh[i][j] = (rand.Float64()*2 - 1) * scale
		}
	}

	// Wx: input-to-hidden
	rnn.Wx = make([][]float64, hiddenDim)
	for i := 0; i < hiddenDim; i++ {
		rnn.Wx[i] = make([]float64, inputDim)
		for j := 0; j < inputDim; j++ {
			rnn.Wx[i][j] = (rand.Float64()*2 - 1) * scale
		}
	}

	// Bias
	rnn.B = make([]float64, hiddenDim)
	for i := 0; i < hiddenDim; i++ {
		rnn.B[i] = 0.0
	}

	return rnn
}

// Forward performs a forward pass through the RNN cell
// h_new = tanh(Wh * h_old + Wx * x + b)
func (rnn *RNNCell) Forward(hiddenState, input []float64) []float64 {
	newHidden := make([]float64, rnn.HiddenDim)

	for i := 0; i < rnn.HiddenDim; i++ {
		// Wh * h_old
		sum := 0.0
		for j := 0; j < rnn.HiddenDim; j++ {
			sum += rnn.Wh[i][j] * hiddenState[j]
		}

		// Wx * x
		for j := 0; j < rnn.InputDim; j++ {
			sum += rnn.Wx[i][j] * input[j]
		}

		// Add bias
		sum += rnn.B[i]

		// Apply tanh activation
		newHidden[i] = math.Tanh(sum)
	}

	return newHidden
}

// Update updates RNN parameters with gradient descent
// Simplified update: only update based on prediction error
func (rnn *RNNCell) Update(hiddenState, input, target []float64, learningRate float64) {
	// Forward pass
	predicted := rnn.Forward(hiddenState, input)

	// Compute error
	error := make([]float64, rnn.HiddenDim)
	for i := 0; i < rnn.HiddenDim; i++ {
		error[i] = target[i] - predicted[i]
	}

	// Gradient of tanh: 1 - tanh^2(x)
	gradient := make([]float64, rnn.HiddenDim)
	for i := 0; i < rnn.HiddenDim; i++ {
		gradient[i] = error[i] * (1 - predicted[i]*predicted[i])
	}

	// Update Wh (hidden-to-hidden)
	for i := 0; i < rnn.HiddenDim; i++ {
		for j := 0; j < rnn.HiddenDim; j++ {
			rnn.Wh[i][j] += learningRate * gradient[i] * hiddenState[j]
		}
	}

	// Update Wx (input-to-hidden)
	for i := 0; i < rnn.HiddenDim; i++ {
		for j := 0; j < rnn.InputDim; j++ {
			rnn.Wx[i][j] += learningRate * gradient[i] * input[j]
		}
	}

	// Update bias
	for i := 0; i < rnn.HiddenDim; i++ {
		rnn.B[i] += learningRate * gradient[i]
	}
}

// Project projects an embedding forward in time
// Used to predict future embeddings based on time difference
func (rnn *RNNCell) Project(embedding []float64, timeDelta float64) []float64 {
	// Create input from time delta
	input := make([]float64, rnn.InputDim)
	if rnn.InputDim > 0 {
		input[0] = timeDelta
	}

	// Forward pass to get projected embedding
	return rnn.Forward(embedding, input)
}

// MatrixVectorMultiply performs matrix-vector multiplication
func MatrixVectorMultiply(matrix [][]float64, vector []float64) []float64 {
	rows := len(matrix)
	if rows == 0 {
		return nil
	}
	cols := len(matrix[0])
	if cols != len(vector) {
		return nil
	}

	result := make([]float64, rows)
	for i := 0; i < rows; i++ {
		sum := 0.0
		for j := 0; j < cols; j++ {
			sum += matrix[i][j] * vector[j]
		}
		result[i] = sum
	}

	return result
}

// VectorAdd adds two vectors
func VectorAdd(a, b []float64) []float64 {
	if len(a) != len(b) {
		return nil
	}

	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] + b[i]
	}

	return result
}

// VectorScale scales a vector by a scalar
func VectorScale(vector []float64, scalar float64) []float64 {
	result := make([]float64, len(vector))
	for i, v := range vector {
		result[i] = v * scalar
	}
	return result
}

// VectorNorm computes L2 norm of a vector
func VectorNorm(vector []float64) float64 {
	sum := 0.0
	for _, v := range vector {
		sum += v * v
	}
	return math.Sqrt(sum)
}

// NormalizeVector normalizes a vector to unit length
func NormalizeVector(vector []float64) []float64 {
	norm := VectorNorm(vector)
	if norm == 0 {
		return vector
	}

	result := make([]float64, len(vector))
	for i, v := range vector {
		result[i] = v / norm
	}
	return result
}
