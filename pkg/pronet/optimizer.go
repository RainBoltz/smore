package pronet

import (
	"math/rand"
)

// UpdatePairs updates embeddings for a batch of vertex-context pairs using SGD
func (pn *ProNet) UpdatePairs(
	wVertex, wContext [][]float64,
	vertices, contexts []int64,
	dim, negativeSamples int,
	alpha float64,
	rng *rand.Rand,
) {
	for i := 0; i < len(vertices); i++ {
		pn.UpdatePair(wVertex, wContext, vertices[i], contexts[i], dim, negativeSamples, alpha, rng)
	}
}

// UpdatePair updates embeddings for a single vertex-context pair
func (pn *ProNet) UpdatePair(
	wVertex, wContext [][]float64,
	vertex, context int64,
	dim, negativeSamples int,
	alpha float64,
	rng *rand.Rand,
) {
	// Temporary gradient vectors
	vertexGrad := make([]float64, dim)
	contextGrad := make([]float64, dim)

	// Positive sample
	label := 1.0
	pn.sgdUpdate(wVertex[vertex], wContext[context], label, alpha, vertexGrad, contextGrad)

	// Negative samples
	for i := 0; i < negativeSamples; i++ {
		negSample := pn.NegativeSample(rng)
		if negSample == context {
			continue
		}

		label = 0.0
		negGrad := make([]float64, dim)
		pn.sgdUpdate(wVertex[vertex], wContext[negSample], label, alpha, vertexGrad, negGrad)

		// Update negative context
		for d := 0; d < dim; d++ {
			wContext[negSample][d] += negGrad[d]
		}
	}

	// Update vertex and context embeddings
	for d := 0; d < dim; d++ {
		wVertex[vertex][d] += vertexGrad[d]
		wContext[context][d] += contextGrad[d]
	}
}

// sgdUpdate performs SGD update for a single pair
func (pn *ProNet) sgdUpdate(
	vertexEmb, contextEmb []float64,
	label, alpha float64,
	vertexGrad, contextGrad []float64,
) {
	// Compute dot product
	score := 0.0
	dim := len(vertexEmb)
	for d := 0; d < dim; d++ {
		score += vertexEmb[d] * contextEmb[d]
	}

	// Compute prediction using fast sigmoid
	pred := pn.FastSigmoid(score)

	// Compute gradient
	grad := alpha * (label - pred)

	// Update gradients
	for d := 0; d < dim; d++ {
		vertexGrad[d] += grad * contextEmb[d]
		contextGrad[d] += grad * vertexEmb[d]
	}
}

// UpdateBPRPair updates embeddings using Bayesian Personalized Ranking
func (pn *ProNet) UpdateBPRPair(
	wVertex, wContext [][]float64,
	vertex, posContext, negContext int64,
	dim int,
	alpha, lambda float64,
	rng *rand.Rand,
) {
	// Compute scores
	posScore := 0.0
	negScore := 0.0
	for d := 0; d < dim; d++ {
		posScore += wVertex[vertex][d] * wContext[posContext][d]
		negScore += wVertex[vertex][d] * wContext[negContext][d]
	}

	// BPR gradient: sigmoid(neg - pos)
	diff := negScore - posScore
	gradCoef := alpha * pn.FastSigmoid(diff)

	// Update embeddings
	for d := 0; d < dim; d++ {
		vertexGrad := gradCoef * (wContext[posContext][d] - wContext[negContext][d])
		posContextGrad := gradCoef * wVertex[vertex][d]
		negContextGrad := -gradCoef * wVertex[vertex][d]

		// L2 regularization
		wVertex[vertex][d] += vertexGrad - lambda*alpha*wVertex[vertex][d]
		wContext[posContext][d] += posContextGrad - lambda*alpha*wContext[posContext][d]
		wContext[negContext][d] += negContextGrad - lambda*alpha*wContext[negContext][d]
	}
}

// UpdateCBOW updates embeddings using Continuous Bag of Words
func (pn *ProNet) UpdateCBOW(
	wVertex, wContext [][]float64,
	contexts []int64,
	target int64,
	dim, negativeSamples int,
	alpha float64,
	rng *rand.Rand,
) {
	if len(contexts) == 0 {
		return
	}

	// Average context vectors
	avgContext := make([]float64, dim)
	for _, ctx := range contexts {
		for d := 0; d < dim; d++ {
			avgContext[d] += wContext[ctx][d]
		}
	}
	for d := 0; d < dim; d++ {
		avgContext[d] /= float64(len(contexts))
	}

	// Temporary gradient vectors
	vertexGrad := make([]float64, dim)
	contextGrad := make([]float64, dim)

	// Positive sample
	pn.sgdUpdate(wVertex[target], avgContext, 1.0, alpha, vertexGrad, contextGrad)

	// Negative samples
	for i := 0; i < negativeSamples; i++ {
		negSample := pn.NegativeSample(rng)
		if negSample == target {
			continue
		}

		negGrad := make([]float64, dim)
		pn.sgdUpdate(wVertex[negSample], avgContext, 0.0, alpha, negGrad, contextGrad)

		// Update negative sample
		for d := 0; d < dim; d++ {
			wVertex[negSample][d] += negGrad[d]
		}
	}

	// Update target vertex
	for d := 0; d < dim; d++ {
		wVertex[target][d] += vertexGrad[d]
	}

	// Distribute context gradient to all context vectors
	for _, ctx := range contexts {
		for d := 0; d < dim; d++ {
			wContext[ctx][d] += contextGrad[d] / float64(len(contexts))
		}
	}
}

// UpdateSBPRPair updates embeddings using Skewed Bayesian Personalized Ranking
// This implements the Skew-Opt (Skewness Ranking Optimization) algorithm with parameters:
// - xi: skewness parameter
// - omega: proximity weight
// - eta: balance parameter (exponent)
func (pn *ProNet) UpdateSBPRPair(
	wVertex, wContext [][]float64,
	vertex, contextI int64,
	dim int,
	reg, xi, omega float64,
	eta int,
	alpha float64,
	rng *rand.Rand,
) {
	vertexErr := make([]float64, dim)
	contextErr := make([]float64, dim)
	contextVec := make([]float64, dim)

	updateCount := 0

	// Sample 16 negative contexts and update
	for n := 0; n < 16; n++ {
		contextJ := pn.NegativeSample(rng)

		// Reset context error and compute difference vector
		for d := 0; d < dim; d++ {
			contextErr[d] = 0.0
			contextVec[d] = wContext[contextI][d] - wContext[contextJ][d]
		}

		// Apply skewed BPR SGD
		if pn.optSBPRSGD(wVertex[vertex], contextVec, xi, omega, eta, alpha, vertexErr, contextErr) {
			// Apply L2 regularization and update
			for d := 0; d < dim; d++ {
				wContext[contextI][d] -= alpha * 0.01 * wContext[contextI][d]
				wContext[contextJ][d] -= alpha * 0.01 * wContext[contextJ][d]

				wContext[contextI][d] += contextErr[d]
				wContext[contextJ][d] -= contextErr[d]
			}
			updateCount++
		}
	}

	// Update vertex embedding with averaged gradient
	if updateCount > 0 {
		for d := 0; d < dim; d++ {
			wVertex[vertex][d] -= alpha * 0.01 * wVertex[vertex][d]
			wVertex[vertex][d] += vertexErr[d] / float64(updateCount)
		}
	}
}

// optSBPRSGD performs the core Skew-Opt (Skewed BPR) SGD optimization
func (pn *ProNet) optSBPRSGD(
	wVertex, wContext []float64,
	xi, omega float64,
	eta int,
	alpha float64,
	lossVertex, lossContext []float64,
) bool {
	dim := len(wVertex)

	// Compute dot product
	f := 0.0
	for d := 0; d < dim; d++ {
		f += wVertex[d] * wContext[d]
	}

	// Transform score with skewness and proximity parameters
	g := (f - xi) / omega
	if g > 2.0 {
		return false
	}
	if g < -2.0 {
		g = -2.0
	}

	// Apply eta power: g^eta
	gInSigmoid := 1.0
	for i := 0; i < eta; i++ {
		gInSigmoid *= g
	}
	gChainDiff := gInSigmoid / g

	// Apply sigmoid and chain rule for gradient
	g = pn.FastSigmoid(-1.0*gInSigmoid) * gChainDiff / omega
	g *= alpha

	// Compute and accumulate gradients
	for d := 0; d < dim; d++ {
		lossVertex[d] += g * wContext[d]
	}
	for d := 0; d < dim; d++ {
		lossContext[d] += g * wVertex[d]
	}

	return true
}
