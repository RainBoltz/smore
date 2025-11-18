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
