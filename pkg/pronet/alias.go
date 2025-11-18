package pronet

import (
	"math"
	"math/rand"
)

// buildAliasMethod builds alias table for O(1) weighted sampling
// This implements the alias method (Walker's alias method) for efficient sampling
func buildAliasMethod(distribution []float64, power float64) []AliasTable {
	n := len(distribution)
	if n == 0 {
		return nil
	}

	aliasTable := make([]AliasTable, n)

	// Apply power transformation and normalize
	sum := 0.0
	norm := make([]float64, n)
	for i := 0; i < n; i++ {
		if distribution[i] > 0 {
			norm[i] = math.Pow(distribution[i], power)
		} else {
			norm[i] = 0
		}
		sum += norm[i]
	}

	if sum == 0 {
		// Uniform distribution if all weights are zero
		for i := 0; i < n; i++ {
			aliasTable[i].Prob = 1.0
			aliasTable[i].Alias = int64(i)
		}
		return aliasTable
	}

	// Normalize to probability distribution
	for i := 0; i < n; i++ {
		norm[i] = norm[i] * float64(n) / sum
	}

	// Build alias table using the Vose's alias method
	small := make([]int, 0, n)
	large := make([]int, 0, n)

	for i := 0; i < n; i++ {
		if norm[i] < 1.0 {
			small = append(small, i)
		} else {
			large = append(large, i)
		}
	}

	for len(small) > 0 && len(large) > 0 {
		l := small[len(small)-1]
		small = small[:len(small)-1]

		g := large[len(large)-1]
		large = large[:len(large)-1]

		aliasTable[l].Prob = norm[l]
		aliasTable[l].Alias = int64(g)

		norm[g] = norm[g] + norm[l] - 1.0
		if norm[g] < 1.0 {
			small = append(small, g)
		} else {
			large = append(large, g)
		}
	}

	// Handle remaining elements
	for len(large) > 0 {
		g := large[len(large)-1]
		large = large[:len(large)-1]
		aliasTable[g].Prob = 1.0
		aliasTable[g].Alias = int64(g)
	}

	for len(small) > 0 {
		l := small[len(small)-1]
		small = small[:len(small)-1]
		aliasTable[l].Prob = 1.0
		aliasTable[l].Alias = int64(l)
	}

	return aliasTable
}

// aliasSample performs O(1) weighted random sampling using alias table
func aliasSample(aliasTable []AliasTable, rng *rand.Rand) int64 {
	if len(aliasTable) == 0 {
		return -1
	}

	n := len(aliasTable)
	i := rng.Intn(n)
	r := rng.Float64()

	if r < aliasTable[i].Prob {
		return int64(i)
	}
	return aliasTable[i].Alias
}
