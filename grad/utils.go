package grad

import (
	"math/rand"
)

// generates a slice of random floats comprised between min and max
// e.g if min=-1 and max=1 the formula below will generate floats
// 	   from -1 + rand.Float64() * 2 ==> -1+0*2 and -1+1*2
// rand.Float64() generates numbers between [0, 1)
func randFloats(min, max float64, n int) []*Value {
	res := make([]*Value, n)

	for i := range res {
		res[i] = NewValue(min + rand.Float64()*(max-min))
	}

	return res
}

func MSE(predicted *Value, expected float64) *Value {
	exp := NewValue(expected)

	return predicted.Sub(exp).Pow(2)
}

func group(data [][2]float64, labels []float64, k int) ([][][2]float64, [][]float64) {
	var dataGroups [][][2]float64
	var labelGroups [][]float64

	start := 0
	size := len(data) / k
	for i := 0; i < k; i++ {
		dataGroups = append(dataGroups, data[start:start+size])
		labelGroups = append(labelGroups, labels[start:start+size])
		start += size
	}

	return dataGroups, labelGroups
}

// dynamic learning rate
func Alpha(pass int, iterations int) float64 {
	return 1.0 - 0.9*float64(pass)/float64(iterations)
}

// implementation of pop for data values
// return selected element and new slice (without the selected element)
func popValue(slice [][][2]float64, i int) ([][2]float64, [][][2]float64) {
	v := slice[i]
	rest := append(slice[:i], slice[i+1:]...)

	return v, rest
}

// implementation of pop for label values
// return selected element and new slice (without the selected element)
func popLabel(slice [][]float64, i int) ([]float64, [][]float64) {
	return slice[i], append(slice[:i], slice[i+1:]...)
}

func avgValue(slice []float64) float64 {
	result := 0.0
	for _, el := range slice {
		result += el
	}

	return result / float64(len(slice))
}

func L2(input []*Value, lambda *Value) *Value {
	squared := make([]*Value, len(input))

	// square each element
	for idx, ps := range input {
		squared[idx] = ps.Pow(2)
	}

	// sum the squares
	sum := NewValue(0.0)
	for _, sq := range squared {
		sum = sum.Add(sq)
	}

	return lambda.Mul(sum)
}

func map2Pred(input [][2]float64, f func([2]float64) []*Value) []*Value {
	mapped := make([]*Value, len(input))

	for idx, inp := range input {
		y := f(inp)
		if len(y) == 1 {
			mapped[idx] = y[0]
		} else {
			panic("Something went wrong with gograd.map2Pred")
		}
	}

	return mapped
}

func map2Losses(input []*Value, expected []float64, f func(*Value, float64) *Value) []*Value {
	mapped := make([]*Value, len(input))

	for idx, inp := range input {
		mapped[idx] = f(inp, expected[idx])
	}

	return mapped
}
