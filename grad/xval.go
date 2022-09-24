package grad

import (
	"fmt"
	"math"
	"strconv"
	"sync"
)

type floatingRange struct {
	start   float64
	end     float64
	step    float64
	current float64
}

func NewFloatingRange(start float64, end float64, step float64) *floatingRange {
	return &floatingRange{
		start:   start,
		end:     end,
		step:    step,
		current: start,
	}
}

func (ft *floatingRange) next() float64 {
	// this statement allows to return the starting value first
	v := ft.current

	// increment if ok, otherwise return the control value to signal exhaustion
	if ft.current <= ft.end {
		ft.current += ft.step
	} else {
		v = math.MaxFloat64
	}

	return v
}

type XVal struct {
	model      *Model
	modelArch  []int
	k          int
	alpha      func(int, int) float64
	lossFn     func(*Value, float64) *Value
	values     [][][2]float64
	labels     [][]float64
	hyperRange *floatingRange
	cvScores   map[string][]float64
}

func NewXVal(
	data [][2]float64,
	labels []float64,
	arch []int,
	fr *floatingRange,
	alpha func(int, int) float64,
	lossFunc func(*Value, float64) *Value,
	k int,
) *XVal {
	groupedValues, groupedLabels := group(data, labels, k)

	xv := XVal{
		model:      nil, // initialized later (when testing against hyperparameters)
		modelArch:  arch,
		k:          k,
		alpha:      alpha,
		lossFn:     lossFunc,
		values:     groupedValues,
		labels:     groupedLabels,
		hyperRange: fr,
		cvScores:   make(map[string][]float64),
	}

	return &xv
}

func (xv *XVal) SearchBestHyperpar() *Value {
	var hyperpars []float64

	fmt.Printf(
		"==> Using Cross Validation to look for the best L2 lambda hyperparameter in values ranging from %.4f to %.4f\n",
		xv.hyperRange.start,
		xv.hyperRange.end,
	)

	// check against a control value to know if there are more steps
	h := xv.hyperRange.next()
	for h != math.MaxFloat64 {
		var scores []float64

		// here golang's concurrency is used to speed up cross validation (with goroutines and channels)
		var xvalWg sync.WaitGroup            // goroutines wait group
		scoresCh := make(chan float64, xv.k) // holdouts scores buffered channel

		// loop for k times changing the holdout each time
		for ki := 0; ki < xv.k; ki++ {
			// increment workers counter
			xvalWg.Add(1)

			// make a copy of the data and relative labels to pass to each goroutine
			valuesCopy := make([][][2]float64, len(xv.values))
			copy(valuesCopy, xv.values)
			labelsCopy := make([][]float64, len(xv.labels))
			copy(labelsCopy, xv.labels)

			// goroutine to compute cross validation on this iteration group (out of xv.k for each hypervalue in range)
			go func(data [][][2]float64, labels [][]float64, idx int, lambda float64, ch chan<- float64) {
				defer xvalWg.Done()

				// prepping
				holdoutValues, trainingValues := popValue(data, idx)
				holdoutLabels, trainingLabels := popLabel(labels, idx)

				// small training session (each time with a different model)
				xv.miniTrain(trainingValues, trainingLabels, NewValue(lambda))
				// holdout testing on previous training session to compute accuracy metric w.r.t. current hyperpar
				xv.holdout(holdoutValues, holdoutLabels, ch)
			}(valuesCopy, labelsCopy, ki, h, scoresCh)
		}

		// wait until all workers are done then close channel
		xvalWg.Wait()
		close(scoresCh)

		// read scores from each of the xv.k cross validation workers of this hyperparameter
		for acc := range scoresCh {
			scores = append(scores, acc)
		}

		// average the xv.k holdout scores, creating a mean for this hyperparameter
		avgScore := avgValue(scores)
		fmt.Printf("hyperpar=%.4f, accuracy=%.0f%%\n", h, avgScore*100)

		// add score to map if not present
		if _, ok := xv.cvScores[fmt.Sprintf("%.4f", avgScore)]; !ok {
			// create slice beacuse not present
			xv.cvScores[fmt.Sprintf("%.4f", avgScore)] = []float64{h}
		} else {
			// append element to existing slice
			xv.cvScores[fmt.Sprintf("%.4f", avgScore)] = append(xv.cvScores[fmt.Sprintf("%.4f", avgScore)], h)
		}

		// get hyperparameter candidate value for next iteration
		h = xv.hyperRange.next()
	}

	// get hyperpar associated with highest accuracy rate
	maxKey := math.Inf(-1)
	for k, v := range xv.cvScores {
		if scoreK, err := strconv.ParseFloat(k, 64); err == nil {
			if scoreK > maxKey {
				maxKey = scoreK
				hyperpars = v
			}
		}
	}

	// get first element of the slice as they all have the same score
	return NewValue(hyperpars[0])
}

func (xv *XVal) miniTrain(inputs [][][2]float64, expectations [][]float64, hyperpar *Value) {
	// each time a mini train occurs a new model is created
	xv.model = NewModel(xv.modelArch[0], xv.modelArch[1:])

	// check inputs are the same length of expectations
	if len(inputs) != len(expectations) {
		panic("Something bad occurred during a mini training session of cross validation. Inputs and Labels are not the same length")
	}

	// train the model on each input value for 10 times each
	for idx, inp := range inputs {
		// for each slice of input in inputs train the model 10 times
		for pass := 0; pass < 10; pass++ {
			// prepping
			xv.model.ZeroGrad()

			// prediction and loss
			preds := map2Pred(inp, xv.model.FeedForward)
			losses := map2Losses(preds, expectations[idx], MSE)
			loss := NewValue(0.0)
			for _, el := range losses {
				loss.Add(el)
			}
			loss.Div(len(losses))

			// regularize loss with L2
			reg := L2(xv.model.Params(), hyperpar)
			totLoss := loss.Add(reg)

			// backward pass
			totLoss.BackwardPass()
			for _, el := range xv.model.Params() {
				// TODO: change this learning rate with the dynamic one
				// el.data -= 0.0005 * el.grad
				el.Update(0.0005)
			}
		}
	}
}

func (xv *XVal) holdout(inputs [][2]float64, expectations []float64, scoresCh chan<- float64) {
	preds := map2Pred(inputs, xv.model.FeedForward)

	// compute accuracy (i.e. the value to be returned)
	directionsSum := 0.0
	for idx := range preds {
		if (preds[idx].GetData() > 0.0) == (expectations[idx] > 0.0) {
			directionsSum += 1.0
		}
	}
	acc := directionsSum / float64(len(preds))

	scoresCh <- acc
}
