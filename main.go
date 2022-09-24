package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/dsprn/gograd/grad"
)

func main() {
	// setting hyperparameters
	alpha := 0.0005
	arch := []int{16, 16, 1} // check this in the rust version

	// creating model
	m := grad.NewModel(2, arch)

	// creating cross validation struct used to get best L2 lambda hyperparameter
	xv := grad.NewXVal(
		grad.GetInputs(),
		grad.GetLabels(),
		append([]int{2}, arch...), // prepend number of inputs to network architecture
		grad.NewFloatingRange(0.0, 0.01, 0.0005),
		grad.Alpha,
		grad.MSE,
		10,
	)
	l2Lambda := xv.SearchBestHyperpar()
	fmt.Printf("==> L2 lambda value=%.4f\n", l2Lambda.GetData())

	// seed and random int generator
	s1 := rand.NewSource(time.Now().UnixNano())
	r1 := rand.New(s1)

	// choosing random data and label
	fmt.Println("\n==> Choosing inputs and relative label from a preloaded dataset...")
	dataIndex := r1.Intn(100)
	fmt.Printf("==> Getting inputs at index %d and relative label\n", dataIndex)
	inputs := grad.GetInputs()[dataIndex]
	label := grad.GetLabels()[dataIndex]
	fmt.Printf("==> Input values=%v\n", inputs)
	fmt.Printf("==> Expected value=%f\n", label)

	// main loop
	fmt.Println("\n==> Start training the model...")
	for round := 1; round < 100; round++ {
		// prepping for this round
		m.ZeroGrad()

		// forward pass
		pred := m.FeedForward(inputs)
		loss := grad.MSE(pred[0], label)

		// L2 regularization
		reg := grad.L2(m.Params(), l2Lambda)
		totLoss := loss.Add(reg)

		// backward pass
		totLoss.BackwardPass()
		for _, el := range m.Params() {
			el.Update(alpha)
		}

		fmt.Printf(
			"pass=%d, predicted=%f, expected=%.1f, loss=%f, reg=%f, tot_loss=%f\n",
			round,
			pred[0].GetData(),
			label,
			loss.GetData(),
			reg.GetData(),
			totLoss.GetData(),
		)

		// early exit (when results are good enough)
		if loss.GetData() < 0.0001 {
			break
		}
	}
	fmt.Println("==> DONE")
}
