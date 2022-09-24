package grad

import (
	"math/rand"
	"time"
)

// base interface
type base interface {
	params() []*Value
}

// feed forward neural network neuron implementation
type Neuron struct {
	weights []*Value
	bias    *Value
	nonlin  bool
}

func NewNeuron(inputNum int, nonlin bool) *Neuron {
	n := Neuron{
		weights: nil,
		bias:    NewValue(0.0),
	}
	n.weights = randFloats(-1.0, 1.0, inputNum)
	n.nonlin = nonlin

	return &n
}

func (n Neuron) feedForward(inputs []*Value) *Value {
	dot := NewValue(0.0)

	for i := range n.weights {
		dot = dot.Add(n.weights[i].Mul(inputs[i]))
	}
	dot = dot.Add(n.bias)

	return dot

}

func (n Neuron) params() []*Value {
	var ps []*Value
	// for _, el := range n.weights {
	// 	ps = append(ps, el)
	// }
	ps = append(ps, n.weights...)
	ps = append(ps, n.bias)

	return ps
}

func (n *Neuron) zeroGrad() {
	for _, p := range n.params() {
		p.SetGrad(0.0)
	}
}

// feed forward neural network layer implementation
type Layer struct {
	neurons []*Neuron
}

func NewLayer(inputNum int, neuronsNum int, nonlin bool) *Layer {
	l := Layer{neurons: []*Neuron{}}
	for i := 0; i < neuronsNum; i++ {
		l.neurons = append(l.neurons, NewNeuron(inputNum, nonlin))
	}

	return &l
}

func (l Layer) feedForward(inputs []*Value) []*Value {
	var temp []*Value

	for _, n := range l.neurons {
		temp = append(temp, n.feedForward(inputs))
	}

	return temp
}

func (l Layer) params() []*Value {
	var ps []*Value
	for _, n := range l.neurons {
		// for _, p := range n.params() {
		// 	ps = append(ps, p)
		// }
		ps = append(ps, n.params()...)
	}

	return ps
}

func (l *Layer) zeroGrad() {
	for _, p := range l.params() {
		p.SetGrad(0.0)
	}
}

// feed forward fully connected nn implementation
type Model struct {
	layers []*Layer
}

func NewModel(inputNum int, networkArch []int) *Model {
	// seeding random number generator with time in nanoseconds
	rand.Seed(time.Now().UnixNano())

	arch := []int{inputNum}
	arch = append(arch, networkArch...)

	m := Model{layers: nil}
	// looping through networkArch but reading from arch
	// remember that len(networkArch)=len(arch)-1
	for l := range networkArch {
		m.layers = append(m.layers, NewLayer(arch[l], arch[l+1], l != len(networkArch)-1))
	}

	return &m
}

func (m Model) FeedForward(inputs [2]float64) []*Value {
	// convert inputs to Value instances
	var inps []*Value
	for _, el := range inputs {
		inps = append(inps, NewValue(el))
	}

	for _, l := range m.layers {
		inps = l.feedForward(inps)
	}

	return inps
}

func (m Model) Params() []*Value {
	var ps []*Value
	for _, l := range m.layers {
		// for _, p := range l.params() {
		// 	ps = append(ps, p)
		// }
		ps = append(ps, l.params()...)
	}

	return ps
}

func (m *Model) ZeroGrad() {
	for _, p := range m.Params() {
		p.SetGrad(0.0)
	}
}
