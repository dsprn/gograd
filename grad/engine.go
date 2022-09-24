package grad

import (
	"math"
)

// === START OF OPERATIONS ARITHMETIC TYPE ===
type operation interface {
	isOperation()
}

// === START OF VARIANTS ===
// addition variant
type Addition struct {
	operand string
}

func NewAddition() Addition {
	return Addition{
		operand: "+",
	}
}

func (a Addition) isOperation() {}

// subtraction variant
type Subtraction struct {
	operand string
}

func NewSubtraction() Subtraction {
	return Subtraction{
		operand: "-",
	}
}

func (s Subtraction) isOperation() {}

// multiplication variant
type Multiplication struct {
	operand string
}

func NewMultiplication() Multiplication {
	return Multiplication{
		operand: "*",
	}
}

func (m Multiplication) isOperation() {}

// division variant
type Division struct {
	operand string
}

func NewDivision() Division {
	return Division{
		operand: "/",
	}
}

func (d Division) isOperation() {}

// negation variant
type Negation struct {
	operand string
}

func NewNegation() Negation {
	return Negation{
		operand: "*-",
	}
}

func (d Negation) isOperation() {}

// power variant
type Power struct {
	operand string
}

func NewPower() Power {
	return Power{
		operand: "^",
	}
}

func (p Power) isOperation() {}

// relu variant
type Relu struct {
	operand string
}

func NewRelu() Relu {
	return Relu{
		operand: "ReLU",
	}
}

func (r Relu) isOperation() {}

// none variant
type None struct {
	operand string
}

func NewNone() None {
	return None{
		operand: "None",
	}
}

func (n None) isOperation() {}

// === END OF VARIANTS ===
// === END OF OPERATIONS ARITHMETIC TYPE ===

// implementing a SET type for node's children
// as golang does not have such a type
type cset map[*Value]struct{}

func NewSet(childA *Value, childB *Value) cset {
	children := make(map[*Value]struct{})

	if childA != nil {
		children[childA] = struct{}{}
	}
	if childB != nil {
		children[childB] = struct{}{}
	}

	return children
}

// a representation of each value for/computed by the engine
type Value struct {
	data     float64
	grad     float64
	nonlin   bool
	children cset
	backward func()
	op       operation
}

// Value constructor
func NewValue(d float64) *Value {
	v := Value{
		data:     d,
		grad:     0,
		op:       NewNone(),
		children: NewSet(nil, nil),
		backward: func() {},
	}

	return &v
}

func (v Value) GetData() float64 {
	return v.data
}

func (v *Value) Update(learningRate float64) {
	v.data -= learningRate * v.grad
}

func (v Value) GetGrad() float64 {
	return v.grad
}

func (v *Value) SetGrad(newGrad float64) {
	v.grad = newGrad
}

func (v *Value) Add(other *Value) *Value {
	out := Value{
		data:     v.data + other.data,
		grad:     0,
		op:       NewAddition(),
		children: NewSet(v, other),
		backward: func() {},
	}

	// addition derivative
	out.backward = func() {
		v.grad += 1 * out.grad
		other.grad += 1 * out.grad
	}

	return &out
}

func (v *Value) Mul(other *Value) *Value {
	out := Value{
		data:     v.data * other.data,
		grad:     0,
		op:       NewMultiplication(),
		children: NewSet(v, other),
		backward: func() {},
	}

	// multiplication derivative
	out.backward = func() {
		v.grad += other.data * out.grad
		other.grad += v.data * out.grad
	}

	return &out
}

func (v *Value) Sub(other *Value) *Value {
	return v.Add(other.Neg())
}

// func (v *Value) Div(other *Value) *Value {
// 	return v.Mul(other.Pow(-1))
// }

func (v *Value) Div(other interface{}) *Value {
	var o *Value

	switch other.(type) {
	case float64:
		o = NewValue(other.(float64))
	case int:
		o = NewValue(float64(other.(int)))
	case *Value:
		o = other.(*Value)
	}

	return v.Mul(o.Pow(-1))
}

func (v *Value) Neg() *Value {
	return v.Mul(NewValue(-1.0))
}

func (v *Value) Pow(exp float64) *Value {
	out := Value{
		data:     math.Pow(v.data, exp),
		grad:     0,
		op:       NewPower(),
		children: NewSet(v, nil),
		backward: func() {},
	}

	// power derivative
	out.backward = func() {
		v.grad += exp * math.Pow(v.data, exp-1) * out.grad
	}

	return &out
}

func (v *Value) Relu() Value {
	var d float64

	if v.data <= 0.0 {
		d = 0.0
	} else {
		d = v.data
	}

	out := Value{
		data:     d,
		grad:     0,
		op:       NewRelu(),
		children: NewSet(v, nil),
		backward: func() {},
	}

	// ReLU derivative
	out.backward = func() {
		if v.data < 0 {
			v.grad += 0 * out.grad
		} else {
			v.grad += 1 * out.grad
		}
	}

	return out
}

func (v *Value) BackwardPass() {
	v.grad = 1
	visited := map[*Value]bool{}
	tpOrder := []*Value{}

	// can use Value below only if all its fields are comparable
	// but a pointer is way more lightweight
	topologicalSort(v, &visited, &tpOrder)
	for _, node := range reverse(tpOrder) {
		node.backward()
	}
}

func topologicalSort(node *Value, visited *map[*Value]bool, tpOrder *[]*Value) {
	if _, ok := (*visited)[node]; !ok {
		(*visited)[node] = true
		for child := range node.children {
			topologicalSort(child, visited, tpOrder)
		}
		*tpOrder = append(*tpOrder, node)
	}
}

func reverse(ordered []*Value) []*Value {
	reversed := ordered

	for i, j := 0, len(reversed)-1; i < j; i, j = i+1, j-1 {
		reversed[i], reversed[j] = reversed[j], reversed[i]
	}

	return reversed
}
