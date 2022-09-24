package grad

import (
	"math"
	"testing"
)

func TestSum(t *testing.T) {
	// data table for test
	sumTestTable := []struct {
		x float64
		y float64
		n float64
	}{
		{1.013451871, 0.762946109, 1.776397980},
		{0.002774926, 0.110349676, 0.113124602},
		{2.006839573, 1.887548122, 3.894387695},
		{3.119647378, 0.000538234, 3.120185612},
	}

	for _, table := range sumTestTable {
		a := NewValue(table.x)
		b := NewValue(table.y)
		out := a.Add(b)

		if math.Round(out.data*1_000_000_000)/1_000_000_000 != math.Round(table.n*1_000_000_000)/1_000_000_000 {
			t.Errorf(
				"Sum between Value types was incorrect, got: %f, want: %f.",
				math.Round(out.data*1_000_000_000)/1_000_000_000,
				math.Round(table.n*1_000_000_000)/1_000_000_000,
			)
		}
	}
}

func TestSub(t *testing.T) {
	// data table for test
	subTestTable := []struct {
		x float64
		y float64
		n float64
	}{
		{1.013451871, 0.762946109, 0.250505762},
		{0.002774926, 0.110349676, -0.107574750},
		{2.006839573, 1.887548122, 0.119291451},
		{3.119647378, 0.000538234, 3.119109144},
	}

	for _, table := range subTestTable {
		a := NewValue(table.x)
		b := NewValue(table.y)
		out := a.Sub(b)

		if math.Round(out.data*1_000_000_000)/1_000_000_000 != math.Round(table.n*1_000_000_000)/1_000_000_000 {
			t.Errorf(
				"Sum between Value types was incorrect, got: %f, want: %f.",
				math.Round(out.data*1_000_000_000)/1_000_000_000,
				math.Round(table.n*1_000_000_000)/1_000_000_000,
			)
		}
	}
}

func TestMul(t *testing.T) {
	// data table for test
	mulTestTable := []struct {
		x float64
		y float64
		n float64
	}{
		{1.013451871, 0.762946109, 0.773209162},
		{0.002774926, 0.110349676, 0.000306212},
		{2.006839573, 1.887548122, 3.788006267},
		{3.119647378, 0.000538234, 0.001679100},
	}

	for _, table := range mulTestTable {
		a := NewValue(table.x)
		b := NewValue(table.y)
		out := a.Mul(b)

		if math.Round(out.data*1_000_000_000)/1_000_000_000 != math.Round(table.n*1_000_000_000)/1_000_000_000 {
			t.Errorf(
				"Sum between Value types was incorrect, got: %f, want: %f.",
				math.Round(out.data*1_000_000_000)/1_000_000_000,
				math.Round(table.n*1_000_000_000)/1_000_000_000,
			)
		}
	}
}

func TestDiv(t *testing.T) {
	// data table for test
	divTestTable := []struct {
		x float64
		y float64
		n float64
	}{
		{1.013451871, 0.762946109, 1.328340048},
		{0.002774926, 0.110349676, 0.025146662},
		{2.006839573, 1.887548122, 1.063199157},
		{3.119647378, 0.000538234, 5796.080102706},
	}

	for _, table := range divTestTable {
		a := NewValue(table.x)
		b := NewValue(table.y)
		out := a.Div(b)

		if math.Round(out.data*1_000_000_000)/1_000_000_000 != math.Round(table.n*1_000_000_000)/1_000_000_000 {
			t.Errorf(
				"Sum between Value types was incorrect, got: %0.9f, want: %0.9f.",
				math.Round(out.data*1_000_000_000)/1_000_000_000,
				math.Round(table.n*1_000_000_000)/1_000_000_000,
			)
		}
	}
}

func TestNeg(t *testing.T) {
	// data table for test
	negTestTable := []struct {
		x float64
		n float64
	}{
		{-1.013451871, 1.013451871},
		{0.002774926, -0.002774926},
		{2.006839573, -2.006839573},
		{-3.119647378, 3.119647378},
	}

	for _, table := range negTestTable {
		a := NewValue(table.x)
		out := a.Neg()

		if math.Round(out.data*1_000_000_000)/1_000_000_000 != math.Round(table.n*1_000_000_000)/1_000_000_000 {
			t.Errorf(
				"Sum between Value types was incorrect, got: %0.9f, want: %0.9f.",
				math.Round(out.data*1_000_000_000)/1_000_000_000,
				math.Round(table.n*1_000_000_000)/1_000_000_000,
			)
		}
	}
}

func TestPow(t *testing.T) {
	// data table for test
	powerTestTable := []struct {
		x float64
		y float64
		n float64
	}{
		{1.013451871, 2.0, 1.027084695},
		{0.002774926, 5.0, 1.64535e-13},
		{2.006839573, 1.0, 2.006839573},
		{3.119647378, -1.0, 0.320549049},
		{7.342752101, 1.256999, 12.256987099},
	}

	for _, table := range powerTestTable {
		a := NewValue(table.x)
		out := a.Pow(table.y)

		if math.Round(out.data*1_000_000_000)/1_000_000_000 != math.Round(table.n*1_000_000_000)/1_000_000_000 {
			t.Errorf(
				"Sum between Value types was incorrect, got: %0.9f, want: %0.9f.",
				math.Round(out.data*1_000_000_000)/1_000_000_000,
				math.Round(table.n*1_000_000_000)/1_000_000_000,
			)
		}
	}
}

func TestRelu(t *testing.T) {
	// data table for test
	reluTestTable := []struct {
		x float64
		n float64
	}{
		{-1.013451871, 0.0},
		{0.002774926, 0.002774926},
		{2.006839573, 2.006839573},
		{-3.119647378, 0.0},
	}

	for _, table := range reluTestTable {
		a := NewValue(table.x)
		out := a.Relu()

		if math.Round(out.data*1_000_000_000)/1_000_000_000 != math.Round(table.n*1_000_000_000)/1_000_000_000 {
			t.Errorf(
				"Sum between Value types was incorrect, got: %0.9f, want: %0.9f.",
				math.Round(out.data*1_000_000_000)/1_000_000_000,
				math.Round(table.n*1_000_000_000)/1_000_000_000,
			)
		}
	}
}

func TestReverse(t *testing.T) {
	s := []*Value{
		NewValue(1.0),
		NewValue(2.0),
		NewValue(3.0),
		NewValue(4.0),
		NewValue(5.0),
		NewValue(6.0),
		NewValue(7.0),
		NewValue(8.0),
		NewValue(9.0),
	}

	got := reverse(s)
	want := []*Value{
		NewValue(9.0),
		NewValue(8.0),
		NewValue(7.0),
		NewValue(6.0),
		NewValue(5.0),
		NewValue(4.0),
		NewValue(3.0),
		NewValue(2.0),
		NewValue(1.0),
	}

	for i, g := range got {
		if g.data != want[i].data {
			t.Errorf(
				"The slice reversal didn't work as expected, at index %d got: %v, wanted: %v.",
				i,
				g,
				want[i],
			)
		}
	}
}

func TestTopologicalSort(t *testing.T) {
	visited := map[*Value]bool{}
	tpOrder := []*Value{}
	expectedOrderedValues := []float64{6.0, 36.0, 36.0, 1296.0}

	// as the topological order is not unique
	// this test could be done only with unary operations
	a := NewValue(6.0)
	b := a.Pow(2)
	c := b.Pow(1)
	d := c.Pow(2)

	topologicalSort(d, &visited, &tpOrder)

	for idx := range expectedOrderedValues {
		if tpOrder[idx].GetData() != expectedOrderedValues[idx] {
			t.Errorf(
				"The topological order is wrong. At index %d got value %f, expected value %f",
				idx,
				tpOrder[idx].GetData(),
				expectedOrderedValues[idx],
			)
		}
	}
}

func TestReversedTopological(t *testing.T) {
	visited := map[*Value]bool{}
	tpOrder := []*Value{}
	expectedOrderedValues := []float64{1296.0, 36.0, 36.0, 6.0}

	// as the topological order is not unique
	// this test could be done only with unary operations
	a := NewValue(6.0)
	b := a.Pow(2)
	c := b.Pow(1)
	d := c.Pow(2)

	topologicalSort(d, &visited, &tpOrder)
	reversedOrder := reverse(tpOrder)

	for idx := range expectedOrderedValues {
		if reversedOrder[idx].GetData() != expectedOrderedValues[idx] {
			t.Errorf(
				"The topological order is wrong. At index %d got value %f, expected value %f",
				idx,
				reversedOrder[idx].GetData(),
				expectedOrderedValues[idx],
			)
		}
	}
}
