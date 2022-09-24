package grad

import (
	"testing"
)

func TestGroups(t *testing.T) {
	numberOfGroups := 5

	// dummy data and dummy labels
	dummyDataset := [][2]float64{
		{5.39412337e-01, 8.61363932e-01},
		{-1.03234535e+00, 5.77661126e-02},
		{-1.12251058e+00, 4.40911069e-01},
		{6.34512779e-01, -3.86770491e-01},
		{4.74812014e-01, 7.05693581e-01},
		{9.23972493e-01, 4.34679296e-01},
		{6.05938266e-01, -3.99049289e-01},
		{3.38158252e-01, 1.00461575e+00},
		{-9.65489273e-01, 1.44116250e-01},
		{1.73508562e+00, -3.03348212e-01},
	}
	dummyLabels := []float64{-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0}

	// grouped data and labels
	groupedValues, groupedLabels := group(dummyDataset, dummyLabels, numberOfGroups)

	// check the lenght of both data and labels
	if (len(groupedValues) != numberOfGroups) || (len(groupedLabels) != numberOfGroups) {
		t.Errorf("The grouped dataset does not contain the correct number of groups, got:%d, want:%d",
			len(groupedValues),
			numberOfGroups,
		)
	}

	// check that each element in the data groups correspond to the relative one in the data control dataset
	controlDataDataset := [][][2]float64{
		{{5.39412337e-01, 8.61363932e-01}, {-1.03234535e+00, 5.77661126e-02}},
		{{-1.12251058e+00, 4.40911069e-01}, {6.34512779e-01, -3.86770491e-01}},
		{{4.74812014e-01, 7.05693581e-01}, {9.23972493e-01, 4.34679296e-01}},
		{{6.05938266e-01, -3.99049289e-01}, {3.38158252e-01, 1.00461575e+00}},
		{{-9.65489273e-01, 1.44116250e-01}, {1.73508562e+00, -3.03348212e-01}},
	}
	for groupIdx := 0; groupIdx < len(groupedValues); groupIdx++ {
		for elIdx := 0; elIdx < len(groupedValues[groupIdx]); elIdx++ {
			if groupedValues[groupIdx][elIdx] != controlDataDataset[groupIdx][elIdx] {
				t.Errorf("The group at index:%d is not the same as the relative one in the control dataset, got:%v, want:%v",
					groupIdx,
					groupedValues[groupIdx],
					controlDataDataset[groupIdx],
				)
			}
		}
	}

	// check that each element in the labels groups correspond to the relative one in the labels control dataset
	controlLabelDataset := [][]float64{{-1.0, -1.0}, {-1.0, 1.0}, {-1.0, -1.0}, {1.0, -1.0}, {-1.0, 1.0}}
	for groupIdx := 0; groupIdx < len(groupedLabels); groupIdx++ {
		for elIdx := 0; elIdx < len(groupedLabels[groupIdx]); elIdx++ {
			if groupedLabels[groupIdx][elIdx] != controlLabelDataset[groupIdx][elIdx] {
				t.Errorf("The group at index:%d is not the same as the relative one in the control dataset, got:%v, want:%v",
					groupIdx,
					groupedLabels[groupIdx],
					controlLabelDataset[groupIdx],
				)
			}
		}
	}
}

func TestPopGroupedDataset(t *testing.T) {
	groupedDataDataset := [][][2]float64{
		{{5.39412337e-01, 8.61363932e-01}, {-1.03234535e+00, 5.77661126e-02}},
		{{-1.12251058e+00, 4.40911069e-01}, {6.34512779e-01, -3.86770491e-01}},
		{{4.74812014e-01, 7.05693581e-01}, {9.23972493e-01, 4.34679296e-01}},
		{{6.05938266e-01, -3.99049289e-01}, {3.38158252e-01, 1.00461575e+00}},
		{{-9.65489273e-01, 1.44116250e-01}, {1.73508562e+00, -3.03348212e-01}},
	}
	controlDataDataset := [][][2]float64{
		{{5.39412337e-01, 8.61363932e-01}, {-1.03234535e+00, 5.77661126e-02}},
		{{-1.12251058e+00, 4.40911069e-01}, {6.34512779e-01, -3.86770491e-01}},
		{{4.74812014e-01, 7.05693581e-01}, {9.23972493e-01, 4.34679296e-01}},
		{{-9.65489273e-01, 1.44116250e-01}, {1.73508562e+00, -3.03348212e-01}},
	}
	controlValues := [][2]float64{
		{6.05938266e-01, -3.99049289e-01}, {3.38158252e-01, 1.00461575e+00},
	}

	values, modifiedDataset := popValue(groupedDataDataset, 3)

	// check that the original dataset has now the right length
	if len(modifiedDataset) != len(controlDataDataset) {
		t.Errorf("The dataset from which a value was popped has a different length from the control one, got:%d, want:%d",
			len(modifiedDataset),
			len(controlDataDataset),
		)
	}

	// check the length of the extracted element
	if len(values) != len(controlValues) {
		t.Errorf("The popped element has a different length from the control one, got:%d, want:%d",
			len(values),
			len(controlValues),
		)
	}

	// check the extracted element's values
	for i := 0; i < len(values); i++ {
		for j := 0; j < len(values[i]); j++ {
			if values[i][j] != controlValues[i][j] {
				t.Errorf("The extracted element's value is different from the control one, got:%v, want:%v",
					values[i][j],
					controlValues[i][j],
				)
			}
		}
	}
}
