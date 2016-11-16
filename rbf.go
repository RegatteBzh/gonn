package gonn

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

type RBFNetwork struct {
	InputCount       int
	InputLayer       []float64
	OutputLayer      []float64
	Centers          [][]float64
	WeightOutput     [][]float64
	LastChangeOutput [][]float64
	Regression       bool
	Rate1            float64
	Rate2            float64
}

func DefaultRBFNetwork(iInputCount, iOutputCount, iCenters int, iRegression bool) *RBFNetwork {
	return NewRBFNetwork(iInputCount, iOutputCount, iCenters, iRegression, 0.25, 0.1)
}

func NewRBFNetwork(iInputCount, iOutputCount, iCenters int, iRegression bool, iRate1, iRate2 float64) *RBFNetwork {
	n := &RBFNetwork{}
	n.InputCount = iInputCount
	rand.Seed(time.Now().UnixNano())
	n.InputLayer = make([]float64, iCenters+1)
	n.OutputLayer = make([]float64, iOutputCount)
	n.Centers = make([][]float64, iCenters)
	n.WeightOutput = randomMatrix(iOutputCount, iCenters+1, -1.0, 1.0)
	n.LastChangeOutput = makeMatrix(iOutputCount, iCenters+1, 0.0)
	n.Regression = iRegression
	n.Rate1 = iRate1
	n.Rate2 = iRate2
	return n
}

func (n *RBFNetwork) Forward(input []float64) ([]float64, error) {
	return n.ForwardRBF(n.makeRBF(input))
}

func (n *RBFNetwork) ForwardRBF(input []float64) ([]float64, error) {
	if len(input)+1 != len(n.InputLayer) {
		return nil, fmt.Errorf("amount of input variable doesn't match")
	}
	for i := 0; i < len(input); i++ {
		n.InputLayer[i] = input[i]
	}
	n.InputLayer[len(n.InputLayer)-1] = 1.0 //bias node for input layer

	for i := 0; i < len(n.OutputLayer); i++ {
		sum := 0.0
		for j := 0; j < len(n.InputLayer); j++ {
			sum += n.InputLayer[j] * n.WeightOutput[i][j]
		}
		if n.Regression {
			n.OutputLayer[i] = sum
		} else {
			n.OutputLayer[i] = sigmoid(sum)
		}
	}

	return n.OutputLayer, nil
}

func (n *RBFNetwork) Feedback(target []float64) {
	for i := 0; i < len(n.OutputLayer); i++ {
		err_i := n.OutputLayer[i] - target[i]
		for j := 0; j < len(n.InputLayer); j++ {
			var delta float64
			if n.Regression {
				delta = err_i
			} else {
				delta = err_i * dsigmoid(n.OutputLayer[i])
			}
			change := n.Rate1*delta*n.InputLayer[j] + n.Rate2*n.LastChangeOutput[i][j]
			n.WeightOutput[i][j] -= change
			n.LastChangeOutput[i][j] = change
		}
	}
}

func (n *RBFNetwork) CalcError(target []float64) float64 {
	errSum := 0.0
	for i := 0; i < len(n.OutputLayer); i++ {
		err := n.OutputLayer[i] - target[i]
		errSum += 0.5 * err * err
	}
	return errSum
}

func (n *RBFNetwork) makeRBF(input []float64) []float64 {
	result := make([]float64, len(n.Centers))
	div := 0.0
	for j := 0; j < len(n.Centers); j++ {
		sum := 0.0
		for i := 0; i < n.InputCount; i++ {
			delta := input[i] - n.Centers[j][i]
			sum += delta * delta
		}
		result[j] = math.Exp(-8 * sum)
		div += result[j]
	}
	for j := 0; j < len(n.Centers); j++ {
		result[j] = result[j] / div
	}
	return result
}

func (n *RBFNetwork) Train(inputs [][]float64, targets [][]float64, iteration int) error {
	if len(inputs[0]) != n.InputCount {
		panic("amount of input variable doesn't match")
	}
	if len(targets[0]) != len(n.OutputLayer) {
		panic("amount of output variable doesn't match")
	}
	if len(n.Centers) > len(inputs) {
		panic("too many centers, should be less than samples count")
	}
	sfIDX := genRandomIdx(len(inputs))
	for i := range n.Centers {
		n.Centers[i] = inputs[sfIDX[i]] //random centers
	}

	rInputs := make([][]float64, len(inputs))
	for i := range rInputs {
		rInputs[i] = n.makeRBF(inputs[i])
	}

	for i := 0; i < iteration; i++ {
		var curErr float64
		idxAry := genRandomIdx(len(inputs))
		for j := range inputs {
			if _, err := n.ForwardRBF(rInputs[idxAry[j]]); err != nil {
				return err
			}
			n.Feedback(targets[idxAry[j]])
			curErr += n.CalcError(targets[idxAry[j]])
		}
	}
	return nil
}
