package gonn

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

type NeuralNetwork struct {
	HiddenLayer      []float64
	InputLayer       []float64
	OutputLayer      []float64
	WeightHidden     [][]float64
	WeightOutput     [][]float64
	ErrOutput        []float64
	ErrHidden        []float64
	LastChangeHidden [][]float64
	LastChangeOutput [][]float64
	Regression       bool
	Rate1            float64 //learning rate
	Rate2            float64
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Pow(math.E, -x))
}

func dsigmoid(Y float64) float64 {
	return Y * (1.0 - Y)
}

func makeMatrix(rows, colums int, value float64) [][]float64 {
	mat := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		mat[i] = make([]float64, colums)
		for j := 0; j < colums; j++ {
			mat[i][j] = value
		}
	}
	return mat
}

func randomMatrix(rows, colums int, lower, upper float64) [][]float64 {
	mat := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		mat[i] = make([]float64, colums)
		for j := 0; j < colums; j++ {
			mat[i][j] = rand.Float64()*(upper-lower) + lower
		}
	}
	return mat
}

//DefaultNetwork intializes a default network
func DefaultNetwork(iInputCount, iHiddenCount, iOutputCount int, iRegression bool) *NeuralNetwork {
	return NewNetwork(iInputCount, iHiddenCount, iOutputCount, iRegression, 0.25, 0.1)
}

//NewNetwork initializes a new Network
func NewNetwork(iInputCount, iHiddenCount, iOutputCount int, iRegression bool, iRate1, iRate2 float64) *NeuralNetwork {
	iInputCount++
	iHiddenCount++
	rand.Seed(time.Now().UnixNano())
	return &NeuralNetwork{
		Regression:       iRegression,
		Rate1:            iRate1,
		Rate2:            iRate2,
		InputLayer:       make([]float64, iInputCount),
		OutputLayer:      make([]float64, iOutputCount),
		ErrOutput:        make([]float64, iOutputCount),
		ErrHidden:        make([]float64, iHiddenCount),
		WeightHidden:     randomMatrix(iHiddenCount, iInputCount, -1.0, 1.0),
		WeightOutput:     randomMatrix(iOutputCount, iHiddenCount, -1.0, 1.0),
		LastChangeHidden: makeMatrix(iHiddenCount, iInputCount, 0.0),
		LastChangeOutput: makeMatrix(iOutputCount, iHiddenCount, 0.0),
	}
}

//Forward push weights to the next layer
func (n *NeuralNetwork) Forward(input []float64) ([]float64, error) {
	if len(input)+1 != len(n.InputLayer) {
		return nil, fmt.Errorf("amount of input variable doesn't match")
	}
	for i := 0; i < len(input); i++ {
		n.InputLayer[i] = input[i]
	}
	n.InputLayer[len(n.InputLayer)-1] = 1.0 //bias node for input layer

	for i := 0; i < len(n.HiddenLayer)-1; i++ {
		sum := 0.0
		for j := 0; j < len(n.InputLayer); j++ {
			sum += n.InputLayer[j] * n.WeightHidden[i][j]
		}
		n.HiddenLayer[i] = sigmoid(sum)
	}

	n.HiddenLayer[len(n.HiddenLayer)-1] = 1.0 //bias node for hidden layer
	for i := 0; i < len(n.OutputLayer); i++ {
		sum := 0.0
		for j := 0; j < len(n.HiddenLayer); j++ {
			sum += n.HiddenLayer[j] * n.WeightOutput[i][j]
		}
		if n.Regression {
			n.OutputLayer[i] = sum
		} else {
			n.OutputLayer[i] = sigmoid(sum)
		}
	}
	return n.OutputLayer, nil
}

//Feedback get weights to the previous layer
func (n *NeuralNetwork) Feedback(target []float64) {
	for i := 0; i < len(n.OutputLayer); i++ {
		n.ErrOutput[i] = n.OutputLayer[i] - target[i]
	}

	for i := 0; i < len(n.HiddenLayer)-1; i++ {
		err := 0.0
		for j := 0; j < len(n.OutputLayer); j++ {
			if n.Regression {
				err += n.ErrOutput[j] * n.WeightOutput[j][i]
			} else {
				err += n.ErrOutput[j] * n.WeightOutput[j][i] * dsigmoid(n.OutputLayer[j])
			}

		}
		n.ErrHidden[i] = err
	}

	for i := 0; i < len(n.OutputLayer); i++ {
		for j := 0; j < len(n.HiddenLayer); j++ {
			var delta float64
			if n.Regression {
				delta = n.ErrOutput[i]
			} else {
				delta = n.ErrOutput[i] * dsigmoid(n.OutputLayer[i])
			}
			change := n.Rate1*delta*n.HiddenLayer[j] + n.Rate2*n.LastChangeOutput[i][j]
			n.WeightOutput[i][j] -= change
			n.LastChangeOutput[i][j] = change

		}
	}

	for i := 0; i < len(n.HiddenLayer)-1; i++ {
		for j := 0; j < len(n.InputLayer); j++ {
			delta := n.ErrHidden[i] * dsigmoid(n.HiddenLayer[i])
			change := n.Rate1*delta*n.InputLayer[j] + n.Rate2*n.LastChangeHidden[i][j]
			n.WeightHidden[i][j] -= change
			n.LastChangeHidden[i][j] = change

		}
	}
}

//CalcError computes error rate
func (n *NeuralNetwork) CalcError(target []float64) float64 {
	errSum := 0.0
	for i := 0; i < len(n.OutputLayer); i++ {
		err := n.OutputLayer[i] - target[i]
		errSum += 0.5 * err * err
	}
	return errSum
}

func genRandomIdx(N int) []int {
	A := make([]int, N)
	for i := 0; i < N; i++ {
		A[i] = i
	}
	//randomize
	for i := 0; i < N; i++ {
		j := i + int(rand.Float64()*float64(N-i))
		A[i], A[j] = A[j], A[i]
	}
	return A
}

//Train is the main function of a network. It trains all the network for a given iteration numbers
func (n *NeuralNetwork) Train(inputs [][]float64, targets [][]float64, iteration int) error {
	if len(inputs[0])+1 != len(n.InputLayer) {
		return fmt.Errorf("amount of input variable doesn't match")
	}
	if len(targets[0]) != len(n.OutputLayer) {
		return fmt.Errorf("amount of output variable doesn't match")
	}

	for i := 0; i < iteration; i++ {
		var curErr float64
		var idxAry = genRandomIdx(len(inputs))
		for j := 0; j < len(inputs); j++ {
			if _, err := n.Forward(inputs[idxAry[j]]); err != nil {
				return err
			}
			n.Feedback(targets[idxAry[j]])
			curErr += n.CalcError(targets[idxAry[j]])
			if (j+1)%1000 == 0 {
				fmt.Printf("iteration %vth / progress %.2f %% \r", i+1, float64(j)*100/float64(len(inputs)))
			}
		}
		if (iteration >= 10 && (i+1)%(iteration/10) == 0) || iteration < 10 {
			fmt.Printf("\niteration %vth MSE: %.5f", i+1, curErr/float64(len(inputs)))
		}
	}
	fmt.Println("\ndone.")
	return nil
}
