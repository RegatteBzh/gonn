package main

import (
	"fmt"

	"github.com/fsamin/gonn"
)

func main() {
	nn := gonn.DefaultRBFNetwork(2, 1, 4, true)
	inputs := [][]float64{
		[]float64{0, 0},
		[]float64{0, 1},
		[]float64{1, 0},
		[]float64{1, 1},
		[]float64{2, 2},
	}

	targets := [][]float64{
		[]float64{0}, //0+0=0
		[]float64{1}, //0+1=1
		[]float64{1}, //1+0=1
		[]float64{2}, //1+1=2
		[]float64{4}, //1+1=2
	}

	nn.Train(inputs, targets, 1000)

	for _, p := range inputs {
		fmt.Println(nn.Forward(p))
	}

	fmt.Println(nn.Forward([]float64{2, 2}))

}
