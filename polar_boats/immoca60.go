package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path"
	"regexp"
	"strconv"

	"github.com/regattebzh/gonn"
)

// Polar is a polar for a given sail and a given angle
type Polar struct {
	Angle float64
	Speed []float64
}

// SailCharacteristic is the characteristic of a sail
type SailCharacteristic struct {
	Name   string
	Winds  []float64
	Polars []Polar
}

type loadStruct struct {
	File io.Reader
	Name string
}

func knotToMeter(knot float64) float64 {
	return knot * float64(0.514444)
}

func loadAllPolars(pathName string, shipName string) (models []*gonn.RBFNetwork, err error) {

	dic := make(map[string]string)
	dic["1"] = "foc"
	dic["2"] = "spi"
	dic["4"] = "foc2"
	dic["8"] = "genois"
	dic["16"] = "zero-code"
	dic["32"] = "light-spi"
	dic["64"] = "gennaker"

	files, err := ioutil.ReadDir(pathName)
	if err != nil {
		log.Fatal(err)
	}

	filter, err := regexp.Compile(`(\d*)\.csv$`)
	if err != nil {
		log.Fatal(err)
	}

	var toLoad []loadStruct
	for _, f := range files {
		match := filter.FindStringSubmatch(f.Name())
		if len(match) > 0 {
			filename := path.Join(pathName, f.Name())
			file, err := os.Open(filename)
			if err != nil {
				log.Fatal(err)
			}
			defer file.Close()
			newLoader := loadStruct{
				File: file,
				Name: shipName + "-" + dic[match[1]],
			}
			toLoad = append(toLoad, newLoader)
		}
	}

	models = make([]*gonn.RBFNetwork, len(toLoad))

	for index, elt := range toLoad {
		inputs, targets, err := csvLoader(elt.File, elt.Name)
		if err != nil {
			log.Fatal(err)
		}
		models[index] = gonn.DefaultRBFNetwork(2, 1, 4, true)
		models[index].Train(inputs, targets, 1000)
	}

	return
}

func csvLoader(csvFile io.Reader, name string) (input [][]float64, output [][]float64, err error) {

	reader := csv.NewReader(csvFile)
	reader.Comma = ';'
	reader.FieldsPerRecord = -1

	fmt.Printf("Loading %s => ", name)

	csvData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	winds := make([]float64, len(csvData[0])-1)
	for windIndex, wind := range csvData[0][1:] {
		windLevel, err := strconv.ParseFloat(wind, 32)
		if err != nil {
			log.Fatal("Error parsing wind data")
		}
		// knot to m/s conversion
		winds[windIndex] = knotToMeter(windLevel)
	}

	outData := csvData[1:] // ignore first line

	sampleSize := len(winds) * len(outData)

	fmt.Printf("%d samples\n", sampleSize)

	input = make([][]float64, sampleSize)
	output = make([][]float64, sampleSize)
	for angleIndex, polarSample := range outData {

		angle, err := strconv.ParseFloat(polarSample[0], 32)
		if err != nil {
			log.Fatal("Error parsing wind angle")
		}

		for speedIndex, speed := range polarSample[1:] {
			sampleIndex := len(winds)*angleIndex + speedIndex
			newPolarVal, err := strconv.ParseFloat(speed, 32)
			if err != nil {
				log.Fatal("Error parsing wind speed")
			}
			input[sampleIndex] = []float64{angle, winds[speedIndex]}
			output[sampleIndex] = []float64{newPolarVal}
		}
	}

	fmt.Printf("%v ...\n", input[:3])
	fmt.Printf("%v ...\n", output[:3])

	return
}

func main() {

	models, _ := loadAllPolars("./polaires/imoca60", "imoca60")

	/*nn := gonn.DefaultRBFNetwork(2, 1, 4, true)
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
	}*/

	fmt.Println(models[2].Forward([]float64{2, 2}))

}
