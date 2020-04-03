package me.hivaze.neural;

import java.util.Arrays;
import java.util.Random;

public class NeuronLayer {

    private final Neuron[] neurons;
    private transient NeuronLayer input = null, output = null;

    NeuronLayer(int neurons, NeuronLayer input, NeuronLayer output, Random random, Neuron.Activation activation) {
        this.neurons = new Neuron[neurons];
        for (int i = 0; i < neurons; i++) {
            this.neurons[i] = new Neuron(input != null ? input.neurons.length : 1, random, activation);
        }
        if (input != null) {
            this.input = input;
            input.output = this;
        }
        if (output != null) {
            this.output = output;
            output.input = this;
        }
    }

    NeuronLayer(Neuron[] neurons, NeuronLayer input, NeuronLayer output) {
        this.neurons = neurons;
        if (input != null) {
            this.input = input;
            input.output = this;
        }
        if (output != null) {
            this.output = output;
            output.input = this;
        }
    }

    public double[] output(double[] inputs) {
        double[] result = new double[neurons.length];;
        for (int i = 0; i < neurons.length; i++) {
            result[i] = neurons[i].output(inputs, input == null ? i : 0);
        }
        return result;
    }

    public boolean isHiddenLayer() {
        return input != null && output != null;
    }

    public boolean inInputLayer() {
        return input == null;
    }

    public boolean isOutputLayer() {
        return output == null;
    }

    public NeuronLayer getInput() {
        return input;
    }

    public NeuronLayer getOutput() {
        return output;
    }

    public Neuron getNeuron(int index) {
        return neurons[index];
    }

    public Neuron[] getNeurons() {
        return neurons;
    }

    @Override
    public String toString() {
        return "NeuronLayer{" + "neurons=" + Arrays.toString(neurons) + '}';
    }

}