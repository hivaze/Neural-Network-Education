package me.hivaze.neural;

import me.hivaze.utils.Pair;

import java.io.*;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class NeuralNetwork {

    private final NeuronLayer inputLayer, outputLayer;
    private final NeuronLayer[] hiddenLayers;
    private final Initialization initializationMethod;

    public NeuralNetwork(int inputSize, int[] hiddenLayerSizes, int outputSize, Random random, Neuron.Activation activation, Initialization initializationMethod) {
        this.hiddenLayers = new NeuronLayer[hiddenLayerSizes.length];
        NeuronLayer parent = inputLayer = new NeuronLayer(inputSize, null, null, random, Neuron.Activation.LINEAR);
        for (Neuron neuron : inputLayer.getNeurons()) neuron.initializeWeights(1);
        for (int i = 0; i < hiddenLayers.length; i++) {
            parent = this.hiddenLayers[i] = new NeuronLayer(hiddenLayerSizes[i], parent, null, random, activation);
        }
        this.outputLayer = new NeuronLayer(outputSize, parent, null, random, activation);
        this.initializationMethod = initializationMethod;
    }

    public NeuralNetwork(Path filePath) throws IOException, ClassNotFoundException {
        File file = filePath.toFile();
        assert file.getName().endsWith(".dnn");
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
            this.hiddenLayers = new NeuronLayer[ois.readInt()];
            Neuron[] neurons = (Neuron[]) ois.readObject();
            NeuronLayer parent = inputLayer = new NeuronLayer(neurons, null, null);
            for (int i = 0; i < hiddenLayers.length; i++) {
                neurons = (Neuron[]) ois.readObject();
                parent = this.hiddenLayers[i] = new NeuronLayer(neurons, parent, null);
            }
            this.outputLayer = new NeuronLayer((Neuron[]) ois.readObject(), parent, null);
            this.initializationMethod = (Initialization) ois.readObject();
        }
    }

    public long mutationTrain(List<Pair<double[]>> trainingDataset, float mutateChance, double minError, long maxIterations) {
        long iterations = 0;
        for (;;) {
            iterations++;
            boolean valid = true;
            inputsLoop: for (Pair<double[]> pair : trainingDataset) {
                double[] output = output(pair.getFirst());
                for (int j = 0; j < output.length; j++) {
                    double error = Math.abs(pair.getSecond()[j] - output[j]);
                    if (error > minError) {
                        valid = false;
                        break inputsLoop;
                    }
                }
            }
            if (valid) break;
            else if (iterations < maxIterations) {
                // Mutation
                float rate = 0.1f;
                for (NeuronLayer layer = inputLayer.getOutput(); layer != null; layer = layer.getOutput()) {
                    for (Neuron neuron : layer.getNeurons()) {
                        if (neuron.getLocalRandom().nextFloat() <= mutateChance) {
                            for (int i = 0; i < neuron.getWeights().length; i++) {
                                neuron.getWeights()[i] += (1 - 2 * neuron.getLocalRandom().nextFloat()) * rate;
                            }
                            neuron.setupBias(neuron.getBias() + (1 - 2 * neuron.getLocalRandom().nextFloat()) * rate);
                            neuron.normalize(1d);
                        }
                    }
                }
            } else break;
        }
        return iterations;
    }

    public long backPropagationTrain(List<Pair<double[]>> trainingDataset, double learningSpeed, double[] minAllowedError, long maxIterations) {
        assert minAllowedError.length == outputLayer.getNeurons().length;
        assert inputLayer.getNeuron(0).hasCache();
        long iterations = 0;
        for (; iterations < maxIterations; iterations++) {
            double[] currentError = new double[outputLayer.getNeurons().length];
            for (Pair<double[]> pair : trainingDataset) {
                double[] clearInput = pair.getFirst(), targetOutput = pair.getSecond();
                double[] networkAnswer = output(clearInput);
                double[] errors = new double[networkAnswer.length];
                for (int j = 0; j < errors.length; j++) {
                    Neuron outputNeuron = outputLayer.getNeuron(j);
                    double error = targetOutput[j] - networkAnswer[j];
                    currentError[j] = Math.max(currentError[j], Math.abs(error));
                    errors[j] = error * outputNeuron.getActivation().getDerivate().apply(outputNeuron.cache().getRawSum());
                    for (int k = 0; k < outputNeuron.getWeights().length; k++) {
                        outputNeuron.cache().correctWeight(k, learningSpeed * errors[j] * outputLayer.getInput().getNeuron(k).cache().getNormalizedSum());
                    }
                    outputNeuron.cache().correctBias(learningSpeed * errors[j]);
                }
                for (NeuronLayer layer = outputLayer.getInput(); layer != inputLayer; layer = layer.getInput()) {
                    double[] newErrors = new double[layer.getNeurons().length];
                    for (int j = 0; j < layer.getNeurons().length; j++) {
                        Neuron hiddenNeuron = layer.getNeuron(j);
                        for (int k = 0; k < errors.length; k++) {
                            newErrors[j] += errors[k] * layer.getOutput().getNeuron(k).getWeights()[j];
                        }
                        newErrors[j] *= hiddenNeuron.getActivation().getDerivate().apply(hiddenNeuron.cache().getRawSum());
                        for (int k = 0; k < hiddenNeuron.getWeights().length; k++) {
                            hiddenNeuron.cache().correctWeight(k, learningSpeed * newErrors[j] * layer.getInput().getNeuron(k).cache().getNormalizedSum());
                        }
                        hiddenNeuron.cache().correctBias(learningSpeed * newErrors[j]);
                    }
                    errors = newErrors;
                }
                for (NeuronLayer layer = inputLayer.getOutput(); layer != null; layer = layer.getOutput()) {
                    for (Neuron neuron : layer.getNeurons()) {
                        neuron.executeCorrection();
                    }
                }
            }
//            if (iterations % 1000 == 0) System.out.println(iterations + " - " + Arrays.toString(currentError));
            boolean valid = true;
            errorCheck: for (int i = 0; i < currentError.length; i++) {
                if (currentError[i] > minAllowedError[i]) {
                    valid = false;
                    break errorCheck;
                }
            }
            if (valid) break;
            // else TODO: shuffle inputs & target
        }
        return iterations;
    }

    public double[] output(double[] inputs) {
        assert inputs.length == inputLayer.getNeurons().length;
        double[] convertedInput = new double[inputs.length];
        System.arraycopy(inputs, 0, convertedInput, 0, inputs.length);
        for (NeuronLayer layer = inputLayer; layer != null; layer = layer.getOutput()) {
            convertedInput = layer.output(convertedInput);
        }
        return convertedInput;
    }

    public NeuralNetwork randomise() {
        for (NeuronLayer layer = inputLayer.getOutput(); layer != null; layer = layer.getOutput()) {
            for (Neuron neuron : layer.getNeurons()) {
                for (int i = 0; i < neuron.getWeights().length; i++) {
                    neuron.getWeights()[i] = 0.5d - neuron.getLocalRandom().nextDouble();
                }
                neuron.setupBias(0.5d - neuron.getLocalRandom().nextDouble());
            }
        }
        if (initializationMethod == Initialization.NGUYEN_WIDROW) {
            for (NeuronLayer layer = inputLayer.getOutput(); layer != outputLayer; layer = layer.getOutput()) {
                int neuronsCount = layer.getNeurons().length;
                int inputNeuronsCount = layer.getInput().getNeurons().length;
                double beta = 0.7d * Math.pow(neuronsCount, 1d / inputNeuronsCount);
                for (Neuron neuron : layer.getNeurons()) {
                    double sum = 0;
                    for (int i = 0; i < neuron.getWeights().length; i++) {
                        sum += Math.pow(neuron.getWeights()[i], 2);
                    }
                    sum = Math.sqrt(sum);
                    for (int i = 0; i < neuron.getWeights().length; i++) {
                        neuron.getWeights()[i] = beta * neuron.getWeights()[i] / sum;
                    }
                    neuron.setupBias((1 - 2 * neuron.getLocalRandom().nextDouble()) * beta);
                }
            }
        }
        return this;
    }

    public NeuralNetwork enableStateCaches() {
        for (NeuronLayer layer = inputLayer; layer != null; layer = layer.getOutput()) {
            for (Neuron neuron : layer.getNeurons()) {
                neuron.enableCache();
            }
        }
        return this;
    }

    public NeuralNetwork saveTo(Path path) throws IOException {
        File file = path.toFile();
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file))) {
            oos.writeInt(hiddenLayers.length);
            for (NeuronLayer layer = inputLayer; layer != null; layer = layer.getOutput()) {
                oos.writeObject(layer.getNeurons());
            }
            oos.writeObject(initializationMethod);
        }
        return this;
    }

    public NeuronLayer getInputLayer() {
        return inputLayer;
    }

    public NeuronLayer getOutputLayer() {
        return outputLayer;
    }

    public NeuronLayer[] getHiddenLayers() {
        return hiddenLayers;
    }

    public Initialization getInitializationMethod() {
        return initializationMethod;
    }

    @Override
    public String toString() {
        return "NeuralNetwork{" + "inputLayer=" + inputLayer + ", outputLayer=" + outputLayer + ", hiddenLayers=" + Arrays.toString(hiddenLayers) + ", initializationMethod=" + initializationMethod + '}';
    }


    public enum Initialization implements Serializable { RANDOM, NGUYEN_WIDROW }

}