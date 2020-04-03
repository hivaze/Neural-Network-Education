package me.hivaze.neural;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

public class Neuron implements Serializable {

    private final double[] weights;
    private double bias = 0;

    private final Random localRandom;
    private final Activation activation;
    private transient Cache cache;

    public Neuron(int inputs, Random random, Activation activation) {
        this.weights = new double[inputs];
        this.localRandom = random;
        this.activation = activation;
        this.cache = null;
    }

    public double output(double[] inputs, int startFromInput) {
        assert inputs.length >= weights.length; // for the first layer inputs.length
        double sum = 0;
        for (int i = 0; i < weights.length; sum += inputs[startFromInput + i] * weights[i], i++);
        sum += bias;
        if (cache != null) cache.rawSum = sum;
        sum = activation.function.apply(sum);
        if (cache != null) cache.normalizedSum = sum;
        return sum;
    }

    public Neuron normalize(double range) {
        assert range > 0;
        for (int i = 0; i < weights.length; i++) {
            if (weights[i] < -range) weights[i] = -range;
            else if (weights[i] > range) weights[i] = range;
        }
        if (bias < -range) bias = -range;
        if (bias > range) bias = range;
        return this;
    }

    public Neuron initializeWeights(double value) {
        Arrays.fill(weights, value);
        return this;
    }

    public Neuron setupBias(double bias) {
        this.bias = bias;
        return this;
    }

    public Neuron executeCorrection() {
        for (int i = 0; i < cache.weightCorrection.length; i++) {
            this.weights[i] += cache.weightCorrection[i];
        }
        this.bias += cache.biasCorrection;
        return this;
    }

    public double[] getWeights() {
        return weights;
    }

    public double getBias() {
        return bias;
    }

    public Random getLocalRandom() {
        return localRandom;
    }

    public Activation getActivation() {
        return activation;
    }

    public Cache cache() {
        return cache;
    }

    public Neuron enableCache() {
        if (cache == null) cache = new Cache(weights.length);
        return this;
    }

    public Neuron disableCache() {
        cache = null;
        return this;
    }

    public boolean hasCache() {
        return cache != null;
    }

    @Override
    public String toString() {
        return "Neuron{" + "weights=" + Arrays.toString(weights) + ", bias=" + bias + ", activation=" + activation + '}';
    }

    public static class Cache {

        private double rawSum = 0, normalizedSum = 0;
        private double weightCorrection[], biasCorrection = 0;

        public Cache(int weights) {
            this.weightCorrection = new double[weights];
        }

        public void correctWeight(int index, double value) {
            this.weightCorrection[index] = value;
        }

        public void correctBias(double value) {
            this.biasCorrection = value;
        }

        public double getRawSum() {
            return rawSum;
        }

        public double getNormalizedSum() {
            return normalizedSum;
        }

        public double[] getWeightCorrection() {
            return weightCorrection;
        }

        public double getBiasCorrection() {
            return biasCorrection;
        }

    }


    public enum Activation implements Serializable {

        LINEAR(x -> x, x -> 1d),

        TANH(Math::tanh, x -> 1d - Math.pow(Math.tanh(x), 2)),

        ReLU(x -> Math.max(0, x), x -> {
            if (x < 0) return 0d;
            if (x == 0) return Double.NaN;
            else return 1d;
        }),

        SIGMOID(x -> 1d / (1d + Math.pow(Math.E, -x)),
                x -> {
                    double temp = 1d / (1d + Math.pow(Math.E, -x));
                    return temp * (1 - temp);
                }),

        BIPOLAR_SIGMOID(x -> 2d / (1d + Math.pow(Math.E, -x)) - 1d,
                x -> {
                    double temp = 2d / (1d + Math.pow(Math.E, -x)) - 1d;
                    return 0.5d * (1 + temp) * (1 - temp);
                });

        private final Function<Double, Double> function, derivate;

        Activation(Function<Double, Double> function, Function<Double, Double> derivate) {
            this.function = function;
            this.derivate = derivate;
        }

        public Function<Double, Double> getFunction() {
            return function;
        }

        public Function<Double, Double> getDerivate() {
            return derivate;
        }

    }

}