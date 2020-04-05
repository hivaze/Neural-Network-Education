package me.hivaze.tests;

import me.hivaze.neural.NeuralNetwork;
import me.hivaze.neural.Neuron;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class BasicTests {

    private final Random random = new Random();

    @Test
    public void creationTest() {
        NeuralNetwork neuralNetwork = new NeuralNetwork(1, new int[] { 2 }, 1, random,
                Neuron.Activation.TANH, NeuralNetwork.Initialization.RANDOM);
        assertEquals(neuralNetwork.getHiddenLayers().length, 1);
        assertEquals(neuralNetwork.getHiddenLayers()[0].getNeurons().length, 2);
        assertEquals(neuralNetwork.getInitializationMethod(), NeuralNetwork.Initialization.RANDOM);
    }

    @Test
    public void serializationTest(@TempDir Path tempDir) throws IOException, ClassNotFoundException {
        Path destination = tempDir.resolve("test.dnn");
        NeuralNetwork neuralNetwork = new NeuralNetwork(500, new int[] { 2000, 5000, 3000, 5000, 2000 }, 100, random,
                Neuron.Activation.SIGMOID, NeuralNetwork.Initialization.NGUYEN_WIDROW).randomise();
        neuralNetwork.saveTo(destination);
        System.out.println("File size: " + destination.toFile().length());
        NeuralNetwork loadedNetwork = new NeuralNetwork(destination);
        assertEquals(neuralNetwork.getInputLayer().getNeurons().length, loadedNetwork.getInputLayer().getNeurons().length);
        assertEquals(neuralNetwork.getHiddenLayers().length, loadedNetwork.getHiddenLayers().length);
        for (int i = 0; i < loadedNetwork.getHiddenLayers().length; i++) {
            assertEquals(neuralNetwork.getHiddenLayers()[i].getNeurons().length, loadedNetwork.getHiddenLayers()[i].getNeurons().length);
        }
        assertEquals(neuralNetwork.getOutputLayer().getNeurons().length, loadedNetwork.getOutputLayer().getNeurons().length);
        double[] input = random.doubles().limit(500).toArray();
        assertArrayEquals(neuralNetwork.output(input),  loadedNetwork.output(input));
        System.out.println(Arrays.toString(loadedNetwork.output(input)));
    }

}