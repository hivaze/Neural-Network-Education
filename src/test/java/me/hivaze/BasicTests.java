package me.hivaze;

import me.hivaze.neural.NeuralNetwork;
import me.hivaze.neural.Neuron;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Random;

public class BasicTests {

    @Test
    public void creationTest() {
        NeuralNetwork neuralNetwork = new NeuralNetwork(1, new int[] { 2 }, 1, new Random(), Neuron.Activation.TANH, NeuralNetwork.Initialization.RANDOM);
        assertEquals(neuralNetwork.getHiddenLayers().length, 1);
        assertEquals(neuralNetwork.getHiddenLayers()[0].getNeurons().length, 2);
        assertEquals(neuralNetwork.getInitializationMethod(), NeuralNetwork.Initialization.RANDOM);
    }

}