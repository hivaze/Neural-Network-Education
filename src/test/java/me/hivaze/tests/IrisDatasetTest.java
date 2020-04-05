package me.hivaze.tests;

import me.hivaze.neural.NeuralNetwork;
import me.hivaze.neural.Neuron;
import me.hivaze.utils.DatasetUtils;
import me.hivaze.utils.Pair;
import me.hivaze.utils.SimpleCSV;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

public class IrisDatasetTest {

    private final Path datasets = Paths.get("src/test/resources", "datasets");
    private final Random random = new Random();

    @Test
    public void backPropagationTest() throws IOException {
        List<List<String>> irisDataset = SimpleCSV.readFileWithoutHeaders(datasets.resolve("iris.csv"));
        Pair<List<List<String>>> split = DatasetUtils.splitInRandomCondition(irisDataset, 0.7);
        Function<List<String>, double[]> inputBuilder = row -> new double[] { Double.parseDouble(row.get(0)), Double.parseDouble(row.get(1)), Double.parseDouble(row.get(2)), Double.parseDouble(row.get(3)) };
        Function<List<String>, double[]> outputBuilder = row -> {
            switch (row.get(4)) {
                case "Iris-setosa":
                    return new double[]{ 1.0 };
                case "Iris-versicolor":
                    return new double[]{ 0.0 };
                case "Iris-virginica":
                    return new double[]{ -1.0 };
            }
            return null;
        };
        List<Pair<double[]>> trainingDataset = DatasetUtils.buildExamplesSet(split.getFirst(), inputBuilder, outputBuilder);
        List<Pair<double[]>> testDataset = DatasetUtils.buildExamplesSet(split.getSecond(), inputBuilder, outputBuilder);
        NeuralNetwork neuralNetwork = new NeuralNetwork(4, new int[] { 8, 4, 8 }, 1, random, Neuron.Activation.TANH, NeuralNetwork.Initialization.NGUYEN_WIDROW)
                .enableStateCaches().randomise();
        System.out.println(neuralNetwork.backPropagationTrain(trainingDataset, 0.005, new double[] { 0.01 }, 30_000));
        testDataset.forEach(example -> {
            double[] networkAnswer = neuralNetwork.output(example.getFirst());
            System.out.println(Arrays.toString(example.getFirst()) + " -> " + Arrays.toString(networkAnswer) + " | " + Arrays.toString(example.getSecond()));
        });
    }

}
