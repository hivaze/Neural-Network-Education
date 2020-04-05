package me.hivaze.utils;

import java.util.Collections;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

public class DatasetUtils {

    public static <T> Pair<List<T>> splitInRandomCondition(List<T> dataset, double firstPartPercent) {
        int dataSetSize = dataset.size(), firstPartEnd = (int) (dataSetSize * firstPartPercent);
        Collections.shuffle(dataset);
        return new Pair<>(dataset.subList(0, firstPartEnd), dataset.subList(firstPartEnd, dataSetSize));
    }

    public static <T> List<Pair<double[]>> buildExamplesSet(List<T> dataset, Function<T, double[]> inputBuilder, Function<T, double[]> outputBuilder) {
        return dataset.stream().map(o -> buildExample(o, inputBuilder, outputBuilder)).collect(Collectors.toList());
    }

    public static <T> Pair<double[]> buildExample(T object, Function<T, double[]> inputBuilder, Function<T, double[]> outputBuilder) {
        return new Pair<>(inputBuilder.apply(object), outputBuilder.apply(object));
    }

}