package me.hivaze.utils;

import java.io.*;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;

public class SimpleCSV {

    public static void writeToFile(Path path, List<String> headers, List<List<Object>> data) throws IOException {
        File file = path.toFile();
        try (PrintWriter fw = new PrintWriter(file)) {
            fw.println(String.join(",", headers));
            data.forEach(line -> fw.println(line.stream().map(Object::toString).collect(Collectors.joining(","))));
        }
    }

    public static HashMap<String, List<String>> readFileWithHeaders(Path path) throws IOException {
        File file = path.toFile();
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            HashMap<String, List<String>> result = new HashMap<>();
            String line, headers[] = br.readLine().split(",");
            for (String header : headers) result.put(header, new ArrayList<>());
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                for (int i = 0; i < values.length; i++) {
                    result.get(headers[i]).add(values[i]);
                }
            }
            return result;
        }
    }

    public static List<List<String>> readFileWithoutHeaders(Path path) throws IOException {
        File file = path.toFile();
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            List<List<String>> result = new ArrayList<>();
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                result.add(Arrays.asList(values));
            }
            return result;
        }
    }

}