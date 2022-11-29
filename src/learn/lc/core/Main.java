package learn.lc.core;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        String filename = args[0];
                //"/Users/sauceeeeage/IdeaProjects/CSC242/Project4/src/learn/lc/examples/house-votes-84.data.num.txt";
        int nsteps = Integer.parseInt(args[1]);
        double alpha = Double.parseDouble(args[2]);
        String instruction = args[3];
        try {
            if (instruction.equals("Perceptron")) {
                perceptionTest(filename, nsteps, alpha);
            } else if (instruction.equals("Logistic")) {
                logisticTest(filename, nsteps, alpha);
            } else {
                System.out.println("Instruction not found.");
                System.exit(0);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void perceptionTest(String filename, int nsteps, double alpha) throws IOException {
        List<Example> examples = readEarthquakeDataFromFile(filename);
        int ninputs = examples.get(0).inputs.length;

        PerceptronClassifier classifier = new PerceptronClassifier(ninputs);

        if (alpha > 0) {
            classifier.train(examples, nsteps, alpha);
        } else {
            classifier.train(examples, nsteps, new DecayingLearningRateSchedule());
        }
        //classifier.fileWriter("perceptron_output.csv");
    }

    public static void logisticTest(String filename, int nsteps, double alpha) throws IOException {
        List<Example> examples = readEarthquakeDataFromFile(filename);
        int ninputs = examples.get(0).inputs.length;

        LogisticClassifier classifier = new LogisticClassifier(ninputs);

        if (alpha > 0) {
            classifier.train(examples, nsteps, alpha);
        } else {
            classifier.train(examples, nsteps, new DecayingLearningRateSchedule());
        }
        //classifier.fileWriter("logistic_output.csv");
    }

    public static List<Example> readEarthquakeDataFromFile(String filename) throws IOException {
        List<Example> examples = new ArrayList<Example>();
        Scanner in = new Scanner(new File(filename));
        while (in.hasNext()) {
            String[] nextline = in.nextLine().split(",");
            double[] input = new double[nextline.length];
            input[0] = 1.0;
            for (int i = 1; i < nextline.length; i++) {
                input[i] = Double.parseDouble(nextline[i - 1]);
            }

            examples.add(new Example(input, Double.parseDouble(nextline[nextline.length - 1])));
        }
        in.close();
        return examples;
    }
}
