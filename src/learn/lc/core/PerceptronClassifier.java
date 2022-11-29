package learn.lc.core;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class PerceptronClassifier extends LinearClassifier {

    private ArrayList<Double> accuracyList;

    public PerceptronClassifier(double[] weights) {
        super(weights);
        this.accuracyList = new ArrayList<>();
    }

    public PerceptronClassifier(int ninputs) {
        super(ninputs);
        this.accuracyList = new ArrayList<>();
    }

    /**
     * A PerceptronClassifier uses the perceptron learning rule
     * (AIMA Eq. 18.7): w_i \leftarrow w_i+\alpha(y-h_w(x)) \times x_i
     */
    public void update(double[] x, double y, double alpha) {
        // This must be implemented by you
        double h_w = threshold(VectorOps.dot(this.weights, x));
        for (int i = 0; i < x.length; i++)
            this.weights[i] = this.weights[i] + ((alpha * (y - h_w)) * x[i]);
    }

    /**
     * A PerceptronClassifier uses a hard 0/1 threshold.
     */
    public double threshold(double z) {
        // This must be implemented by you
        if (z >= 0)
            return 1;
        else
            return 0;
    }

    @Override
    protected void trainingReport(List<Example> examples, int stepnum, int nsteps) {
        double a = accuracy(examples);
        System.out.println(stepnum + "," + a);
        accuracyList.add(a);
    }

    public void fileWriter(String fileName) throws IOException {
        FileWriter fileWriter = new FileWriter("src/learn/lc/examples/" + fileName);
        fileWriter.write("Num" + "," + "Prop" + "\n");
        for(int i = 0; i < accuracyList.size(); i++){
            String step = String.valueOf(i + 1);
            fileWriter.write(step + "," + accuracyList.get(i) + "\n");
        }
        fileWriter.close();
    }
}
