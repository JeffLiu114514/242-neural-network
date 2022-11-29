package learn.lc.core;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class LogisticClassifier extends LinearClassifier {

    private ArrayList<Double> accuracyList;

    public LogisticClassifier(double[] weights) {
        super(weights);
        this.accuracyList = new ArrayList<>();
    }

    public LogisticClassifier(int ninputs) {
        super(ninputs);
        this.accuracyList = new ArrayList<>();
    }

    /**
     * A LogisticClassifier uses the logistic update rule
     * (AIMA Eq. 18.8): w_i \leftarrow w_i+\alpha(y-h_w(x)) \times h_w(x)(1-h_w(x)) \times x_i
     */
    public void update(double[] x, double y, double alpha) {
        // This must be implemented by you
        double h_w = threshold(VectorOps.dot(this.weights, x));
        for (int i = 0; i < x.length; i++)
            this.weights[i] = this.weights[i] + ((alpha * (y - h_w)) * h_w * (1 - h_w) * x[i]);
    }

    /**
     * A LogisticClassifier uses a 0/1 sigmoid threshold at z=0.
     */
    public double threshold(double z) {
        // This must be implemented by you
        return 1.0 / (1.0 + Math.exp(-z));
    }

    @Override
    protected void trainingReport(List<Example> examples, int stepnum, int nsteps) {
        double a = (1.0-squaredErrorPerSample(examples));
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
