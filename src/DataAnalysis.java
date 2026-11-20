import weka.core.Attribute;
import weka.core.ContingencyTables;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.Statistics;
import weka.core.Utils;
import weka.core.converters.ArffSaver;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.TreeSet;

/**
 * IT160IU Data Mining - Fall 2026
 * Step 1 (Java part): Data Analysis + ARFF export
 * Person 1
 *
 * Per the instructor's guidance: Python/pandas is used only for the data
 * CLEANING step and to export a clean CSV (see Preprocessing.ipynb, which
 * writes code/dataset/cleaned_heart_disease.csv). This Java program picks
 * up from that clean CSV and performs, entirely in Java/Weka:
 *
 *   1. Data Analysis:
 *        - target distribution
 *        - outlier count per numeric attribute (IQR rule)
 *        - multicollinearity check among numeric attributes (correlation)
 *        - numeric attributes vs. target (Pearson/point-biserial correlation + t-test)
 *        - categorical attributes vs. target (chi-square test of independence)
 *   2. A stratified 80/20 train/test split (Weka's own stratify/trainCV/testCV).
 *   3. ARFF export (train.arff, test.arff) for Person 2 (J48) and Person 3 (Random Forest).
 *
 * Usage:
 *   java -cp "../lib/weka.jar:." DataAnalysis [cleanedCsvPath] [outDir]
 *
 * Both arguments are OPTIONAL and accept relative or absolute paths. Defaults
 * assume this program is run from code/src/, reading code/dataset/cleaned_heart_disease.csv
 * and writing train.arff / test.arff into code/dataset/.
 */
public class DataAnalysis {

    static final String[] NUMERIC_COLS = {
            "Age", "Blood Pressure", "Cholesterol Level", "BMI", "Sleep Hours",
            "Triglyceride Level", "Fasting Blood Sugar", "CRP Level", "Homocysteine Level"
    };

    static final String[] CATEGORICAL_COLS = {
            "Gender", "Exercise Habits", "Smoking", "Family Heart Disease", "Diabetes",
            "High Blood Pressure", "Low HDL Cholesterol", "High LDL Cholesterol",
            "Alcohol Consumption", "Stress Level", "Sugar Consumption"
    };

    static final String TARGET_COL = "Heart Disease Status";

    public static void main(String[] args) throws Exception {
        String csvPath = args.length > 0 ? args[0] : "../dataset/cleaned_heart_disease.csv";
        String outDir  = args.length > 1 ? args[1] : "../dataset";

        System.out.println("=====================================================");
        System.out.println(" IT160IU - Step 1 (Java): Data Analysis + ARFF Export");
        System.out.println("=====================================================");
        System.out.println("Input CSV  : " + csvPath);
        System.out.println("Output dir : " + outDir);
        System.out.println();

        // ---- 1. Read the cleaned CSV produced by Preprocessing.ipynb ----
        List<String> header = new ArrayList<>();
        List<String[]> rows = readCsv(csvPath, header);
        Map<String, Integer> colIndex = new HashMap<>();
        for (int i = 0; i < header.size(); i++) {
            colIndex.put(header.get(i), i);
        }
        System.out.println("Loaded " + rows.size() + " rows, " + header.size() + " columns");
        System.out.println();

        // ---- 2. Data Analysis ----
        runDataAnalysis(rows, colIndex);

        // ---- 3. Build a Weka Instances object from the raw rows ----
        Instances data = buildInstances(rows, colIndex);

        // ---- 4. Stratified 80/20 train/test split (seed fixed for reproducibility) ----
        Instances[] split = stratifiedSplit80_20(data, 42);
        Instances trainData = split[0];
        Instances testData = split[1];

        System.out.println("---- Train/Test Split (stratified 80/20) ----");
        System.out.println("Train: " + trainData.numInstances() + " instances");
        System.out.println("Test : " + testData.numInstances() + " instances");
        printClassRatio("Full dataset", data);
        printClassRatio("Train set", trainData);
        printClassRatio("Test set", testData);
        System.out.println();

        // ---- 5. Export ARFF ----
        new File(outDir).mkdirs();
        saveArff(trainData, "heart_disease_train", outDir + File.separator + "train.arff");
        saveArff(testData, "heart_disease_test", outDir + File.separator + "test.arff");
    }

    // =================================================================
    // CSV reading (plain split on comma: this dataset has no embedded
    // commas or quoted fields in any column)
    // =================================================================
    static List<String[]> readCsv(String path, List<String> headerOut) throws IOException {
        List<String[]> rows = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line = br.readLine();
            if (line == null) {
                throw new IOException("Empty CSV file: " + path);
            }
            headerOut.addAll(Arrays.asList(line.split(",", -1)));
            while ((line = br.readLine()) != null) {
                if (line.isEmpty()) continue;
                rows.add(line.split(",", -1));
            }
        }
        return rows;
    }

    // =================================================================
    // Data Analysis
    // =================================================================
    static void runDataAnalysis(List<String[]> rows, Map<String, Integer> colIndex) {
        int n = rows.size();
        int targetIdx = colIndex.get(TARGET_COL);

        double[] targetBin = new double[n];
        int yesCount = 0;
        for (int i = 0; i < n; i++) {
            boolean isYes = rows.get(i)[targetIdx].trim().equalsIgnoreCase("Yes");
            targetBin[i] = isYes ? 1.0 : 0.0;
            if (isYes) yesCount++;
        }

        System.out.println("---- 1. Target Variable Distribution ----");
        System.out.printf(Locale.US, "No  : %d (%.2f%%)%n", n - yesCount, 100.0 * (n - yesCount) / n);
        System.out.printf(Locale.US, "Yes : %d (%.2f%%)%n", yesCount, 100.0 * yesCount / n);
        System.out.println();

        // ---- Load numeric columns into arrays once, reused below ----
        Map<String, double[]> numericValues = new LinkedHashMap<>();
        for (String col : NUMERIC_COLS) {
            double[] values = new double[n];
            int idx = colIndex.get(col);
            for (int i = 0; i < n; i++) {
                values[i] = Double.parseDouble(rows.get(i)[idx]);
            }
            numericValues.put(col, values);
        }

        System.out.println("---- 2. Outlier Detection (IQR rule) on Numeric Attributes ----");
        for (String col : NUMERIC_COLS) {
            int nOutliers = countOutliersIQR(numericValues.get(col));
            System.out.printf(Locale.US, "%-22s outliers = %d (%.2f%%)%n",
                    col, nOutliers, 100.0 * nOutliers / n);
        }
        System.out.println();

        System.out.println("---- 3. Multicollinearity Check (numeric-numeric correlation) ----");
        double maxAbsCorr = 0.0;
        String maxPair = "";
        String[] numCols = NUMERIC_COLS;
        for (int a = 0; a < numCols.length; a++) {
            for (int b = a + 1; b < numCols.length; b++) {
                double r = Utils.correlation(numericValues.get(numCols[a]), numericValues.get(numCols[b]), n);
                if (Math.abs(r) > maxAbsCorr) {
                    maxAbsCorr = Math.abs(r);
                    maxPair = numCols[a] + " vs " + numCols[b];
                }
            }
        }
        System.out.printf(Locale.US, "Max pairwise |correlation| among numeric attributes: %.4f  (%s)%n",
                maxAbsCorr, maxPair);
        System.out.println();

        System.out.println("---- 4. Numeric Attributes vs. Target (correlation + t-test) ----");
        System.out.printf(Locale.US, "%-22s %10s %10s %10s%n", "Attribute", "r", "t", "p-value");
        for (String col : NUMERIC_COLS) {
            double[] values = numericValues.get(col);
            double r = Utils.correlation(values, targetBin, n);
            int df = n - 2;
            // guard against r = +-1 (not expected here, but avoids division by zero)
            double denom = Math.max(1e-12, 1 - r * r);
            double tStat = r * Math.sqrt(df / denom);
            // two-sided t-test p-value via the identity t^2 ~ F(1, df)
            double pValue = Statistics.FProbability(tStat * tStat, 1, df);
            System.out.printf(Locale.US, "%-22s %+10.4f %+10.3f %10.4f%s%n",
                    col, r, tStat, pValue, pValue < 0.05 ? "  *" : "");
        }
        System.out.println("(* p < 0.05)");
        System.out.println();

        System.out.println("---- 5. Categorical Attributes vs. Target (Chi-square test) ----");
        System.out.printf(Locale.US, "%-22s %10s %6s %10s%n", "Attribute", "chi2", "df", "p-value");
        for (String col : CATEGORICAL_COLS) {
            int idx = colIndex.get(col);
            LinkedHashMap<String, int[]> table = new LinkedHashMap<>(); // category -> [noCount, yesCount]
            for (int i = 0; i < n; i++) {
                String category = rows.get(i)[idx];
                int[] counts = table.computeIfAbsent(category, k -> new int[2]);
                if (targetBin[i] == 1.0) {
                    counts[1]++;
                } else {
                    counts[0]++;
                }
            }
            double[][] matrix = new double[table.size()][2];
            int r = 0;
            for (int[] counts : table.values()) {
                matrix[r][0] = counts[0];
                matrix[r][1] = counts[1];
                r++;
            }
            double chi2 = ContingencyTables.chiSquared(matrix, false);
            int df = (table.size() - 1) * (2 - 1);
            double pValue = Statistics.chiSquaredProbability(chi2, df);
            System.out.printf(Locale.US, "%-22s %10.3f %6d %10.4f%s%n",
                    col, chi2, df, pValue, pValue < 0.05 ? "  *" : "");
        }
        System.out.println("(* p < 0.05)");
        System.out.println();
    }

    static int countOutliersIQR(double[] values) {
        double[] sorted = values.clone();
        Arrays.sort(sorted);
        double q1 = percentile(sorted, 0.25);
        double q3 = percentile(sorted, 0.75);
        double iqr = q3 - q1;
        double lower = q1 - 1.5 * iqr;
        double upper = q3 + 1.5 * iqr;
        int count = 0;
        for (double v : values) {
            if (v < lower || v > upper) count++;
        }
        return count;
    }

    /** Linear-interpolation percentile, matching pandas/numpy's default. */
    static double percentile(double[] sortedValues, double p) {
        double idx = p * (sortedValues.length - 1);
        int lo = (int) Math.floor(idx);
        int hi = (int) Math.ceil(idx);
        if (lo == hi) return sortedValues[lo];
        double frac = idx - lo;
        return sortedValues[lo] + frac * (sortedValues[hi] - sortedValues[lo]);
    }

    // =================================================================
    // Build a Weka Instances object (attribute names: spaces -> underscores,
    // so Person 2 / Person 3's Java code never needs to quote attribute names)
    // =================================================================
    static Instances buildInstances(List<String[]> rows, Map<String, Integer> colIndex) {
        Map<String, List<String>> nominalDomains = new LinkedHashMap<>();
        for (String col : CATEGORICAL_COLS) {
            nominalDomains.put(col, sortedDistinctValues(rows, colIndex.get(col)));
        }
        nominalDomains.put(TARGET_COL, sortedDistinctValues(rows, colIndex.get(TARGET_COL)));

        ArrayList<Attribute> attributes = new ArrayList<>();
        for (String col : NUMERIC_COLS) {
            attributes.add(new Attribute(safeName(col)));
        }
        for (String col : CATEGORICAL_COLS) {
            attributes.add(new Attribute(safeName(col), nominalDomains.get(col)));
        }
        attributes.add(new Attribute(safeName(TARGET_COL), nominalDomains.get(TARGET_COL)));

        Instances data = new Instances("heart_disease", attributes, rows.size());
        data.setClassIndex(attributes.size() - 1);

        for (String[] row : rows) {
            double[] values = new double[attributes.size()];
            int attrIdx = 0;
            for (String col : NUMERIC_COLS) {
                values[attrIdx] = Double.parseDouble(row[colIndex.get(col)]);
                attrIdx++;
            }
            for (String col : CATEGORICAL_COLS) {
                values[attrIdx] = attributes.get(attrIdx).indexOfValue(row[colIndex.get(col)]);
                attrIdx++;
            }
            values[attrIdx] = attributes.get(attrIdx).indexOfValue(row[colIndex.get(TARGET_COL)]);
            data.add(new DenseInstance(1.0, values));
        }
        return data;
    }

    static String safeName(String col) {
        return col.replace(' ', '_');
    }

    static List<String> sortedDistinctValues(List<String[]> rows, int idx) {
        TreeSet<String> set = new TreeSet<>();
        for (String[] row : rows) {
            set.add(row[idx]);
        }
        return new ArrayList<>(set);
    }

    // =================================================================
    // Stratified 80/20 split using Weka's own stratify/trainCV/testCV
    // (5 folds, stratified -> fold 0 as test = exactly 20%, folds 1-4 as train = 80%)
    // =================================================================
    static Instances[] stratifiedSplit80_20(Instances data, int seed) {
        Instances shuffled = new Instances(data);
        shuffled.randomize(new Random(seed));
        shuffled.stratify(5);
        Instances train = shuffled.trainCV(5, 0);
        Instances test = shuffled.testCV(5, 0);
        return new Instances[]{train, test};
    }

    static void printClassRatio(String label, Instances data) {
        int[] counts = new int[data.classAttribute().numValues()];
        for (int i = 0; i < data.numInstances(); i++) {
            counts[(int) data.instance(i).classValue()]++;
        }
        StringBuilder sb = new StringBuilder(label + ": ");
        for (int i = 0; i < counts.length; i++) {
            sb.append(data.classAttribute().value(i)).append("=")
                    .append(String.format(Locale.US, "%.2f%%", 100.0 * counts[i] / data.numInstances()))
                    .append("  ");
        }
        System.out.println(sb);
    }

    // =================================================================
    // ARFF export
    // =================================================================
    static void saveArff(Instances data, String relationName, String outPath) throws IOException {
        data.setRelationName(relationName);
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(outPath));
        saver.writeBatch();
        System.out.println("Saved: " + outPath + "  (" + data.numInstances() + " instances)");
    }
}
