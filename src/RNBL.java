import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.CSVLoader;
import weka.core.converters.ArffSaver;
import weka.classifiers.bayes.NaiveBayesMultinomial;

import java.util.*;
import java.io.*;

public class RNBL {
	
	/**
	 * Data sets
	 * */
	Instances train;
	Instances test;
	
	/**
	 * Tree object
	 * */
	RNBTree tree;
	
	/**
	 * Load data
	 * @param head The head of data file name
	 * @param dir The directory to the data files
	 * */
	public void loadData(String dir, String head) {
		try {
			DataSource source = new DataSource(dir + head + "_vector_train.arff");
			train = source.getDataSet();
			train.setClassIndex(train.numAttributes()-1);
			source = new DataSource(dir + head + "_vector_test.arff");
			test = source.getDataSet();
			test.setClassIndex(test.numAttributes()-1);
		}
		catch (Exception e) {
			System.out.println("Error in loading data!");
		}	
	}
	
	/**
	 * Convert csv to arff
	 * @throws IOException 
	 * */
	public void convertData(String csv, String arff) throws IOException {
		// load csv
		System.out.println(csv);
		CSVLoader loader = new CSVLoader();
		loader.setSource(new File(csv));
		Instances data = loader.getDataSet();
		// save arff
		ArffSaver saver = new ArffSaver();
		saver.setInstances(data);
		saver.setFile(new File(arff));
		saver.setDestination(new File(arff));
		saver.writeBatch();
	}
	
	/**
	 * Split a full arff file into train set and test set
	 * @throws Exception 
	 * */
	public void splitData(String input, int trainSize, int testSize) throws Exception {
		// load data
		DataSource source = new DataSource(input);
		Instances data = source.getDataSet();
		ArffSaver saver = new ArffSaver();
		// get train
		Instances train = new Instances(data, 0, trainSize);
		saver.setInstances(train);
		saver.setFile(new File(input.substring(0, input.length()-5) + "_train.arff"));
		saver.writeBatch();
		// get test
		Instances test = new Instances(data, trainSize, testSize);
		saver.setInstances(test);
		saver.setFile(new File(input.substring(0, input.length()-5) + "_test.arff"));
		saver.writeBatch();
	}
	
	/**
	 * Compute dictionaries
	 * */
	public HashMap<String, Double> computeDict(Instances data) {
		HashMap<String, Double> hm_true = new HashMap();
		HashMap<String, Double> hm_false = new HashMap();
		double vocab_size = data.numAttributes();
		
		for (int i = 0; i < data.numInstances(); i++) {
			double c = data.instance(i).classValue();
			if (c == 1.0) { // class == true
				for (int j = 0; j < data.numAttributes()-1; j++) {
					double value = data.instance(i).value(j);
					if (value != 0) {
						String attr = data.attribute(j).name();
						if (hm_true.containsKey(attr)) {
							double val = hm_true.get(attr);
							val += value;
						}
						else {
							hm_true.put(attr, value);
						}
					}
				}
			}
			else { // class == false
				for (int j = 0; j < data.numInstances(); j++) {
					double value = data.instance(i).value(j);
					if (value != 0) {
						String attr = data.attribute(j).name();
						if (hm_false.containsKey(attr)) {
							double val = hm_true.get(attr);
							val += value;
						}
						else {
							hm_false.put(attr, value);
						}
					}
				}
			}
		}
		// update the two dicts
		double sum = 0.0;
		for (double val : hm_true.values()) {
			sum += val;
		}
		for (String key : hm_true.keySet()) {
			hm_true.put(key, hm_true.get(key) / sum);
		}
		sum = 0.0;
		for (double val : hm_false.values()) {
			sum += val;
		}
		
		return hm_true;
	}
	
	
	/**
	 * main
	 * @throws Exception 
	 * */
	public static void main(String[] args) throws Exception {
		RNBL exp = new RNBL();
		
		String work_dir = "/Users/yangxu/Google Drive/IST 4th semester courses/IST597K/lab2/";
		String csv_dir = work_dir + "csv/";
		String arff_dir = work_dir + "arff/";
		String[] file_header = {"acq", "corn", "crude", "earn", "grain", "interest", "money-fx", "ship", "trade", "wheat"};
		int[] train_size = {9542, 9498, 9458, 9544, 9519, 9508, 9529, 9451, 9532, 9521};
		int[] test_size = {3279, 3253, 3249, 3282, 3263, 3272, 3277, 3234, 3275, 3263};
		
		// convert csv files to arff files
//		for (String fh : file_header) {
//			String csv_file = csv_dir + fh + "_full_vector.csv";
//			String arff_file = arff_dir + fh + "_vector.arff";
//			System.out.println(csv_file);
//			exp.convertData(csv_file, arff_file);
//		}
		
		// split arff files
//		for (int i = 0; i < file_header.length; i++) {
//			String input_file = arff_dir + file_header[i] + "_vector.arff";
//			System.out.println(input_file);
//			exp.splitData(input_file, train_size[i], test_size[i]);
//		}	
		
		// experiment
		for (String head : file_header) {
			exp.loadData(arff_dir, head);
			exp.tree = new RNBTree(exp.train);
			double prevCMDL = Double.NEGATIVE_INFINITY;
			double CMDL = exp.tree.computeCMDL();
			
//			System.out.println("Experiment with " + head);
			while (prevCMDL <= CMDL) {
				prevCMDL = CMDL;
				exp.tree.stepGrow();
				CMDL = exp.tree.computeCMDL();
//				System.out.println("Tree size: " + String.valueOf(exp.tree.numNode()) + ", CMDL: " + String.valueOf(CMDL));
			}
			exp.tree.revokeLastSplit();
			CMDL = exp.tree.computeCMDL();
			System.out.println("Tree size: " + String.valueOf(exp.tree.numNode()) + ", CMDL: " + String.valueOf(CMDL));
			
			ArrayList<Double> res = exp.tree.evaluate();
			System.out.println(head + " accuracy: " + String.valueOf(res.get(3)));
		}
	}

}
