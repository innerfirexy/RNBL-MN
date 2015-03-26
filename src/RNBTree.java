/* Thanks to http://www.cs.jhu.edu/~joanne/cs226/notes/TreeRoot.java*/

import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.bayes.NaiveBayesMultinomial;

import java.util.*;

public class RNBTree {
	private Instances data;	
	private RNBTree parent;
	private LinkedList<RNBTree> children;
	NaiveBayesMultinomial nbm;
	
	public RNBTree(Instances d) {
		this(d, null);
	}
	
	public RNBTree(Instances d, RNBTree p) {
		this.data = d;
		this.parent = p;
		this.children = new LinkedList<RNBTree>();
		nbm = new NaiveBayesMultinomial();
		try {
			nbm.buildClassifier(data);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public RNBTree parent() {
		return parent;
	}
	
	public LinkedList<RNBTree> children() {
		return children;
	}
	
	public boolean isLeaf() {
		return children.size() == 0;
	}
	
	public void addChild(Instances d) {
		addChild(new RNBTree(d));
	}
	
	public void addChild(RNBTree subtree) {
		children.addLast(subtree);
		subtree.parent = this;
	}
	
	public double computeCMDL() {
		double sizeD = this.data.numInstances();
		double numAttr = this.data.numAttributes();
		double numClass = this.data.numClasses();
		double numNode = this.numNode();
		double size_h = (numClass + numClass * numAttr) * numNode;
		double cmdl = this.computeCLL() - 0.5 * Math.log(sizeD) * size_h;
		return cmdl;
	}
	
	public double computeCLL() {
		if (this.isLeaf()) {
			return CLL(this.data);
		}
		else {
			double sum = 0.0;
			for (RNBTree child : children) {
				sum += child.computeCLL();
			}
			return sum;
		}
	}
	
	public double CLL(Instances d) {
		double cllSum = 0.0;
		try {
			for (int i = 0; i < d.numInstances(); i++) {
				Instance ins = d.instance(i);
				double[] p = nbm.distributionForInstance(ins);
				double classVal = ins.classValue();
				if (classVal == 0.0) {
					cllSum += Math.log(p[0]);
				} else {
					cllSum += Math.log(p[1]);
				}
//				for (int j = 0; j < d.numClasses(); j++) {
//					cllSum += Math.log(p[j]);
//				}
//				if (p[0] > p[1]) {
//					cllSum += Math.log(p[0]);
//				} else {
//					cllSum += Math.log(p[1]);
//				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return d.numInstances() * cllSum;
	}
	
	public void splitNode() {
		ArrayList<Instances> data_pred = new ArrayList<Instances>();
		for (int k = 0; k < this.data.numClasses(); k++) {
			data_pred.add(new Instances(this.data, 0));
		}
		try {
			for (int i = 0; i < this.data.numInstances(); i++) {
				Instance ins = this.data.instance(i);
				double[] p = nbm.distributionForInstance(ins);
				int idx = this.getClassIndex(p);
				data_pred.get(idx).add(ins);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		for (Instances d : data_pred) {
			this.addChild(d);
		}
	}
	
	public void revokeLastSplit() {
		LinkedList<RNBTree> nodeList = this.leafBFS();
		RNBTree parentNode = nodeList.getLast().parent();
		parentNode.children.clear();
	}
	
	private int getClassIndex(double[] p) {
		int index = 0;
		double p_max = 0.0;
		for (int i = 0; i < p.length; i++) {
			if (p[i] > p_max) {
				index = i;
				p_max = p[i];
			}
		}
		return index;
	}
	
	public void stepGrow() {
		LinkedList<RNBTree> nodeList = this.leafBFS();
		RNBTree node = nodeList.getFirst();
		node.splitNode();
	}
	
	public int numNode() {
		LinkedList<RNBTree> nodeList = this.nodeBFS();
		return nodeList.size();
	}
	
	public int numLeaf() {
		LinkedList<RNBTree> leafList = this.leafBFS();
		return leafList.size();
	}
	
	public LinkedList<RNBTree> leafBFS() {
		LinkedList<RNBTree> queue = new LinkedList<RNBTree>();
		LinkedList<RNBTree> nodeList = new LinkedList<RNBTree>();
		queue.addLast(this);
		RNBTree node;
		while (!queue.isEmpty()) {
			node = queue.removeFirst();
			if (node.isLeaf()) {
				nodeList.addLast(node);
			}
			else {
				for (RNBTree child : node.children()) {
					queue.addLast(child);
				}
			}
		}
		return nodeList;
	}
	
	public LinkedList<RNBTree> nodeBFS() {
		LinkedList<RNBTree> queue = new LinkedList<RNBTree>();
		LinkedList<RNBTree> nodeList = new LinkedList<RNBTree>();
		queue.addLast(this);
		RNBTree node;
		while (!queue.isEmpty()) {
			node = queue.removeFirst();
			nodeList.addLast(node);
			for (RNBTree child : node.children()) {
				queue.addLast(child);
			}
		}
		return nodeList;
	}
	
	public ArrayList<Double> evaluate() {
		ArrayList<Double> results = new ArrayList();
		LinkedList<RNBTree> leaves = this.leafBFS();
		double true_positive = 0;
		double false_positive = 0;
		double true_negative = 0;
		double false_negative = 0;
		for (RNBTree leaf : leaves) {
			RNBTree parent = leaf.parent();
			if (parent.children().indexOf(leaf) == 0) { // if leaf is the left node
				for (int i = 0; i < leaf.data.numInstances(); i++) {
					Instance ins = leaf.data.instance(i);
					double label = ins.classValue();
					if (label == 0.0) {
						true_negative += 1;
					} else {
						false_negative += 1;
					}
				}
			}
			else { // if leaf is the right node
				for (int i = 0; i < leaf.data.numInstances(); i++) {
					Instance ins = leaf.data.instance(i);
					double label = ins.classValue();
					if (label == 0.0) {
						false_positive += 1;
					} else {
						true_positive += 1;
					}
				}
			}
		}
		// calculate precision, recall, fmeasure, accuracy
		double precision = true_positive / (true_positive + false_positive);
		double recall = true_positive / (true_positive + false_negative);
		double fmeasure = 2 * precision * recall / (precision + recall);
		double accuracy = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative);
		results.add(precision);
		results.add(recall);
		results.add(fmeasure);
		results.add(accuracy);
		return results;
	}
	
	public double predInstance(Instance d) throws Exception {
		if (this.isLeaf()) {
			double label = this.nbm.classifyInstance(d);
			return label;
			
		}
		else {
			double label = this.nbm.classifyInstance(d);
			if (label == 0.0) {
				return this.children().get(0).nbm.classifyInstance(d);
			} else {
				return this.children().get(1).nbm.classifyInstance(d);
			}
		}
	}
}
