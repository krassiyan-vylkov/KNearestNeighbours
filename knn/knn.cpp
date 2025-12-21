#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <random>

struct Iris {
	std::vector<double> attributes;
	std::string irisClass;

	Iris(std::vector<double> attributes, std::string irisClass) : attributes(attributes), irisClass(irisClass)
	{

	}
};

double distanceBetween(const std::vector<double> trainAttributes ,const std::vector<double> testAttributes) {
	double distance = 0;
	for (int i = 0; i < trainAttributes.size(); i++) {
		distance += pow((trainAttributes[i] - testAttributes[i]), 2);
	}

	return sqrt(distance);
}

std::string knnPredict(std::vector<Iris>& trainSet, std::vector<double> testAttributes, std::vector<std::string>& classes ,int k) {
	std::vector<std::pair<std::string, double>> distances;
	for (int i = 0; i < trainSet.size(); i++) {
		double distance = distanceBetween(trainSet[i].attributes, testAttributes);
		distances.push_back({trainSet[i].irisClass, distance});
	}

	//insertion sort
	for (int i = 1; i < distances.size(); i++) {
		std::pair<std::string, double> temp = distances[i];
		int j = i - 1;
		while (j >= 0 && distances[j].second > temp.second) {
			distances[j + 1] = distances[j];
			j = j - 1;
		}
		distances[j + 1] = temp;
	}

	std::vector<int> classInstances(classes.size(), 0);
	for (int i = 0; i < k; i++) {
		for (int j = 0; j < classes.size(); j++) {
			if (distances[i].first == classes[j]) {
				classInstances[j]++;
			}
		}
	}

	int currMaxIndex = 0;
	for (int i = 1; i < classInstances.size(); i++) {
		if (classInstances[i] > classInstances[currMaxIndex]) {
			currMaxIndex = i;
		}
	}

	return classes[currMaxIndex];
}

double testAccuracy(std::vector<Iris>& trainSet, std::vector<Iris>& testSet, std::vector<std::string>& classes, int k) {
	int testEntries = testSet.size();
	int correctPredictions = 0;

	for (int i = 0; i < testSet.size(); i++) {
		std::string predictedClass = knnPredict(trainSet, testSet[i].attributes, classes, k);
		if (predictedClass == testSet[i].irisClass) {
			correctPredictions++;
		}
	}

	double accuracy = (correctPredictions / (double)testEntries) * 100;
	return accuracy;
}

void nFoldValidation(int numberOfFolds, std::vector<Iris>& trainSet, std::vector<std::string>& classes, int k, int numberOfClasses, int entriesPerClass) {
	int foldSizePerClass = entriesPerClass / numberOfFolds;
	std::vector<double> accuraciesPerFold;

	for (int currentFold = 0; currentFold < numberOfFolds; currentFold++) {
		std::vector<Iris> foldTrainSet;
		std::vector<Iris> foldTestSet;
		for (int currClassIndex = 0; currClassIndex < numberOfClasses; currClassIndex++) {
			for (int irisIndex = currClassIndex * entriesPerClass; irisIndex < currentFold * foldSizePerClass + entriesPerClass * currClassIndex; irisIndex++) {
				foldTrainSet.push_back(trainSet[irisIndex]);
			}
			for (int irisIndex = currentFold * foldSizePerClass + entriesPerClass * currClassIndex; irisIndex < (currentFold + 1) * foldSizePerClass + entriesPerClass * currClassIndex; irisIndex++) {
				foldTestSet.push_back(trainSet[irisIndex]);
			}
			for (int irisIndex = (currentFold + 1) * foldSizePerClass + entriesPerClass * currClassIndex; irisIndex < entriesPerClass * (currClassIndex + 1); irisIndex++) {
				foldTrainSet.push_back(trainSet[irisIndex]);
			}
		}
		double currFoldAccuracy = testAccuracy(foldTrainSet, foldTestSet, classes, k);
		accuraciesPerFold.push_back(currFoldAccuracy);
	}

	double accuracySum = 0;
	std::cout << "2. "<< numberOfFolds <<"-Fold Cross-Validation Results:" << std::endl;
	std::cout << std::endl;
	for (int i = 0; i < accuraciesPerFold.size(); i++) {
		accuracySum += accuraciesPerFold[i];
		std::cout << "   Accuracy Fold " << i + 1 << ": " << accuraciesPerFold[i] << "%" << std::endl;
	}
	double averargeAccuracy = accuracySum / accuraciesPerFold.size();
	std::cout << std::endl;
	std::cout << "   Average accuracy: " << averargeAccuracy << "%" << std::endl;

	double deviationSum = 0;
	for (int i = 0; i < accuraciesPerFold.size(); i++) {
		deviationSum = pow(accuraciesPerFold[i] - averargeAccuracy, 2) + deviationSum;
	}

	double standardDeviation = sqrt(deviationSum/accuraciesPerFold.size());
	std::cout << "   Standard deviation: " << standardDeviation << "%" << std::endl;
}

void knn(int k, std::vector<Iris>& dataSet) {
	//shuffling the set while preserving the fact that they are sorted by their iris class
	//we use the fact that the data consists of 150 entries, 50 of each of 3 iris classes
	int numberOfClasses = 3;
	int entriesPerClass = 50;

	std::vector<std::string> classes;
	for (int i = 0; i < numberOfClasses; i++) {
		classes.push_back(dataSet[i * entriesPerClass].irisClass);
	}

	std::random_device rd;
	std::mt19937 g(rd());

	std::shuffle(dataSet.begin(),dataSet.begin()+entriesPerClass, g);
	std::shuffle(dataSet.begin() + entriesPerClass, dataSet.begin() + 2*entriesPerClass, g);
	std::shuffle(dataSet.begin() + 2*entriesPerClass, dataSet.end(), g);
	
	double testSetSizePercentage = 0.2;
	int numberOfFolds = 10;
	int testSetSizePerClass = entriesPerClass * testSetSizePercentage;
	int trainSetSizePerClass = entriesPerClass - testSetSizePerClass;

	std::vector<Iris> trainSet;
	std::vector<Iris> testSet;

	for(int i = 0; i < numberOfClasses; i++) {
		for (int j = 0; j < trainSetSizePerClass; j++) {
			trainSet.push_back(dataSet[i*entriesPerClass + j]);
		}
		for (int j = trainSetSizePerClass; j < entriesPerClass; j++) {
			testSet.push_back(dataSet[i * entriesPerClass + j]);
		}
	}
	double accuracy;
	accuracy = testAccuracy(trainSet,trainSet, classes, k);
	std::cout << "1. Train Set Accuracy:" << std::endl;
	std::cout <<"   Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
	std::cout << std::endl;

	nFoldValidation(numberOfFolds, trainSet, classes, k, numberOfClasses, trainSetSizePerClass);

	accuracy = testAccuracy(trainSet, testSet, classes, k);
	std::cout << std::endl;
	std::cout << "3. Test Set Accuracy:" << std::endl;
	std::cout << "   Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
}

int main()
{
	/*
	The iris.data file can be found at https://archive.ics.uci.edu/dataset/53/iris
	*/

	std::ifstream f("iris.data");
	if (!f.is_open()) {
		std::cerr << "Error opening iris.data file";
		return 1;
	}

	std::string str;
	std::vector<Iris> dataSet;
	while (std::getline(f, str)) {
		if (str.size() == 0) {
			continue;
		}
		std::stringstream input(str);
		std::string partOfInput;
		std::vector<double> irisAttributes;
		std::getline(input, partOfInput, ',');
		double sepalLength = stod(partOfInput);
		irisAttributes.push_back(sepalLength);

		std::getline(input, partOfInput, ',');
		double sepalWidth = stod(partOfInput);
		irisAttributes.push_back(sepalWidth);

		std::getline(input, partOfInput, ',');
		double petalLength = stod(partOfInput);
		irisAttributes.push_back(petalLength);

		std::getline(input, partOfInput, ',');
		double petalWidth = stod(partOfInput);
		irisAttributes.push_back(petalWidth);

		std::getline(input, partOfInput);

		Iris iris(irisAttributes, partOfInput);
		dataSet.push_back(iris);
	}

	int k;
	std::cout << "Enter K: ";
	std::cin >> k;

	knn(k,dataSet);

	return 0;
}
