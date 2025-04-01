// Implementation of the KMeans Algorithm
// reference: https://github.com/marcoscastro/kmeans

//change point and cluster to use a hashmap (unordered map) and maybe the vector clusters in kmeans

#include <sstream>
//#include <tbb/parallel_for.h>
#include <iostream>
#include <unordered_set>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <mutex>
using namespace std;

//kmeansRun(points, total_points, numDimensions, K, max_iterations);
void kmeansRun(std::vector<std::vector<double>> points, int totalPoints, int numDimensions, int K, int max_iterations){
    auto begin = chrono::high_resolution_clock::now();
	if(K > total_points)
		return;

    int pointLengthNeeded = numDimensions * totalPoints;
    double pointsArray[pointLengthNeeded];

    for(int i = 0; i < totalPoints; i++){
        for(int j = 0; j < numDimensions; j++){
            pointsArray[i * numDimensions + j] = points[i][j];
        }
    }

	vector<int> prohibited_indexes;

    int clusterLengthNeeded = numDimensions * K;
    double clustersArray[clusterLengthNeeded];

	// choose K distinct values for the centers of the clusters
	for(int i = 0; i < K; i++){
		while(true){
			int index_point = rand() % total_points;
			if(find(prohibited_indexes.begin(), prohibited_indexes.end(),index_point) == prohibited_indexes.end()){
				prohibited_indexes.push_back(index_point);
                for(int j = 0; j < numDimensions; j++){
                    clustersArray[i * numDimensions + j] = pointsArray[index_point * numDimensions + j];
                }
				break;
			}
		}
	}

    
}

int main()//int argc, char *argv[])
{
	//if(argc == 0){
	//	cout << "wow" << endl;
	//}
	srand(0);//atoi(argv[1]));

	std::ifstream dataFile("datasets/Dry_Bean_Dataset.txt");
	int total_points, numDimensions, K, max_iterations, has_name;

	dataFile >> total_points >> numDimensions >> K >> max_iterations >> has_name;

	std::vector<std::vector<double>> points;
	string point_name;

	for(int i = 0; i < total_points; i++){
		vector<double> values;

		for(int j = 0; j < total_values; j++){
			double value;
			dataFile >> value;
			values.push_back(value);
		}

		if(has_name){
			dataFile >> point_name;
			points.push_back(values);
		} else {
			points.push_back(values);
		}
	}
	//printf("%f\n", points[13610][15]);
	/*
	ifstream dataFile("fifa_final_no_id_dataset.csv");
    if (!dataFile) {
        cerr << "Error opening file!" << endl;
        return 1;
    }

    int total_points, total_values, K, max_iterations, has_name;
    string line;

    // Read the first line (metadata)
    getline(dataFile, line);
    stringstream ss(line);
    string temp;

    getline(ss, temp, ','); total_points = stoi(temp);
    getline(ss, temp, ','); total_values = stoi(temp);
    getline(ss, temp, ','); K = stoi(temp);
    getline(ss, temp, ','); max_iterations = stoi(temp);
    getline(ss, temp, ','); has_name = stoi(temp);

    vector<Point> points;
    
    for (int i = 0; i < total_points; i++) {
        if (!getline(dataFile, line)) break;
        stringstream ss(line);
        vector<double> values;
        
        for (int j = 0; j < total_values; j++) {
            if (!getline(ss, temp, ',')) break;
            values.push_back(stod(temp));
        }

        string point_name = "";
        if (has_name && getline(ss, point_name, ',')) {
            points.emplace_back(i, values, point_name);
        } else {
            points.emplace_back(i, values);
        }
    }*/

	//KMeans kmeans(K, total_points, total_values, max_iterations);
	//kmeans.run(points);

	kmeansRun(points, total_points, numDimensions, K, max_iterations);

	return 0;
}