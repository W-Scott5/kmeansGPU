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
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
using namespace std;

//findClusters<<<1000, 1024>>>(devicePoints, deviceClusters, returnClusterNum, numDimensions, K, totalPoints);

//literally have two arrays one wiht the sum and one with the actual value

__global__ void findClusters(double *devicePoints, double *deviceClusters, double *deviceClustersTotals, int *deviceClusterAssignments, int *clusterAmount, int numDimensions, int numClusters, int numPoints){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < numPoints){
        double sum = 0.0, min_dist;
        int id_cluster_center = 0;

        for(int i = 0; i < numDimensions; i++){
            sum += powf(deviceClusters[i] - devicePoints[idx * numDimensions + i] , 2);
        }

        min_dist = sum;

        for(int i = 1; i < numClusters; i++){
            double dist;
            sum = 0.0;

            for(int j = 0; j < numDimensions; j++){
                sum += powf(deviceClusters[i * numDimensions + j] - devicePoints[idx * numDimensions + j] , 2);
            }

            dist = sum;

            if(dist < min_dist){
                min_dist = dist;
                id_cluster_center = i;
            }
        }
        int id_old = deviceClusterAssignments[idx];
        if(id_old != id_cluster_center){
            for(int i = 0; i < numDimensions; i++){
                atomicAdd(&deviceClustersTotals[id_cluster_center * numDimensions + i], devicePoints[idx * numDimensions + i]);
            }
            atomicAdd(&clusterAmount[id_cluster_center], 1);

            deviceClusterAssignments[idx] = id_cluster_center;
            if(id_old != -1){
                for(int i = 0; i < numDimensions; i++){
                    atomicAdd(&deviceClustersTotals[id_old * numDimensions + i], (-1 * devicePoints[idx * numDimensions + i]));
                }
                atomicAdd(&clusterAmount[id_cluster_center], -1);
            }
            atomicExch(&clusterAmount[numClusters], 1);
            //printf("clustr change: %d\n", changedCluster[numClusters]);

            //atomicAdd(&data[0], 1.0f);
        }
    }

    __syncthreads();

    //future - make sure you check the number of threads is enough and make it numdimensions * numclusters if that is greater than num of points 

    int totalAmount = numDimensions * numClusters;
    if(idx < totalAmount){
        deviceClusters[idx] = deviceClustersTotals[idx] / clusterAmount[idx / numDimensions];// 6/5=1 17/5=
    }
    


}
//calculateClusterCenter<<<1, 4>>>(devicePoints, deviceClusterAssignments, deviceClusters, numDimensions, K, totalPoints);
/*
__global__ void calculateClusterCenter(double *devicePoints, int *deviceClusterAssignments, double *deviceClusters, int *clusterAmount, int numDimensions, int numClusters, int numPoints){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int counter = 0;
	
    if(idx < numClusters){
        if(changedCluster[idx] == 1){
            for(int i = 0; i < numPoints; i++){
                if(deviceClusterAssignments[i] == idx){
                    for(int j = 0; j < numDimensions; j++){
                        if(counter == 0){
                            deviceClusters[idx * numDimensions + j] = 0;
                        }
                        deviceClusters[idx * numDimensions + j] += devicePoints[i * numDimensions + j];
                    }
                    counter++;
                }
            }
            if(counter > 0){
                for(int i = 0; i < numDimensions; i++){
                    deviceClusters[idx * numDimensions + i] = deviceClusters[idx * numDimensions + i] / (counter * 1.0);
                }
            }
            changedCluster[idx] = 0;
        }
    }
    if(idx == 0){
        changedCluster[numClusters] = 0;
    }
}*/

//kmeansRun(points, total_points, numDimensions, K, max_iterations);
void kmeansRun(double* points, int totalPoints, int numDimensions, int K, int max_iterations){
    //auto begin = chrono::high_resolution_clock::now();
    auto begin = chrono::high_resolution_clock::now();
    auto end_phase1 = chrono::high_resolution_clock::now();
    auto end = chrono::high_resolution_clock::now();
	if(K > totalPoints)
		return;

    begin = chrono::high_resolution_clock::now();
    int pointsLengthNeeded = numDimensions * totalPoints;

    std::cout << "hello?" << std::endl;
    //double pointsArray[pointsLengthNeeded];

    //for(int i = 0; i < totalPoints; i++){
    //    for(int j = 0; j < numDimensions; j++){
    //        pointsArray[i * numDimensions + j] = points[i][j];
    //    }
    //}
	//for(int i = 0; i < pointsLengthNeeded; i++){
	//	printf("%f ", pointsArray[i]);
	//}

	vector<int> prohibited_indexes;

    int clustersLengthNeeded = numDimensions * K;
    double clustersArray[clustersLengthNeeded];
    double clustersArrayTotals[clustersLengthNeeded] = {0};
    int clustersAmountPoints[K+1] = {0};

	// choose K distinct values for the centers of the clusters
	for(int i = 0; i < K; i++){
		while(true){
			int index_point = rand() % totalPoints;
			if(find(prohibited_indexes.begin(), prohibited_indexes.end(),index_point) == prohibited_indexes.end()){
				prohibited_indexes.push_back(index_point);
                for(int j = 0; j < numDimensions; j++){
                    clustersArray[i * numDimensions + j] = points[index_point * numDimensions + j];
                    clustersArrayTotals[i * numDimensions + j] = points[index_point * numDimensions + j];
                    clustersAmountPoints[i] = 1;
                }
				break;
			}
		}
	}
	//printf("cluster 0: %d\n", prohibited_indexes[0]);
	//printf("cluster 1: %d\n", prohibited_indexes[1]);

    //printf("what is happening %d", 1);
    std::cout << "hello?" << std::endl;

	int iter = 1;

    //int pointsLengthNeeded = numDimensions * numPoints;
    //int clustersLengthNeeded = numDimensions * numClusters;
    //double pointsArray[pointsLengthNeeded];
    //double clustersArray[clustersLengthNeeded];
    double *devicePoints;
    double *deviceClusters;
    double *deviceClustersTotals;
    int *deviceClusterAssignments;
    int *clusterAmount;

    cudaMalloc((void**)&devicePoints, pointsLengthNeeded * sizeof(double));
    cudaMalloc((void**)&deviceClusters, clustersLengthNeeded * sizeof(double));
    cudaMalloc((void**)&deviceClustersTotals, clustersLengthNeeded * sizeof(double));
    cudaMalloc((void**)&deviceClusterAssignments, totalPoints * sizeof(int));
    cudaMalloc((void**)&clusterAmount, (K+1) * sizeof(int));//the last one is use to tell if there r change or not
	
    cudaMemcpy(devicePoints, points, pointsLengthNeeded * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceClusters, clustersArray, clustersLengthNeeded * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceClustersTotals, clustersArrayTotals, clustersLengthNeeded * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(clusterAmount, clustersAmountPoints, (K+1) * sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(clusterAssignments, returnClusterNum, totalPoints * sizeof(int), cudaMemcpyDeviceToHost);
    int numBlocksPoints = 1;
    int threadPoints = 1024;
    if(totalPoints > 1024){
        numBlocksPoints = totalPoints / 1024 + 1;
    } else {
        threadPoints = totalPoints;
    }
    /*
    int numBlocksClusters = 1;
    int threadClusters = 1024;
    if(K > 1024){
        numBlocksClusters = K / 1024 + 1;
    } else {
        threadClusters = K;
    }*/
    int testChange[K+1];

    end_phase1 = chrono::high_resolution_clock::now();
    cout << "TIME PHASE 1 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end_phase1-begin).count()<<"\n";
    
   //end_phase1 = chrono::high_resolution_clock::now();
    std::cout << "hello?" << std::endl;
    //
	while(true){
        begin = chrono::high_resolution_clock::now();
        //printf("block: %d\n",numBlocksPoints);
        //printf("thread: %d\n",threadPoints);
        //begin = chrono::high_resolution_clock::now();
        findClusters<<<numBlocksPoints, threadPoints>>>(devicePoints, deviceClusters, deviceClustersTotals, deviceClusterAssignments, clusterAmount, numDimensions, K, totalPoints);
		cudaDeviceSynchronize();
        //cudaMemcpyAsync(testChange, changedCluster, (K+1) * sizeof(int), cudaMemcpyDeviceToHost, stream);
        end_phase1 = chrono::high_resolution_clock::now();
        cudaMemcpy(testChange, clusterAmount, (K+1) * sizeof(int), cudaMemcpyDeviceToHost);
        

        if(testChange[K] == 0){
            cout << "Break in iteration " << iter << "\n";
            break;
        }

		//calculateClusterCenter<<<numBlocksClusters, threadClusters>>>(devicePoints, deviceClusterAssignments, deviceClusters, clusterAmount, numDimensions, K, totalPoints);

		cudaDeviceSynchronize();


		if(iter >= max_iterations){
			cout << "Break in iteration " << iter << "\n";
			break;
		}
        std::cout << "1 " << std::endl;

        end = chrono::high_resolution_clock::now();

		iter++;
        cout << "Part 1 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end_phase1-begin).count()<<"\n";

	    cout << "Part 2 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end-end_phase1).count()<<"\n";
	}

    //end = chrono::high_resolution_clock::now();
	double clustersNewCenter[clustersLengthNeeded];
    cudaMemcpy(clustersNewCenter, deviceClusters, clustersLengthNeeded * sizeof(double), cudaMemcpyDeviceToHost);
	//for(int i = 0; i < K; i++){
    //    std::cout << "Cluster #" << i << ": ";
    //    for(int j = 0; j < numDimensions; j++){
    //        std::cout << clustersNewCenter[i * numDimensions + j] << " ";
    //    }
    //    std::cout << std::endl;
    //}
	cudaFree(devicePoints);
    cudaFree(deviceClusters);
    cudaFree(deviceClusterAssignments);

	cout << "TOTAL EXECUTION TIME = "<<std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()<<"\n";

	//cout << "TIME PHASE 1 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end_phase1-begin).count()<<"\n";

	cout << "TIME PHASE 2 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end-end_phase1).count()<<"\n";
    
}

double* read_points_csv(std::ifstream& dataFile, int total_points, int numDimensions, bool has_name) {
    // Allocate the data array for the points
    double* data = (double*)malloc(sizeof(double) * total_points * numDimensions);

    std::string line;
    int point_index = 0;

    // Process each subsequent line with point data
    while (std::getline(dataFile, line) && point_index < total_points) {
        // Split the line into components
        char* token = strtok(const_cast<char*>(line.c_str()), ",");
        
        // Assign values to the flat array (row-major order)
        for (int dim = 0; dim < numDimensions; dim++) {
            data[point_index * numDimensions + dim] = atof(token);
            token = strtok(nullptr, ",");
        }

        // Skip the name if present
        if (has_name) {
            // Move the token past the name (not needed, just skips it)
            token = strtok(nullptr, ",");
        }

        point_index++;
    }

    return data;
}

int main()//int argc, char *argv[])
{
	//if(argc == 0){
	//	cout << "wow" << endl;
	//}
	srand(741);//atoi(argv[1]));
    //std::ifstream dataFile("../datasets/dataset100000.txt");
    //std::ifstream dataFile("../../../exchange/datasets/dataset_100000.txt");
	//int total_points, numDimensions, K, max_iterations, has_name;

	//dataFile >> total_points >> numDimensions >> K >> max_iterations >> has_name;

	//std::vector<std::vector<double>> points;
    //double* points = read_points_csv("data.csv", 100, 5, true);
	//string point_name;
	//for(int i = 0; i < total_points; i++){
	//	vector<double> values;

	//	for(int j = 0; j < numDimensions; j++){
	//		double value;
	//		dataFile >> value;
	//		values.push_back(value);
	//	}

	//	if(has_name){
	//		dataFile >> point_name;
	//		points.push_back(values);
	//	} else {
	//		points.push_back(values);
	//	}
	//}

    //std::ifstream dataFile("../datasets/dataset100000.txt");


    std::ifstream dataFile("../../../exchange/datasets/dataset_100000.txt");

    int total_points, numDimensions, K, max_iterations;
    bool has_name;
    dataFile >> total_points >> numDimensions >> K >> max_iterations >> has_name;

    std::string skip_line;
    std::getline(dataFile, skip_line);

    double* points = read_points_csv(dataFile, total_points, numDimensions, has_name);

    std::cout << "what is happening" << std::endl;

    kmeansRun(points, total_points, numDimensions, K, max_iterations);

    // Example usage: Access point data like points[i * numDimensions + j]
    // Free the allocated memory after use
    free(points);

    // Close the file
    dataFile.close();
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

	//kmeansRun(points, total_points, numDimensions, K, max_iterations);

	return 0;
}