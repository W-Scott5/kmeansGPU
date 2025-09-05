// reference: https://github.com/marcoscastro/kmeans for starter structure

#include <sstream>
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

/*
    Description: 
        - A function for assigning the points to specific clusters based on euclidean distance calculations (with a bunch of optimizations that work for kmeans). As the function is assigning points,
          it also keeps tracks of the totals for each cluster and the amount of points in each cluster to figure our their exact location after each iteration in the second function below.
    Analysis: 
        - This function uses euclidean distance for each point to calculate the shortest distance from each cluster then gets assigned to that cluster that is closest to it. It is optimized to
          have a point per thread for speedup. Also, it gets rid of the sqrt calculations as we can jsut compare the totals without spending time to sqrt the numbers. The function then uses atomic
          sums in each cluster to calculate the totals and amount of points in each cluster. This doesnt look good in theory for thread contention but it is more optimal than utilizing a lock based
          system for the approach in our testing.
*/
__global__ void findClusters(double *devicePoints, double *deviceClusters, double *deviceClustersTotals, int *deviceClusterAssignments, int *clusterAmount, int numDimensions, int numClusters, int numPoints){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx == 0){
        atomicExch(&clusterAmount[numClusters], 0);
    }
    double pointVals[100];

    if(idx < numPoints){
        for(int i = 0; i < numDimensions; i++){
            pointVals[i] = devicePoints[idx * numDimensions + i];
        }


        double sum = 0.0, min_dist;
        int id_cluster_center = 0;

        for(int i = 0; i < numDimensions; i++){
            sum += powf(deviceClusters[i] - pointVals[i], 2);
        }

        min_dist = sum;

        for(int i = 1; i < numClusters; i++){
            double dist;
            sum = 0.0;

            for(int j = 0; j < numDimensions; j++){
                sum += powf(deviceClusters[i * numDimensions + j] - pointVals[j] , 2);
            }

            dist = sum;

            if(dist < min_dist){
                min_dist = dist;
                id_cluster_center = i;
            }
        }
        int id_old = deviceClusterAssignments[idx];
        if(id_old != id_cluster_center){
            deviceClusterAssignments[idx] = id_cluster_center;

            for(int i = 0; i < numDimensions; i++){
                atomicAdd(&deviceClustersTotals[id_cluster_center * numDimensions + i], pointVals[i]);
            }
            atomicAdd(&clusterAmount[id_cluster_center], 1);

            if(id_old != -1){
                for(int i = 0; i < numDimensions; i++){
                    atomicAdd(&deviceClustersTotals[id_old * numDimensions + i], (-1.0 * pointVals[i]));
                }
                atomicAdd(&clusterAmount[id_old], -1);
            }

            atomicExch(&clusterAmount[numClusters], 1);
        }
    }
}


/*
    Description: 
        - A function for doing the final cluster locations based on the totals and amount calculated in the findclusters function
    
    Analysis: 
        - This function splits all the clusters dimension into having one dimension per thread. For example if there are 10 clusters having 5 dimensions,
          then there would be 50 threads each calculating the exact decimal point location for that specific dimension in a specific cluster. This solution
          maximizes parallelism and only takes as long as the max thread to do their division calculation.
*/

__global__ void calculations(double *deviceClusters, double *deviceClustersTotals, int *clusterAmount, int numDimensions, int numClusters){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalAmount = numDimensions * numClusters;
    int clusterSpecificDim = idx / numDimensions;

    if(idx < totalAmount){
        deviceClusters[idx] = deviceClustersTotals[idx] / clusterAmount[clusterSpecificDim];
    }
}

/*
    Description: 
        - A function for putting all the kmeans operations together in the kernals and iterating through until no point changes their assigned cluster or if the maximum put in the data file is reached.
    Analysis: 
        - This function sets up all the points and to their designated cuda arrays to be transferred to gpu memory. All the arrays are flatened for this process to maximize efficiency. The function then
          assigns random points to be the starting clusters. We could have used k++ ideas and further optimized clusters at the start to be efficient and involve
          less calculation cycles where the points are changing clusters. The function then loops through each iteration until the boolean of no changes is true then it breaks and outputs the result and times it took.
*/

void kmeansRun(double* points, int totalPoints, int numDimensions, int K, int max_iterations){
    auto begin = chrono::high_resolution_clock::now();
    auto end_phase1 = chrono::high_resolution_clock::now();
    auto end = chrono::high_resolution_clock::now();
	if(K > totalPoints)
		return;

    begin = chrono::high_resolution_clock::now();
    int pointsLengthNeeded = numDimensions * totalPoints;

	vector<int> prohibited_indexes;

    int clustersLengthNeeded = numDimensions * K;
    double clustersArray[clustersLengthNeeded];
    double clustersArrayTotals[clustersLengthNeeded];
    int clustersAmountPoints[K+1];
    int* clusterAssignments = (int*)malloc(totalPoints * sizeof(int));

    for(int i = 0; i < totalPoints; i++){
        clusterAssignments[i] = -1;
    }

	for(int i = 0; i < K; i++){
		while(true){
			int index_point = rand() % totalPoints;
			if(find(prohibited_indexes.begin(), prohibited_indexes.end(),index_point) == prohibited_indexes.end()){
				prohibited_indexes.push_back(index_point);
                for(int j = 0; j < numDimensions; j++){
                    clustersArray[i * numDimensions + j] = points[index_point * numDimensions + j];
                    clustersArrayTotals[i * numDimensions + j] = points[index_point * numDimensions + j];
                }
                clustersAmountPoints[i] = 1;
                clusterAssignments[index_point] = i;
				break;
			}
		}
	}

	int iter = 1;

    double *devicePoints;
    double *deviceClusters;
    double *deviceClustersTotals;
    int *deviceClusterAssignments;
    int *clusterAmount;

    cudaMalloc((void**)&devicePoints, pointsLengthNeeded * sizeof(double));
    cudaMalloc((void**)&deviceClusters, clustersLengthNeeded * sizeof(double));
    cudaMalloc((void**)&deviceClustersTotals, clustersLengthNeeded * sizeof(double));
    cudaMalloc((void**)&deviceClusterAssignments, totalPoints * sizeof(int));
    cudaMalloc((void**)&clusterAmount, (K+1) * sizeof(int));
	
    cudaMemcpy(devicePoints, points, pointsLengthNeeded * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceClusters, clustersArray, clustersLengthNeeded * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceClustersTotals, clustersArrayTotals, clustersLengthNeeded * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(clusterAmount, clustersAmountPoints, (K+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceClusterAssignments, clusterAssignments, totalPoints * sizeof(int), cudaMemcpyHostToDevice);
    
    int numBlocksPoints = 1;
    int threadPoints = 1024;
    if(totalPoints > 1024){
        numBlocksPoints = totalPoints / 1024 + 1;
    } else {
        threadPoints = totalPoints;
    }
    
    int threadsNeededSecondKernel = numDimensions * K;
    int numBlocksClusters = 1;
    int threadClusters = 1024;
    if(threadsNeededSecondKernel > 1024){
        numBlocksClusters = threadsNeededSecondKernel / 1024 + 1;
    } else {
        threadClusters = threadsNeededSecondKernel;
    }
    int testChange[K+1];

    end_phase1 = chrono::high_resolution_clock::now();
    
	while(true){
        findClusters<<<numBlocksPoints, threadPoints>>>(devicePoints, deviceClusters, deviceClustersTotals, deviceClusterAssignments, clusterAmount, numDimensions, K, totalPoints);
		cudaDeviceSynchronize();
        
        cudaMemcpy(testChange, clusterAmount, (K+1) * sizeof(int), cudaMemcpyDeviceToHost);

        if(testChange[K] == 0){
            cout << "Break in iteration " << iter << "\n";
            break;
        }

        calculations<<<numBlocksClusters, threadClusters>>>(deviceClusters, deviceClustersTotals, clusterAmount, numDimensions, K);
		cudaDeviceSynchronize();

		if(iter >= max_iterations){
			cout << "Break in iteration " << iter << "\n";
			break;
		}

		iter++;
	}
    end = chrono::high_resolution_clock::now();

    for(int i = 0; i < K; i++){
        std::cout << testChange[i] << " ";
    }
    std::cout << "\n";

	cudaFree(devicePoints);
    cudaFree(deviceClusters);
    cudaFree(deviceClusterAssignments);
    cudaFree(deviceClustersTotals);
    cudaFree(clusterAmount);

	cout << "Total Execution Time = "<<std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()<<"\n";

	cout << "Time Phase 1 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end_phase1-begin).count()<<"\n";

	cout << "Time Phase 2 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end-end_phase1).count()<<"\n";
    
}

//A method for reading csv files for kmeans point and cluster amounts/data
double* read_points_csv(std::ifstream& dataFile, int total_points, int numDimensions, bool has_name) {
    
    double* data = (double*)malloc(sizeof(double) * total_points * numDimensions);
    std::string line;
    int point_index = 0;

    while (std::getline(dataFile, line) && point_index < total_points) {
        char* token = strtok(const_cast<char*>(line.c_str()), ",");

        for (int dim = 0; dim < numDimensions; dim++) {
            data[point_index * numDimensions + dim] = atof(token);
            token = strtok(nullptr, ",");
        }

        if (has_name) {
            token = strtok(nullptr, ",");
        }
        point_index++;
    }

    return data;
}

//A method for reading text files for kmeans point and cluster amounts/data
double* read_points_txt(std::ifstream& dataFile, int total_points, int numDimensions, bool has_name) {

    double* data = (double*)malloc(sizeof(double) * total_points * numDimensions);
    std::string line;
    int point_index = 0;

    while (std::getline(dataFile, line) && point_index < total_points) {

        std::istringstream stream(line);
        std::string token;

        for (int dim = 0; dim < numDimensions; dim++) {
            stream >> token;
            data[point_index * numDimensions + dim] = atof(token.c_str());
        }

        if (has_name) {
            stream >> token;
        }
        point_index++;
    }
    return data;
}

int main(){

	srand(741); //using 741 for consistent test but also tested with many other random values for correctness 
    std::ifstream dataFile("../datasets/dataset50K.txt");
    
    int total_points, numDimensions, K, max_iterations;
    bool has_name;
    dataFile >> total_points >> numDimensions >> K >> max_iterations >> has_name;

    std::string skip_line;
    std::getline(dataFile, skip_line);
    double* points = read_points_txt(dataFile, total_points, numDimensions, has_name);

    kmeansRun(points, total_points, numDimensions, K, max_iterations);

    free(points);
    dataFile.close();

	return 0;
}