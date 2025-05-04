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
    if(idx == 0){
        atomicExch(&clusterAmount[numClusters], 0);
    }

    //this is faster but has to be static just remember to make this more than the num of dimensions sohoudl be good at 100
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
    //__syncthreads();
    //future - make sure you check the number of threads is enough and make it numdimensions * numclusters if that is greater than num of points 
}

__global__ void calculations(double *deviceClusters, double *deviceClustersTotals, int *clusterAmount, int numDimensions, int numClusters){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalAmount = numDimensions * numClusters;
    int clusterSpecificDim = idx / numDimensions;

    if(idx < totalAmount){
        deviceClusters[idx] = deviceClustersTotals[idx] / clusterAmount[clusterSpecificDim];// 6/5=1 17/5=
        //printf("val: %f = %f / %d\n", deviceClusters[idx], deviceClustersTotals[idx], clusterAmount[clusterSpecificDim]);
    }
}

void kmeansRun(double* points, int totalPoints, int numDimensions, int K, int max_iterations){
    auto begin = chrono::high_resolution_clock::now();
    auto end_phase1 = chrono::high_resolution_clock::now();
    auto end = chrono::high_resolution_clock::now();
	if(K > totalPoints)
		return;

    begin = chrono::high_resolution_clock::now();
    int pointsLengthNeeded = numDimensions * totalPoints;

	vector<int> prohibited_indexes;
    std::cout << "here 2" << std::endl;

    int clustersLengthNeeded = numDimensions * K;
    double clustersArray[clustersLengthNeeded];
    double clustersArrayTotals[clustersLengthNeeded];
    int clustersAmountPoints[K+1];
    std::cout << "here 3.1" << std::endl;
    int clusterAssignments[totalPoints];
    std::cout << "here 3" << std::endl;

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
    std::cout << "here" << std::endl;

    cudaMalloc((void**)&devicePoints, pointsLengthNeeded * sizeof(double));
    cudaMalloc((void**)&deviceClusters, clustersLengthNeeded * sizeof(double));
    cudaMalloc((void**)&deviceClustersTotals, clustersLengthNeeded * sizeof(double));
    cudaMalloc((void**)&deviceClusterAssignments, totalPoints * sizeof(int));
    cudaMalloc((void**)&clusterAmount, (K+1) * sizeof(int));//the last one is use to tell if there r change or not
	
    cudaMemcpy(devicePoints, points, pointsLengthNeeded * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceClusters, clustersArray, clustersLengthNeeded * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceClustersTotals, clustersArrayTotals, clustersLengthNeeded * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(clusterAmount, clustersAmountPoints, (K+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceClusterAssignments, clusterAssignments, totalPoints * sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(clusterAssignments, returnClusterNum, totalPoints * sizeof(int), cudaMemcpyDeviceToHost);
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
    std::cout << "here" << std::endl;
    
	while(true){

        findClusters<<<numBlocksPoints, threadPoints>>>(devicePoints, deviceClusters, deviceClustersTotals, deviceClusterAssignments, clusterAmount, numDimensions, K, totalPoints);
		cudaDeviceSynchronize();
        
        cudaMemcpy(testChange, clusterAmount, (K+1) * sizeof(int), cudaMemcpyDeviceToHost);
        
        //for(int i = 0; i < K; i++){
        //    std::cout << testChange[i] << " ";
        //}
        //std::cout << "\n";
        

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

	cout << "TOTAL EXECUTION TIME = "<<std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()<<"\n";

	cout << "TIME PHASE 1 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end_phase1-begin).count()<<"\n";

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

double* read_points_txt(std::ifstream& dataFile, int total_points, int numDimensions, bool has_name) {
    // Allocate the data array for the points
    double* data = (double*)malloc(sizeof(double) * total_points * numDimensions);

    std::string line;
    int point_index = 0;

    // Process each subsequent line with point data
    while (std::getline(dataFile, line) && point_index < total_points) {
        // Use a stringstream to split by space
        std::istringstream stream(line);
        std::string token;

        // Assign values to the flat array (row-major order)
        for (int dim = 0; dim < numDimensions; dim++) {
            stream >> token;
            data[point_index * numDimensions + dim] = atof(token.c_str());
        }

        // Skip the name if present
        if (has_name) {
            // Move the stream past the name (not needed, just skips it)
            stream >> token;
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

    std::ifstream dataFile("../datasets/drybeans.txt");


    //std::ifstream dataFile("../../../exchange/datasets/dataset_5000000.txt");

    
    int total_points, numDimensions, K, max_iterations;
    bool has_name;
    dataFile >> total_points >> numDimensions >> K >> max_iterations >> has_name;

    std::string skip_line;
    std::getline(dataFile, skip_line);

    //double* points = read_points_csv(dataFile, total_points, numDimensions, has_name);
    double* points = read_points_txt(dataFile, total_points, numDimensions, has_name);

    std::cout << "here" << std::endl;

    kmeansRun(points, total_points, numDimensions, K, max_iterations);

    // Example usage: Access point data like points[i * numDimensions + j]
    // Free the allocated memory after use
    free(points);

    // Close the file
    dataFile.close();
    

    ///////////////////////////////////////////////////////////////////

    /*

    int total_points, total_values, K, max_iterations, has_name;
    string line;

    // Read the first line (metadata)
    getline(dataFile, line);
    stringstream ss(line);
    string temp;

    getline(ss, temp, ' '); total_points = stoi(temp);
    getline(ss, temp, ' '); total_values = stoi(temp);
    getline(ss, temp, ' '); K = stoi(temp);
    getline(ss, temp, ' '); max_iterations = stoi(temp);
    getline(ss, temp, ' '); has_name = stoi(temp);

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