//reference: https://github.com/marcoscastro/kmeans

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
using namespace std;

__global__ void findClusters(double *devicePoints, double *deviceClusters,int *deviceClusterAssignments, int *changedCluster, int numDimensions, int numClusters, int numPoints){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < numPoints){
        double sum = 0.0, min_dist;
        int id_cluster_center = 0;

        for(int i = 0; i < numDimensions; i++){
            sum += powf(deviceClusters[i] - devicePoints[idx * numDimensions + i] , 2);
        }

        min_dist = sqrtf(sum);

        for(int i = 1; i < numClusters; i++){
            double dist;
            sum = 0.0;

            for(int j = 0; j < numDimensions; j++){
                sum += powf(deviceClusters[i * numDimensions + j] - devicePoints[idx * numDimensions + j] , 2);
            }

            dist = sqrtf(sum);
            if(dist < min_dist){
                min_dist = dist;
                id_cluster_center = i;
            }
        }
        int id_old = deviceClusterAssignments[idx];
        if(id_old != id_cluster_center){
            printf("old: %d\n", id_old);
            printf("new: %d\n", id_cluster_center);
            deviceClusterAssignments[idx] = id_cluster_center;
            int what = atomicExch(&changedCluster[id_cluster_center], 1);
            if(id_old != -1){
                int who = atomicExch(&changedCluster[id_old], 1);

            }
            int why = atomicExch(&changedCluster[numClusters], 1);
            printf("clustr change: %d\n", changedCluster[numClusters]);
        }
    }
}

__global__ void calculateClusterCenter(double *devicePoints, int *deviceClusterAssignments, double *deviceClusters, int *changedCluster, int numDimensions, int numClusters, int numPoints){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int counter = 0;
	
    if(idx < numClusters){
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
    if(idx == 0){
        changedCluster[numClusters] = 0;
    }
}

void kmeansRun(std::vector<std::vector<double>> points, int totalPoints, int numDimensions, int K, int max_iterations){
    auto begin = chrono::high_resolution_clock::now();
	if(K > totalPoints)
		return;

    int pointsLengthNeeded = numDimensions * totalPoints;
    double pointsArray[pointsLengthNeeded];

    for(int i = 0; i < totalPoints; i++){
        for(int j = 0; j < numDimensions; j++){
            pointsArray[i * numDimensions + j] = points[i][j];
        }
    }

	vector<int> prohibited_indexes;

    int clustersLengthNeeded = numDimensions * K;
    double clustersArray[clustersLengthNeeded];

	for(int i = 0; i < K; i++){
		while(true){
			int index_point = rand() % totalPoints;
			if(find(prohibited_indexes.begin(), prohibited_indexes.end(),index_point) == prohibited_indexes.end()){
				prohibited_indexes.push_back(index_point);
                for(int j = 0; j < numDimensions; j++){
                    clustersArray[i * numDimensions + j] = pointsArray[index_point * numDimensions + j];
                }
				break;
			}
		}
	}

    auto end_phase1 = chrono::high_resolution_clock::now();

	int iter = 1;
    double *devicePoints;
    double *deviceClusters;
    int *deviceClusterAssignments;
    int *changedCluster;

    cudaMalloc((void**)&devicePoints, pointsLengthNeeded * sizeof(double));
    cudaMalloc((void**)&deviceClusters, clustersLengthNeeded * sizeof(double));
    cudaMalloc((void**)&deviceClusterAssignments, totalPoints * sizeof(int));
    cudaMalloc((void**)&changedCluster, (K+1) * sizeof(int));
	
    cudaMemcpy(devicePoints, pointsArray, pointsLengthNeeded * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceClusters, clustersArray, clustersLengthNeeded * sizeof(double), cudaMemcpyHostToDevice);

    int numBlocksPoints = 1;
    int threadPoints = 1024;
    if(totalPoints > 1024){
        numBlocksPoints = totalPoints / 1024 + 1;
    } else {
        threadPoints = totalPoints;
    }
    int numBlocksClusters = 1;
    int threadClusters = 1024;
    if(K > 1024){
        numBlocksClusters = K / 1024 + 1;
    } else {
        threadClusters = K;
    }

	while(true){
        findClusters<<<numBlocksPoints, threadPoints>>>(devicePoints, deviceClusters, deviceClusterAssignments, changedCluster, numDimensions, K, totalPoints);

		cudaDeviceSynchronize();

        int testChange[K+1];
        cudaMemcpy(testChange, changedCluster, (K+1) * sizeof(int), cudaMemcpyDeviceToHost);
        
        if(testChange[K] == 0){
            cout << "Break in iteration " << iter << "\n";
            break;
        }
		calculateClusterCenter<<<numBlocksClusters, threadClusters>>>(devicePoints, deviceClusterAssignments, deviceClusters, changedCluster, numDimensions, K, totalPoints);
		cudaDeviceSynchronize();

		if(iter >= max_iterations){
			cout << "Break in iteration " << iter << "\n";
			break;
		}

		iter++;
	}
	double clustersNewCenter[clustersLengthNeeded];
    cudaMemcpy(clustersNewCenter, deviceClusters, clustersLengthNeeded * sizeof(double), cudaMemcpyDeviceToHost);
    
    auto end = chrono::high_resolution_clock::now();

	for(int i = 0; i < K; i++){
        std::cout << "Cluster #" << i << ": ";
        for(int j = 0; j < numDimensions; j++){
            std::cout << clustersNewCenter[i * numDimensions + j] << " ";
        }
        std::cout << std::endl;
    }
	cudaFree(devicePoints);
    cudaFree(deviceClusters);
    cudaFree(deviceClusterAssignments);

	cout << "TOTAL EXECUTION TIME = "<<std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()<<"\n";

	cout << "TIME PHASE 1 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end_phase1-begin).count()<<"\n";

	cout << "TIME PHASE 2 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end-end_phase1).count()<<"\n";
    
}

int main(){
	
	srand(741);

	std::ifstream dataFile("../datasets/Dry_Bean_Dataset.txt");
	int total_points, numDimensions, K, max_iterations, has_name;

	dataFile >> total_points >> numDimensions >> K >> max_iterations >> has_name;

	std::vector<std::vector<double>> points;
	string point_name;

	for(int i = 0; i < total_points; i++){
		vector<double> values;

		for(int j = 0; j < numDimensions; j++){
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

	kmeansRun(points, total_points, numDimensions, K, max_iterations);

	return 0;
}