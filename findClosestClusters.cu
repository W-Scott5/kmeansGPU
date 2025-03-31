#include <stdio.h>
#include <iostream>
#include <vector>
#include <cuda_runtime.h> //powf to use in cuda device main thing

//__device__ int adding(int x, int y){
//    int result = x + y;
//    return result;
//}
//(devicePoints, deviceClusters, returnClusterNum, numDimensions, numClusters, numPoints);
__global__ void findClusters(double *devicePoints, double *deviceClusters,int *returnClusterNum , int numDimensions, int numClusters, int numPoints){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //if(idx < size){
    //    deviceArray[idx] = adding(idx, deviceArray[idx]);

    //}


    double sum = 0.0, min_dist;
	int id_cluster_center = 0;

	for(int i = 0; i < numDimensions; i++){
		//sum += pow(clusters[0].getCentralValue(i) - point.getValue(i), 2.0);
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
    //printf("%d",id_cluster_center);
	returnClusterNum[idx] = id_cluster_center;
}

int main() {
    int numDimensions = 4;
    int numClusters = 3;
    int numPoints = 5;
    //float arrayPoints[];
    std::vector<std::vector<double>> points = {
        {8.8, 9.9, 0.0, 1.1},
        {3.2, 3.4, 7.6, 2.8},
        {4.4, 5.5, 6.6, 7.7},
        {4.4, 5.5, 6.6, 7.7},
        {8.8, 9.9, 0.0, 1.1}
    };
    std::vector<std::vector<double>> clusters = {
        {3.2, 3.4, 7.6, 2.8},
        {4.4, 5.5, 6.6, 7.7},
        {8.8, 9.9, 0.0, 1.1},
    };
    int pointsLengthNeeded = numDimensions * numPoints;
    int clustersLengthNeeded = numDimensions * numClusters;
    double pointsArray[pointsLengthNeeded];
    double clustersArray[clustersLengthNeeded];
    double *devicePoints;
    double *deviceClusters;
    int *returnClusterNum;
    
    for(int i = 0; i < numPoints; i++){
        for(int j = 0; j < numDimensions; j++){
            pointsArray[i * numDimensions + j] = points[i][j];
        }
        //changePoint++;
    }
    for(int i = 0; i < numClusters; i++){
        for(int j = 0; j < numDimensions; j++){
            clustersArray[i * numDimensions + j] = clusters[i][j];
        }
    }



    //this stuff allocates the memory needed for the array
    cudaMalloc((void**)&devicePoints, pointsLengthNeeded * sizeof(double));
    cudaMalloc((void**)&deviceClusters, clustersLengthNeeded * sizeof(double));
    cudaMalloc((void**)&returnClusterNum, numPoints * sizeof(int));

    //yo basically this just copies the data from the host array to the device array to the position that is already allocated and it will be used
    cudaMemcpy(devicePoints, pointsArray, pointsLengthNeeded * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceClusters, clustersArray, clustersLengthNeeded * sizeof(double), cudaMemcpyHostToDevice);

    //this is one block of 10 threads - try multiple blocks and just get an understanding of it more than high level
    findClusters<<<1, numPoints>>>(devicePoints, deviceClusters, returnClusterNum, numDimensions, numClusters, numPoints);
    int clusterAssignments[numPoints];
    cudaMemcpy(clusterAssignments, returnClusterNum, numPoints * sizeof(int), cudaMemcpyDeviceToHost);
    
    for(int i = 0; i < numPoints; i++){
        std::cout << clusterAssignments[i] << " ";
    }

    cudaFree(devicePoints);
    cudaFree(deviceClusters);
    cudaFree(returnClusterNum);
    
    return 0;
}