#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h> //powf to use in cuda device main thing

//__device__ int adding(int x, int y){
//    int result = x + y;
//    return result;
//}
//(devicePoints, deviceClusters, returnClusterNum, numDimensions, numClusters, numPoints);

std::vector<std::vector<double>> generatePoints(int numPoints, int dimensions) {
    std::vector<std::vector<double>> points(numPoints, std::vector<double>(dimensions));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 10.0);

    for (int i = 0; i < numPoints; ++i) {
        for (int j = 0; j < dimensions; ++j) {
            points[i][j] = dis(gen);
        }
    }
    return points;
}

int main() {
    int numDimensions = 4;
    int numClusters = 3;
    int numPoints = 2024;
    //float arrayPoints[];
    std::vector<std::vector<double>> points = generatePoints(numPoints, numDimensions);
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
    }
    for(int i = 0; i < numClusters; i++){
        for(int j = 0; j < numDimensions; j++){
            clustersArray[i * numDimensions + j] = clusters[i][j];
        }
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Max blocks per multiprocessor: " << prop.maxGridSize[0] << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;



    //this stuff allocates the memory needed for the array
    cudaMalloc((void**)&devicePoints, pointsLengthNeeded * sizeof(double));
    cudaMalloc((void**)&deviceClusters, clustersLengthNeeded * sizeof(double));
    cudaMalloc((void**)&returnClusterNum, numPoints * sizeof(int));

    //yo basically this just copies the data from the host array to the device array to the position that is already allocated and it will be used
    cudaMemcpy(devicePoints, pointsArray, pointsLengthNeeded * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceClusters, clustersArray, clustersLengthNeeded * sizeof(double), cudaMemcpyHostToDevice);

    //this is one block of 10 threads - try multiple blocks and just get an understanding of it more than high level
    findClusters<<<1000000, 1024>>>(devicePoints, deviceClusters, returnClusterNum, numDimensions, numClusters, numPoints);
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