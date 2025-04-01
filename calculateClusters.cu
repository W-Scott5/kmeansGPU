#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h> //powf to use in cuda device main thing

//__device__ int adding(int x, int y){
//    int result = x + y;
//    return result;
//}
//(devicePoints, deviceClusters, returnClusterNum, numDimensions, numClusters, numPoints);

//findClusters<<<1, numClusters>>>(devicePoints, deviceClusterAssignments, deviceClusters, numDimensions, numClusters, numPoints);
__global__ void calculateClusterCenter(double *devicePoints, int *deviceClusterAssignments, double *deviceClusters, int numDimensions, int numClusters, int numPoints){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < numClusters){
        printf("%d",idx);
        int counter = 0;
        for(int i = 0; i < numPoints; i++){
            if(deviceClusterAssignments[i] == idx){
                for(int j = 0; j < numDimensions; j++){
                    deviceClusters[idx * numDimensions + j] += devicePoints[i * numDimensions + j];
                }
                counter++;
            }
        }
        if(counter > 0){
            for(int i = 0; i < numDimensions; i++){
                deviceClusters[idx * numDimensions + i] = deviceClusters[idx * numDimensions + i] / counter;
            }
        }
    }
}

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

std::vector<int> generateRandomPoints(int rangeStart, int rangeEnd, int numPoints) {
    std::vector<int> points;
    // Seed the random number generator
    std::srand(std::time(0));

    // Generate numPoints random numbers in the given range
    for (int i = 0; i < numPoints; ++i) {
        int randomPoint = rangeStart + std::rand() % (rangeEnd - rangeStart + 1);
        points.push_back(randomPoint);
    }

    return points;
}

int main() {
    int numDimensions = 4;
    int numClusters = 6;
    int numPoints = 20000;
    //float arrayPoints[];
    std::vector<std::vector<double>> points = generatePoints(numPoints, numDimensions);

    std::vector<int> clusterAssignmentsIn = generateRandomPoints(0, 5, numPoints);
    std::vector<std::vector<double>> clusters = {
        {3.2, 3.4, 7.6, 2.8},
        {4.4, 5.5, 6.6, 7.7},
        {8.8, 9.9, 0.0, 1.1},
        {1.2, 3.4, 1.6, 4.8},
        {2.4, 1.5, 0.6, 9.7},
        {3.0, 9.9, 3.0, 9.1},
    };
    

    int pointsLengthNeeded = numDimensions * numPoints;
    int clustersLengthNeeded = numDimensions * numClusters;
    double pointsArray[pointsLengthNeeded];
    int clusterAssignmentsArray[numPoints];
    //float clustersArray[clustersLengthNeeded];
    double *devicePoints;
    int *deviceClusterAssignments;
    double *deviceClusters;
    
    for(int i = 0; i < numPoints; i++){
        for(int j = 0; j < numDimensions; j++){
            pointsArray[i * numDimensions + j] = points[i][j];
        }
        //changePoint++;
    }

    for(int i = 0; i < numPoints; i++){
        clusterAssignmentsArray[i] = clusterAssignmentsIn[i];//in the actual implementation just store everything in the array already so you dont need to copy it from vec to array plus the clusters calculations will just be in the same array you used
        //printf("%d\n", clusterAssignmentsArray[i]);
    }
   // for(int i = 0; i < numClusters; i++){
    //    for(int j = 0; j < numDimensions; j++){
    ///        clustersArray[i * numDimensions + j] = clusters[i][j];
    //    }
    //}



    //this stuff allocates the memory needed for the array
    cudaMalloc((void**)&devicePoints, pointsLengthNeeded * sizeof(double));
    cudaMalloc((void**)&deviceClusterAssignments, numPoints * sizeof(int));
    cudaMalloc((void**)&deviceClusters, clustersLengthNeeded * sizeof(double));

    //yo basically this just copies the data from the host array to the device array to the position that is already allocated and it will be used
    cudaMemcpy(devicePoints, pointsArray, pointsLengthNeeded * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceClusterAssignments, clusterAssignmentsArray, numPoints * sizeof(int), cudaMemcpyHostToDevice);

    //this is one block of 10 threads - try multiple blocks and just get an understanding of it more than high level
    calculateClusterCenter<<<300, 1000>>>(devicePoints, deviceClusterAssignments, deviceClusters, numDimensions, numClusters, numPoints);
    double clustersNewCenter[clustersLengthNeeded];
    cudaMemcpy(clustersNewCenter, deviceClusters, clustersLengthNeeded * sizeof(double), cudaMemcpyDeviceToHost);
    
    for(int i = 0; i < numClusters; i++){
        std::cout << "Cluster #" << i << ": ";
        for(int j = 0; j < numDimensions; j++){
            std::cout << clustersNewCenter[i * numDimensions + j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(devicePoints);
    cudaFree(deviceClusterAssignments);
    cudaFree(deviceClusters);
    
    return 0;
}