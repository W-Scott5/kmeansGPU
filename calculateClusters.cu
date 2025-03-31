#include <stdio.h>
#include <iostream>
#include <vector>
#include <cuda_runtime.h> //powf to use in cuda device main thing

//__device__ int adding(int x, int y){
//    int result = x + y;
//    return result;
//}
//(devicePoints, deviceClusters, returnClusterNum, numDimensions, numClusters, numPoints);

//findClusters<<<1, numClusters>>>(devicePoints, deviceClusterAssignments, deviceClusters, numDimensions, numClusters, numPoints);
__global__ void calculateClusterCenter(float *devicePoints, int *deviceClusterAssignments, float *deviceClusters, int numDimensions, int numClusters, int numPoints){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int counter = 0;
    for(int i = 0; i < numPoints; i++){
        if(deviceClusterAssignments[i] == idx){
            for(int j = 0; j < numDimensions; j++){
                deviceClusters[idx * numDimensions + j] += devicePoints[i * numDimensions + j];
            }
            counter++;
        }
    }

    for(int i = 0; i < numDimensions; i++){
        printf("Cluster %d, Value: %f\n",idx, deviceClusters[idx * numDimensions + numDimensions]);
        deviceClusters[idx * numDimensions + numDimensions] = deviceClusters[idx * numDimensions + numDimensions] / counter;
    }



    /*
    float sum = 0.0, min_dist;
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
    */
}

int main() {
    int numDimensions = 4;
    int numClusters = 3;
    int numPoints = 5;
    //float arrayPoints[];
    std::vector<std::vector<float>> points = {
        {8.8, 9.9, 0.0, 1.1},
        {2.1, 4.3, 6.5, 8.7},
        {4.4, 5.5, 6.6, 7.7},
        {4.4, 5.5, 6.6, 7.7},
        {8.8, 9.9, 0.0, 1.1}
    };
    std::vector<int> clusterAssignmentsIn = {1,1,2,1,0};
    std::vector<std::vector<float>> clusters = {
        {3.2, 3.4, 7.6, 2.8},
        {4.4, 5.5, 6.6, 7.7},
        {8.8, 9.9, 0.0, 1.1},
    };
    

    int pointsLengthNeeded = numDimensions * numPoints;
    int clustersLengthNeeded = numDimensions * numClusters;
    float pointsArray[pointsLengthNeeded];
    int clusterAssignmentsArray[numClusters];
    //float clustersArray[clustersLengthNeeded];
    float *devicePoints;
    int *deviceClusterAssignments;
    float *deviceClusters;
    
    for(int i = 0; i < numPoints; i++){
        for(int j = 0; j < numDimensions; j++){
            pointsArray[i * numDimensions + j] = points[i][j];
        }
        //changePoint++;
    }

    for(int i = 0; i < numClusters; i++){
        clusterAssignmentsArray[i] = clusterAssignmentsIn[i];//in the actual implementation just store everything in the array already so you dont need to copy it from vec to array plus the clusters calculations will just be in the same array you used
    }
   // for(int i = 0; i < numClusters; i++){
    //    for(int j = 0; j < numDimensions; j++){
    ///        clustersArray[i * numDimensions + j] = clusters[i][j];
    //    }
    //}



    //this stuff allocates the memory needed for the array
    cudaMalloc((void**)&devicePoints, pointsLengthNeeded * sizeof(float));
    cudaMalloc((void**)&deviceClusterAssignments, numClusters * sizeof(int));
    cudaMalloc((void**)&deviceClusters, clustersLengthNeeded * sizeof(float));

    //yo basically this just copies the data from the host array to the device array to the position that is already allocated and it will be used
    cudaMemcpy(devicePoints, pointsArray, pointsLengthNeeded * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceClusterAssignments, clusterAssignmentsArray, numClusters * sizeof(int), cudaMemcpyHostToDevice);

    //this is one block of 10 threads - try multiple blocks and just get an understanding of it more than high level
    calculateClusterCenter<<<1, numClusters>>>(devicePoints, deviceClusterAssignments, deviceClusters, numDimensions, numClusters, numPoints);
    float clustersNewCenter[clustersLengthNeeded];
    cudaMemcpy(clustersNewCenter, deviceClusters, clustersLengthNeeded * sizeof(float), cudaMemcpyDeviceToHost);
    
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