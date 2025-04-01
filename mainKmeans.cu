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
using namespace std;

//findClusters<<<1000, 1024>>>(devicePoints, deviceClusters, returnClusterNum, numDimensions, K, totalPoints);

__global__ void findClusters(double *devicePoints, double *deviceClusters,int *deviceClusterAssignments , int numDimensions, int numClusters, int numPoints){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < numPoints){
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

			//printf("dist: %f\n", dist);
			//printf("min: %f\n", min_dist);
            if(dist < min_dist){
                min_dist = dist;
                id_cluster_center = i;
            }
        }
        deviceClusterAssignments[idx] = id_cluster_center;
    }
}
//calculateClusterCenter<<<1, 4>>>(devicePoints, deviceClusterAssignments, deviceClusters, numDimensions, K, totalPoints);
__global__ void calculateClusterCenter(double *devicePoints, int *deviceClusterAssignments, double *deviceClusters, int numDimensions, int numClusters, int numPoints){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("idx: %d\n", idx);
	int counter = 0;
	
    if(idx < numClusters){
        //printf("%d",idx);
        for(int i = 0; i < numPoints; i++){
            if(deviceClusterAssignments[i] == idx){
                for(int j = 0; j < numDimensions; j++){
                    if(counter == 0){
                        deviceClusters[idx * numDimensions + j] = 0;
                    }
					
                    deviceClusters[idx * numDimensions + j] += devicePoints[i * numDimensions + j];
                    //printf("cluter: %d, test: %f\n", idx, deviceClusters[idx * numDimensions + j]);
                }
                counter++;
            }
        }
        if(counter > 0){
            for(int i = 0; i < numDimensions; i++){
                deviceClusters[idx * numDimensions + i] = deviceClusters[idx * numDimensions + i] / (counter * 1.0);
            }
        }
    }
}

//kmeansRun(points, total_points, numDimensions, K, max_iterations);
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
	//for(int i = 0; i < pointsLengthNeeded; i++){
	//	printf("%f ", pointsArray[i]);
	//}

	vector<int> prohibited_indexes;

    int clustersLengthNeeded = numDimensions * K;
    double clustersArray[clustersLengthNeeded];

	// choose K distinct values for the centers of the clusters
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
	//printf("cluster 0: %d\n", prohibited_indexes[0]);
	//printf("cluster 1: %d\n", prohibited_indexes[1]);

    auto end_phase1 = chrono::high_resolution_clock::now();

	int iter = 1;

    //int pointsLengthNeeded = numDimensions * numPoints;
    //int clustersLengthNeeded = numDimensions * numClusters;
    //double pointsArray[pointsLengthNeeded];
    //double clustersArray[clustersLengthNeeded];
    double *devicePoints;
    double *deviceClusters;
    int *deviceClusterAssignments;

    cudaMalloc((void**)&devicePoints, pointsLengthNeeded * sizeof(double));
    cudaMalloc((void**)&deviceClusters, clustersLengthNeeded * sizeof(double));
    cudaMalloc((void**)&deviceClusterAssignments, totalPoints * sizeof(int));
	//need another clusters array to compare and see if we need to stop
    cudaMemcpy(devicePoints, pointsArray, pointsLengthNeeded * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceClusters, clustersArray, clustersLengthNeeded * sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy(clusterAssignments, returnClusterNum, totalPoints * sizeof(int), cudaMemcpyDeviceToHost);
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
		//bool done = false;

        //yo basically this just copies the data from the host array to the device array to the position that is already allocated and it will be used

        //this is one block of 10 threads - try multiple blocks and just get an understanding of it more than high level
        findClusters<<<numBlocksPoints, threadPoints>>>(devicePoints, deviceClusters, deviceClusterAssignments, numDimensions, K, totalPoints);

		cudaDeviceSynchronize();

		calculateClusterCenter<<<numBlocksClusters, threadClusters>>>(devicePoints, deviceClusterAssignments, deviceClusters, numDimensions, K, totalPoints);

		cudaDeviceSynchronize();


        //int clusterAssignments[total];
        
        //for(int i = 0; i < numPoints; i++){
        //    std::cout << clusterAssignments[i] << " ";
        //}


			/*
			tbb::parallel_for(tbb::blocked_range<int>(0, total_points),
        	[&](const tbb::blocked_range<int>& r){
				for (int i = r.begin(); i != r.end(); ++i){
					int id_old_cluster = points[i].getCluster();
					int id_nearest_center = getIDNearestCenter(points[i]);
					if(id_old_cluster != id_nearest_center){
						points[i].setCluster(id_nearest_center);
						done = false;
					}
				}
			});*/
			/*
			tbb::parallel_for(tbb::blocked_range<int>(0, K),
        	[&](const tbb::blocked_range<int>& r){
				for (int i = r.begin(); i != r.end(); ++i){
					//if(clusterSumChange[i] == 1){
						int counter = 0;
						std::vector<double> sums(total_values, 0.0);
						for(int j = 0; j < total_points; j++){						
							if(points[j].getCluster() == i){
								for(int z = 0; z < total_values; z++){
									sums[z] += points[j].getValue(z);
								}
								counter++;
							}
						}
						if(counter != 0){
							for(int j = 0; j < total_values; j++){
								clusters[i].setCentralValue(j, sums[j] / counter);
							}
						}
				}
			});*/

		//printf("max: %d\n", max_iterations);
		if(iter >= max_iterations){
			cout << "Break in iteration " << iter << "\n";
			break;
		}
        //cudaDeviceSynchronize();

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

int main()//int argc, char *argv[])
{
	//if(argc == 0){
	//	cout << "wow" << endl;
	//}
	srand(741);//atoi(argv[1]));

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