//reference: https://github.com/marcoscastro/kmeans

#include <sstream>
#include <tbb/parallel_for.h>
#include <iostream>
#include <unordered_set>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <fstream>
using namespace std;

class Point{
private:
	int id_point, id_cluster;
	vector<double> values;
	int total_values;
	string name;

public:
	Point(int id_point, vector<double>& values, string name = "")
	{
		this->id_point = id_point;
		total_values = values.size();

		for(int i = 0; i < total_values; i++)
			this->values.push_back(values[i]);

		this->name = name;
		id_cluster = -1;
	}

	int getID()
	{
		return id_point;
	}

	void setCluster(int id_cluster)
	{
		this->id_cluster = id_cluster;
	}

	int getCluster()
	{
		return id_cluster;
	}

	double getValue(int index)
	{
		return values[index];
	}

	int getTotalValues()
	{
		return total_values;
	}

	void addValue(double value)
	{
		values.push_back(value);
	}

	string getName()
	{
		return name;
	}
};

class Cluster
{
private:
	int id_cluster;
	vector<double> central_values;

public:
	Cluster(int id_cluster, Point point)
	{
		this->id_cluster = id_cluster;

		int total_values = point.getTotalValues();

		for(int i = 0; i < total_values; i++)
			central_values.push_back(point.getValue(i));
	}

	double getCentralValue(int index)
	{
		return central_values[index];
	}

	void setCentralValue(int index, double value)
	{
		central_values[index] = value;
	}

	int getID()
	{
		return id_cluster;
	}
};

class KMeans
{
private:
	int K; 
	int total_values, total_points, max_iterations;
	vector<Cluster> clusters;

	int getIDNearestCenter(Point point)
	{
		double sum = 0.0, min_dist;
		int id_cluster_center = 0;

		for(int i = 0; i < total_values; i++){
			sum += pow(clusters[0].getCentralValue(i) -
					   point.getValue(i), 2.0);
		}

		min_dist = sum;

		for(int i = 1; i < K; i++){
			double dist;
			sum = 0.0;

			for(int j = 0; j < total_values; j++){
				sum += pow(clusters[i].getCentralValue(j) -
						   point.getValue(j), 2.0);
			}

			dist = sum;

			if(dist < min_dist){
				min_dist = dist;
				id_cluster_center = i;
			}
		}

		return id_cluster_center;
	}

public:
	KMeans(int K, int total_points, int total_values, int max_iterations)
	{
		this->K = K;
		this->total_points = total_points;
		this->total_values = total_values;
		this->max_iterations = max_iterations;
	}

	void run(vector<Point> & points)
	{
        auto begin = chrono::high_resolution_clock::now();
		if(K > total_points)
			return;

        double sums[K][total_values] = {0};
		int clusterCount[K] = {0};

		vector<int> prohibited_indexes;

		for(int i = 0; i < K; i++){
			while(true){
				int index_point = rand() % total_points;

				if(find(prohibited_indexes.begin(), prohibited_indexes.end(),
						index_point) == prohibited_indexes.end())
				{
					prohibited_indexes.push_back(index_point);
					points[index_point].setCluster(i);
                    for(int z = 0; z < total_values; z++){
                        sums[i][z] += points[index_point].getValue(z);
                    }
					clusterCount[i] += 1;
					Cluster cluster(i, points[index_point]);
					clusters.push_back(cluster);
					break;
				}
			}
		}
        auto end_phase1 = chrono::high_resolution_clock::now();

		int iter = 1;

		int changeCluster[K] = {0};
		while(true){
			bool done = true;
			for(int i = 0; i < total_points; i++){
				int id_old_cluster = points[i].getCluster();
				int id_nearest_center = getIDNearestCenter(points[i]);
				if(id_old_cluster != id_nearest_center){
					points[i].setCluster(id_nearest_center);
					clusterCount[id_nearest_center] += 1;
                    for(int z = 0; z < total_values; z++){
                        sums[id_nearest_center][z] += points[i].getValue(z);
                    }
					if(id_old_cluster != -1){
						changeCluster[id_old_cluster] = 1;
                        for(int z = 0; z < total_values; z++){
                            sums[id_old_cluster][z] -= points[i].getValue(z);
                        }
						clusterCount[id_old_cluster] -= 1;
					}
					changeCluster[id_nearest_center] = 1;
					done = false;
				}
			}
			for(int i = 0; i < K; i++){
				int countInd = clusterCount[i];
				if(countInd != 0 && changeCluster[i] != 0){
					for(int j = 0; j < total_values; j++){
						clusters[i].setCentralValue(j, sums[i][j] / countInd);
					}
				}
			}

			if(done == true || iter >= max_iterations){
				cout << "Break in iteration " << iter << "\n";
				break;
			}

			iter++;
		}
        auto end = chrono::high_resolution_clock::now();
		FILE *file = fopen("optimizeSerialSums.csv", "w");

		for(int i = 0; i < K; i++){
			for(int j= 0; j < total_values; j++){
				fprintf(file, "%f", clusters[i].getCentralValue(j));
				if (j < total_values){
					fprintf(file, "%s",",");
				}
			}
			fprintf(file, "\n");
		}
		fclose(file);
	}
};

int main(int argc, char *argv[])
{
	if(argc == 0){
		cout << "wow" << endl;
	}
	srand(atoi(argv[1]));

	std::ifstream dataFile("datasets/Dry_Bean_Dataset.txt");
	int total_points, total_values, K, max_iterations, has_name;

	dataFile >> total_points >> total_values >> K >> max_iterations >> has_name;

	vector<Point> points;
	string point_name;

	for(int i = 0; i < total_points; i++)
	{
		vector<double> values;

		for(int j = 0; j < total_values; j++)
		{
			double value;
			dataFile >> value;
			values.push_back(value);
		}

		if(has_name)
		{
			dataFile >> point_name;
			Point p(i, values, point_name);
			points.push_back(p);
		}
		else
		{
			Point p(i, values);
			points.push_back(p);
		}
	}

	KMeans kmeans(K, total_points, total_values, max_iterations);
	kmeans.run(points);

	return 0;
}