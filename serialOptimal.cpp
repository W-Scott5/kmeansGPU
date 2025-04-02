// Implementation of the KMeans Algorithm
// reference: https://github.com/marcoscastro/kmeans

//change point and cluster to use a hashmap (unordered map) and maybe the vector clusters in kmeans

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

class Point
{
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
	//int numPoints;
	//vector<Point> points;

public:
	Cluster(int id_cluster, Point point)
	{
		this->id_cluster = id_cluster;

		//this-> numPoints = 0;

		int total_values = point.getTotalValues();

		for(int i = 0; i < total_values; i++)
			central_values.push_back(point.getValue(i));

		//points.push_back(point);
	}

	//void addPoint(Point point)
	//{
	//	points.push_back(point);
	//}
	//void addIntPoint(){
	//	numPoints++;
	//}
	//void removeIntPoint(){
	//	numPoints--;
	//}

	//bool removePoint(int id_point)
	//{
	//	int total_points = points.size();

	//	for(int i = 0; i < total_points; i++)
	//	{
	//		if(points[i].getID() == id_point)
	//		{
	//			points.erase(points.begin() + i);
	//			return true;
	//		}
	//	}
	//	return false;
	//}

	double getCentralValue(int index)
	{
		return central_values[index];
	}

	void setCentralValue(int index, double value)
	{
		central_values[index] = value;
	}

	//Point getPoint(int index)
	//{
	//	return points[index];
	//}

	//int getTotalPoints()
	//{
	//	return numPoints;
	//}

	int getID()
	{
		return id_cluster;
	}
};

class KMeans
{
private:
	int K; // number of clusters
	int total_values, total_points, max_iterations;
	vector<Cluster> clusters;

	// return ID of nearest center (uses euclidean distance)
	int getIDNearestCenter(Point point)
	{
		double sum = 0.0, min_dist;
		int id_cluster_center = 0;

		for(int i = 0; i < total_values; i++)
		{
			sum += pow(clusters[0].getCentralValue(i) -
					   point.getValue(i), 2.0);
		}

		min_dist = sqrt(sum);

		for(int i = 1; i < K; i++)
		{
			double dist;
			sum = 0.0;

			for(int j = 0; j < total_values; j++)
			{
				sum += pow(clusters[i].getCentralValue(j) -
						   point.getValue(j), 2.0);
			}

			dist = sqrt(sum);

			if(dist < min_dist)
			{
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

		vector<int> prohibited_indexes;

		for(int i = 0; i < K; i++)
		{
			while(true)
			{
				int index_point = rand() % total_points;

				if(find(prohibited_indexes.begin(), prohibited_indexes.end(),
						index_point) == prohibited_indexes.end())
				{
					prohibited_indexes.push_back(index_point);
					points[index_point].setCluster(i);
					Cluster cluster(i, points[index_point]);
					clusters.push_back(cluster);
					break;
				}
			}
		}
        auto end_phase1 = chrono::high_resolution_clock::now();

		int iter = 1;


		//auto midstuff = chrono::high_resolution_clock::now();
		//auto midstuff2 = chrono::high_resolution_clock::now();
		//auto midstuff3 = chrono::high_resolution_clock::now();
		//FILE *file2 = fopen("id.csv", "w");
		int clusterCount[K] = {0};
		double sums[K][total_values] = {0};
		int changeCluster[K] = {0};
		while(true)
		{
			bool done = true;
			for(int i = 0; i < total_points; i++){
				int id_old_cluster = points[i].getCluster();
				int id_nearest_center = getIDNearestCenter(points[i]);
				if(id_old_cluster != id_nearest_center){
					points[i].setCluster(id_nearest_center);
					if(id_old_cluster != 0){
						changeCluster[id_old_cluster] = 1;
					}
					changeCluster[id_nearest_center] = 1;
					done = false;
				}
			}

			for(int j = 0; j < total_points; j++){
				int idCluster = points[j].getCluster();					
				for(int z = 0; z < total_values; z++){
					sums[idCluster][z] += points[j].getValue(z);
				}
				clusterCount[idCluster] += 1;
			}
			for(int i = 0; i < K; i++){
				int countInd = clusterCount[i];
				if(countInd != 0 && changeCluster[i] != 0){
					for(int j = 0; j < total_values; j++){
						clusters[i].setCentralValue(j, sums[i][j] / countInd);
						sums[i][j] = 0;
					}
					clusterCount[i] = 0;
				}
			}

			if(done == true || iter >= max_iterations){
				cout << "Break in iteration " << iter << "\n";
				break;
			}

			iter++;
		}
        auto end = chrono::high_resolution_clock::now();
		//std::cout << "testing" << std::endl;//save this plz its working kinda
		FILE *file = fopen("serialOptimal.csv", "w");
		// shows elements of clusters
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
		
		{
			/*
			int total_points_cluster =  clusters[i].getTotalPoints();

			cout << "Cluster " << clusters[i].getID() + 1 << endl;
			for(int j = 0; j < total_points_cluster; j++)
			{
				cout << "Point " << clusters[i].getPoint(j).getID() + 1 << ": ";
				for(int p = 0; p < total_values; p++)
					cout << clusters[i].getPoint(j).getValue(p) << " ";

				string point_name = clusters[i].getPoint(j).getName();

				if(point_name != "")
					cout << "- " << point_name;

				cout << endl;
			}
			///////////save this
			cout << "Cluster values: ";

			for(int j = 0; j < total_values; j++)
				cout << clusters[i].getCentralValue(j) << " ";

			cout << "\n\n";
			*/
            cout << "Serial TOTAL EXECUTION TIME = "<<std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()<<"\n";

            cout << "Serial TIME PHASE 1 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end_phase1-begin).count()<<"\n";
			cout << "Serial TIME PHASE 2 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end-end_phase1).count()<<"\n";
			//cout << "TIME PHASE difference part 1 of phase 2 = "<<std::chrono::duration_cast<std::chrono::microseconds>(midstuff-midstuff3).count()<<"\n";
			//cout << "TIME PHASE difference part 2 of phase 2 = "<<std::chrono::duration_cast<std::chrono::microseconds>(midstuff2-midstuff).count()<<"\n";
		}
	}
};

int main(int argc, char *argv[])
{
	if(argc == 0){
		cout << "wow" << endl;
	}
	srand(atoi(argv[1]));

	
	std::ifstream dataFile("datasets/dataset500000.txt");
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

	KMeans kmeans(K, total_points, total_values, max_iterations);
	kmeans.run(points);

	return 0;
}