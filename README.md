# K-means Clustering with CUDA

This project contains a GPU-accelerated implementation of the K-means clustering algorithm. The core of the project is the `atomicChange.cu` file, which represents the most optimized solution developed after exploring various approaches. Other files in the repository serve as a record of different optimization attempts and provide a basis for performance comparison.

## K-means Clustering: An Overview

K-means is a popular unsupervised machine learning algorithm for partitioning a dataset into a predefined number of `K` clusters. The process works by iteratively performing two main steps:

1.  **Assignment Step**: Each data point is assigned to its closest cluster centroid based on a distance metric (typically Euclidean distance).
2.  **Update Step**: The cluster centroids are recalculated as the average of all data points assigned to that cluster.

This loop continues until the cluster centroids no longer move significantly, indicating that the algorithm has converged. The goal is to minimize the distance between points within a cluster and maximize the distance between different clusters.

## My Best Implementation: `atomicChange.cu`

The `atomicChange.cu` file leverages the parallel processing capabilities of NVIDIA GPUs using CUDA to achieve significant speedups. This implementation includes several key optimizations:

-   **Initialization and Data Transfer**: Data points, initial cluster centroids, and point assignments are all copied from the CPU's memory to the GPU's memory at the start of the program.

-   **Parallel Assignment with Atomic Operations**: The `findClusters` CUDA kernel uses a **point-per-thread** approach. Each thread calculates the squared Euclidean distance from a single data point to all `K` cluster centroids. This avoids the computationally expensive `sqrt()` function. To handle concurrent updates to cluster totals and point counts, the kernel uses **atomic operations (`atomicAdd`)**, which I found to be more efficient than a lock-based system for this application for my testing, but there are probably some improvement to be made there in the future.

-   **Parallel Centroid Update**: The `calculations` kernel is highly parallel, with each thread responsible for updating a single dimension of a specific cluster. This ensures that the centroid recalculation step is completed very quickly.

-   **Cycle Iteration**: The program continues to iterate until no point changes its cluster assignment. I check for this with a boolean that will be changed in the kernal and sent back to the cpu everytime to check. I found that this small transfer doesnt take up much time at all in practice with it only being a few microseconds, so this was the best way to do it in all my testing while working with the gpu. There is probably slight improvements that can be made here to check within the kernel and break out of it somehow but I didnt look into it due to time and the small influence it would make.

## Performance Data

The following table compares the performance of the GPU-accelerated solution against an optimized, parallel CPU implementation that I made. The tests were run with 10 dimensions and 10 clusters, and the times are in microseconds (µs).

| # of points | CPU (µs) | GPU (µs) | Speedup |
|-------------|----------|----------|---------|
| 10,000      | 4740     | 60575    | 0.078x  |
| 50,000      | 36748    | 71877    | 0.51x   |
| 100,000     | 213049   | 173068   | 1.23x   |
| 250,000     | 448038   | 256579   | 1.75x   |
| 500,000     | 2489750  | 961212   | 2.59x   |
| 1,000,000   | 10773827 | 3241660  | 3.32x   |
| 5,000,000   | 34349932 | 9910121  | 3.47x   |
| 10,000,000  | 191186943| 46279899 | 4.13x   |

As the data shows, the GPU solution provides a significant **speedup** over the CPU for larger datasets. The speedup factor increases as the number of data points grows, demonstrating the effectiveness of the parallel approach. The speedup is also very quick compared ot the sequential version in the repo. The sequential version is also the fastest version I could get sequentially. 

There are definitely some improvements that could be made for all the different algorithms to optimize them even further. I appreciate you getting far in this repo and enjoying the analysis as much as I did. Have a good day!