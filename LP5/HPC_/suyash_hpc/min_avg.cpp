#include <iostream>
#include <vector>
#include <algorithm> // for std::min_element, std::max_element
#include <numeric>   // for std::accumulate
#include <chrono>    // for timing
#include <omp.h>     // for OpenMP

// Function to compute minimum using parallel reduction
int parallelMin(const std::vector<int>& array) {
    int min_val = array[0]; // Initialize min_val with the first element

    // Use OpenMP parallel reduction for finding the minimum value
    #pragma omp parallel for reduction(min: min_val)
    for (size_t i = 0; i < array.size(); ++i) {
        if (array[i] < min_val) {
            min_val = array[i];
        }
    }

    return min_val;
}

// Function to compute maximum using parallel reduction
int parallelMax(const std::vector<int>& array) {
    int max_val = array[0]; // Initialize max_val with the first element

    // Use OpenMP parallel reduction for finding the maximum value
    #pragma omp parallel for reduction(max: max_val)
    for (size_t i = 0; i < array.size(); ++i) {
        if (array[i] > max_val) {
            max_val = array[i];
        }
    }

    return max_val;
}

// Function to compute sum using parallel reduction
int parallelSum(const std::vector<int>& array) {
    int sum = 0;

    // Use OpenMP parallel reduction for summing up the elements
    #pragma omp parallel for reduction(+: sum)
    for (size_t i = 0; i < array.size(); ++i) {
        sum += array[i];
    }

    return sum;
}

// Function to compute average using parallel reduction
double parallelAverage(const std::vector<int>& array) {
    int sum = parallelSum(array); // Compute sum using parallel reduction
    double avg = static_cast<double>(sum) / array.size(); // Compute average

    return avg;
}

int main() {
    // Example usage
    std::vector<int> array = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3};

    // Sequential execution time measurement for each operation
    auto start_seq = std::chrono::high_resolution_clock::now();
    int min_val_seq = *std::min_element(array.begin(), array.end());
    int max_val_seq = *std::max_element(array.begin(), array.end());
    int sum_seq = std::accumulate(array.begin(), array.end(), 0);
    double avg_seq = static_cast<double>(sum_seq) / array.size();
    auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_seq = end_seq - start_seq;

    // Parallel execution time measurement for each operation
    auto start_par = std::chrono::high_resolution_clock::now();
    int min_val_par = parallelMin(array);
    int max_val_par = parallelMax(array);
    int sum_par = parallelSum(array);
    double avg_par = parallelAverage(array);
    auto end_par = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_par = end_par - start_par;

    // Calculate speedup for each operation
    double speedup_min = duration_seq.count() / duration_par.count();
    double speedup_max = duration_seq.count() / duration_par.count();
    double speedup_sum = duration_seq.count() / duration_par.count();
    double speedup_avg = duration_seq.count() / duration_par.count();

    std::cout << "Sequential Min: " << min_val_seq << ", Time: " << duration_seq.count() << "s\n";
    std::cout << "Parallel Min: " << min_val_par << ", Time: " << duration_par.count() << "s, Speedup: " << speedup_min << "x\n";

    std::cout << "Sequential Max: " << max_val_seq << ", Time: " << duration_seq.count() << "s\n";
    std::cout << "Parallel Max: " << max_val_par << ", Time: " << duration_par.count() << "s, Speedup: " << speedup_max << "x\n";

    std::cout << "Sequential Sum: " << sum_seq << ", Time: " << duration_seq.count() << "s\n";
    std::cout << "Parallel Sum: " << sum_par << ", Time: " << duration_par.count() << "s, Speedup: " << speedup_sum << "x\n";

    std::cout << "Sequential Avg: " << avg_seq << ", Time: " << duration_seq.count() << "s\n";
    std::cout << "Parallel Avg: " << avg_par << ", Time: " << duration_par.count() << "s, Speedup: " << speedup_avg << "x\n";

    return 0;
}
