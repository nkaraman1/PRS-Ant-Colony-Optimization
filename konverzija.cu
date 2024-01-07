#include <iostream>
#include <vector>
#include <array>
#include <cassert>
#include <cmath>
#include <curand_kernel.h>

// Problem Parameters
constexpr int CITIES = 100;
constexpr int ANTS = 2000;
constexpr int MAX_DIST = 100;
constexpr float ALPHA = 1.0f;
constexpr float BETA = 5.0f;      // This parameter raises the weight of distance over pheromone
constexpr float RHO = 0.5f;       // Evaporation rate
constexpr float QVAL = 100.0f;
constexpr int MAX_TOURS = 50;     // The number of times an ant will walk through all the cities
constexpr float INIT_PHER = 1.0f / CITIES; // Initial pheromone for each path
constexpr float MAX_TOTAL_DISTANCE = static_cast<float>(CITIES * MAX_DIST); // MAX possible distance that an ant can walk

struct Ant {
    int curCity, nextCity, pathIndex;
    std::array<int, CITIES> visited;
    std::array<int, CITIES> path;
    float tourLength;
};

// CPU
std::vector<std::vector<float>> distances(CITIES, std::vector<float>(CITIES, 0.0f)); // Distance between city i and j
std::vector<std::vector<double>> hormone(CITIES, std::vector<double>(CITIES, INIT_PHER)); // Hormone between city i and j
std::vector<Ant> ants(ANTS);
std::vector<float> bestdistance(ANTS);
float finalbest = MAX_TOTAL_DISTANCE;

// GPU
float* distances_d;
Ant* ants_d;
double* hormone_d;
float* bestdistance_d;
curandState* state_d;
int BLOCKS, THREADS;

void get_distances_matrix();
void deviceAlloc();
__global__ void initialize_ants(Ant* ants_d, curandState* state_d, float* bestdistance_d, int THREADS);
__global__ void setup_curand_states(curandState* state_d, unsigned long t, int THREADS);
__global__ void restart_ants(Ant* ants_d, curandState* state_d, float* bestdistance_d, int THREADS);
void move_ants();
__global__ void simulate_ants(Ant* ants_d, curandState* state_d, float* distances_d, double* hormone_d, int THREADS);
__device__ double antProduct(int from, int to, double* hormone_d, float* distances_d);
__device__ int NextCity(Ant* ants_d, int pos, float* distances_d, double* hormone_d, curandState* state_d);
void updateTrails();

void get_distances_matrix() {
    int i, j;
    float k;

    while (scanf("%i %i %f", &i, &j, &k) == 3) {
        distances[i][j] = k;
        hormone[i][j] = INIT_PHER;
    }
}

void deviceAlloc() {
    cudaMalloc((void**)&ants_d, sizeof(Ant) * ANTS);
    cudaMalloc((void**)&state_d, sizeof(curandState) * ANTS);

    cudaMalloc((void**)&distances_d, sizeof(float) * CITIES * CITIES);
    cudaMemcpy(distances_d, distances.data(), sizeof(float) * CITIES * CITIES, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&hormone_d, sizeof(double) * CITIES * CITIES);
    cudaMemcpy(hormone_d, hormone.data(), sizeof(double) * CITIES * CITIES, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&bestdistance_d, sizeof(float) * ANTS);
}

// The rest of the functions remain unchanged

int main() {
    // ... (unchanged)
}

// The rest of the functions remain unchanged
