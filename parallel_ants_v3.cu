#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <curand_kernel.h>

//Problem Parameters
constexpr unsigned int CITIES = 100;
constexpr unsigned int ANTS = 3000;
constexpr double MAX_DIST = 100;
constexpr int ALPHA = 1;
constexpr int BETA = 5; 		//This parameter raises the weight of distance over pheromone
constexpr double RHO = 0.5;	//Evapouration rate
constexpr int QVAL = 100;
constexpr unsigned int MAX_TOURS = 50;	// The number of times an ant will walk trough all the cities
constexpr double INIT_PHER = 1.0 / CITIES;	// Initial hormone for each path
constexpr double MAX_TOTAL_DISTANCE = CITIES * MAX_DIST; // MAX possible distance that an ant can walk

static void HandleError( cudaError_t err,
						 const char *file,
						 int line) {
	if(err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit( EXIT_FAILURE );
	}
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__,__LINE__ ))

struct ant{
  int curCity, nextCity, pathIndex;
  int visited[CITIES];
  int path[CITIES];
  float tourLength;
};

//CPU
float        distances[CITIES][CITIES]; // Distance between city i an j
double       hormone[CITIES][CITIES]; //Hormone between city i and j
struct ant   ants[ANTS];
float        bestdistance[ANTS];
float 		 finalbest = (float)MAX_TOTAL_DISTANCE;
curandState  state[ANTS];
const size_t distances_size = sizeof(float) * size_t(CITIES*CITIES);
const size_t hormone_size = sizeof(double) * size_t(CITIES*CITIES);
int          bestPath[CITIES];
//GPU
float        *distances_d;
struct ant   *ants_d;
double       *hormone_d;
float        *bestdistance_d;
curandState  *state_d;
const int THREADS = 256;
const int BLOCKS = (ANTS + THREADS - 1) / THREADS;
const char *FILENAME = "map100.txt";

void get_distances_matrix();
void get_distances_matrix_from_file(const char *filename);
void deviceAlloc();
__global__ void initialize_ants(struct ant *ants_d, curandState *state_d, float *bestdistance_d , int THREADS);
__global__ void setup_curand_states(curandState *state_d, unsigned long t , int THREADS);
__global__ void restart_ants(struct ant *ants_d,curandState *state_d, float *bestdistance_d , int THREADS);
void move_ants();
__global__ void simulate_ants(struct ant *ants_d,curandState *state_d, float *distances_d, double *hormone_d, int THREADS);
__forceinline__ __device__ double antProduct(int from, int to, double *hormone_d, float *distances_d);
__forceinline__ __device__ int NextCity(struct ant *ants_d, int pos, float *distances_d, double *hormone_d, curandState *state_d );
void updateTrails();

int main(){

		float time1;
		cudaEvent_t start, stop;

		//get_distances_matrix(); // Get the distances between cities from the input
    get_distances_matrix_from_file(FILENAME);
		deviceAlloc(); // Mallocs and memcpy of the device variables

		//ovdje pocni mjerenje vremena izvrsavanja
		HANDLE_ERROR( cudaEventCreate(&start) );
		HANDLE_ERROR( cudaEventCreate(&stop) );
		HANDLE_ERROR( cudaEventRecord(start, 0) );

		//Set up an array of curand_states in order to build better random numbers
		time_t t; time(&t);
		setup_curand_states <<< BLOCKS, THREADS >>> (state_d, (unsigned long) t , THREADS);
		cudaThreadSynchronize();

		//initialize the ants array
		initialize_ants <<< BLOCKS, THREADS >>> (ants_d, state_d, bestdistance_d , THREADS);
		cudaThreadSynchronize();

		// Start and control the ants tours
		move_ants();

		HANDLE_ERROR( cudaEventRecord(stop, 0) );
		HANDLE_ERROR( cudaEventSynchronize(stop) );
		HANDLE_ERROR( cudaEventElapsedTime(&time1, start, stop) );

		printf("Time to generate:  %3.1f ms \n", time1);

		//Free Memory
		cudaFree(ants_d);
		cudaFree(bestdistance_d);
		cudaFree(distances_d);
		cudaFree(hormone_d);
		cudaFree(state_d);
		cudaFree(bestdistance_d);

		return 0;
}


void get_distances_matrix(){
  int i,j;
  float k;

  while(scanf("%i %i %f", &i,&j,&k) == 3){
    distances[i][j] = k;
    hormone[i][j] = INIT_PHER;
  }

}

void get_distances_matrix_from_file(const char *filename) {
		FILE *file = fopen(filename, "r");

    if (file == NULL) {
        printf("Error opening file %s\n", filename);
        return;
    }

    int i, j;
    float k;

    while (fscanf(file, "%i %i %f", &i, &j, &k) == 3) {
        distances[i][j] = k;
        hormone[i][j] = INIT_PHER;
    }

    fclose(file);
}

void deviceAlloc(){
	if (cudaMalloc( (void**) &ants_d, sizeof(ants)) != cudaSuccess) {
			printf("Error in allocating memory!");
			exit(1);
	}
	if (cudaMalloc( (void**) &state_d, sizeof(state)) != cudaSuccess) {
			printf("Error in allocating memory!");
			cudaFree(ants_d);
			exit(1);
	}
	if (cudaMalloc( (void**) &distances_d, distances_size) != cudaSuccess) {
			printf("Error in allocating memory!");
			cudaFree(ants_d);
			cudaFree(state_d);
			exit(1);
	}
	if (cudaMemcpy(distances_d, distances, distances_size, cudaMemcpyHostToDevice) != cudaSuccess) {
			printf("Error while memory copying!");
			cudaFree(ants_d);
			cudaFree(state_d);
		    cudaFree(distances_d);
			exit(1);
	}
	if (cudaMalloc( (void**) &hormone_d, hormone_size) != cudaSuccess) {
			printf("Error in allocating memory!");
			cudaFree(ants_d);
			cudaFree(state_d);
		    cudaFree(distances_d);
			exit(1);
	}
	if (cudaMemcpy(hormone_d, hormone, hormone_size, cudaMemcpyHostToDevice) != cudaSuccess) {
			printf("Error while memory copying!");
			cudaFree(ants_d);
			cudaFree(state_d);
		    cudaFree(distances_d);
		 	cudaFree(hormone_d);
			exit(1);
	}
	if (cudaMalloc( (void**) &bestdistance_d, sizeof(bestdistance)) != cudaSuccess) {
			printf("Error in allocating memory!");
			cudaFree(ants_d);
			cudaFree(state_d);
		    cudaFree(distances_d);
		 	cudaFree(hormone_d);
			exit(1);
	}
}

__global__ void setup_curand_states(curandState *state_d, unsigned long t, int THREADS){
	int id = threadIdx.x + blockIdx.x*THREADS;
	if(id < ANTS) curand_init(t, id, 0, &state_d[id]);
}

__global__ void initialize_ants(struct ant *ants_d, curandState *state_d, float *bestdistance_d , int THREADS){
  int position = threadIdx.x + blockIdx.x*THREADS;

  if(position < ANTS) {
      
    // Mark all cities as not visited
    // Mark all path as not traversed

    for(int k = 0; k < CITIES; k++){
      ants_d[position].visited[k] = 0;
      ants_d[position].path[k] = -1;
    }

    bestdistance_d[position] = (float)(MAX_TOTAL_DISTANCE);

    //Random City to begin
    ants_d[position].curCity = curand(&state_d[position])% CITIES;
    ants_d[position].pathIndex = 1;
    ants_d[position].path[0] = ants_d[position].curCity;
    ants_d[position].nextCity = -1;
    ants_d[position].tourLength = 0;
    ants_d[position].visited[ants_d[position].curCity] = 1;

  }
}

__global__ void restart_ants(struct ant *ants_d,curandState *state_d, float *bestdistance_d , int THREADS){

	int position = threadIdx.x + blockIdx.x*THREADS;

	if(position < ANTS){
		if(ants_d[position].tourLength < bestdistance_d[position]){
			bestdistance_d[position] = ants_d[position].tourLength;
		}

		ants_d[position].nextCity = -1;
		ants_d[position].tourLength = 0.0;

		for(int i = 0; i < CITIES; i++){
			ants_d[position].visited[i] = 0;
			ants_d[position].path[i] = -1;
		}

		ants_d[position].curCity = curand(&state_d[position])% CITIES;
		ants_d[position].pathIndex = 1;
		ants_d[position].path[0] = ants_d[position].curCity;
		ants_d[position].visited[ants_d[position].curCity] = 1;
	}
}

void move_ants(){
	int curtour = 0;
	while (curtour++ < MAX_TOURS){
		simulate_ants <<< BLOCKS, THREADS >>> (ants_d, state_d, distances_d, hormone_d, THREADS);
		cudaThreadSynchronize();

		cudaMemcpy(ants, ants_d, sizeof(ants), cudaMemcpyDeviceToHost);

		//update the trails of the ants
		int from,to,i,ant;

		//hormone evaporation
		for(from = 0; from < CITIES; from++)
			for(to = 0;to < CITIES; to++){
				if(from!=to){
					hormone[from][to] *=( 1.0 - RHO);

					if(hormone[from][to] < 0.0){
						hormone[from][to] = INIT_PHER;
					}
				}
			}

		// add new pheromone to the trails
		for(ant = 0; ant < ANTS; ant++)
			for(i = 0; i < CITIES; i++){
				if( i < CITIES - 1 ){
					from = ants[ant].path[i];
					to = ants[ant].path[i+1];
				}
				else{
					from = ants[ant].path[i];
					to = ants[ant].path[0];
				}
			
			hormone[from][to] += (QVAL/ ants[ant].tourLength);
			hormone[to][from] = hormone[from][to];
			}
			
			for (from = 0; from < CITIES; from++)
				for( to = 0; to < CITIES; to++){
					hormone[from][to] *= RHO;
				}
		
		cudaMemcpy(hormone_d, hormone, hormone_size, cudaMemcpyHostToDevice);
		cudaMemcpy(bestdistance, bestdistance_d, sizeof(bestdistance), cudaMemcpyDeviceToHost);
			
			for(i = 0; i < ANTS; i++)
				if(bestdistance[i] < finalbest){
					finalbest = bestdistance[i];
	
					for (int j = 0; j < CITIES; j++) {
					  bestPath[j] = ants[i].path[j];
					}
				}
				
				printf("Best distance %f \n", finalbest);
				printf("Best path: ");
	      for (int i = 0; i < CITIES; i++) {
    	    printf("%d ", bestPath[i]);
        }
        printf("\n");
		restart_ants <<< BLOCKS, THREADS >>> (ants_d, state_d, bestdistance_d, THREADS);
		cudaThreadSynchronize();
	}
}

__global__ void simulate_ants(struct ant *ants_d,curandState *state_d, float *distances_d,
								double *hormone_d , int THREADS ){

	int position = threadIdx.x + blockIdx.x*THREADS;
	int curtime = 0;

	if(position < ANTS){
		while(curtime++ < CITIES){
			//check if all cities were visited
			if( ants_d[position].pathIndex < CITIES ){

				ants_d[position].nextCity = NextCity(ants_d, position, distances_d, hormone_d, state_d);
				ants_d[position].visited[ants_d[position].nextCity] = 1;
				ants_d[position].path[ants_d[position].pathIndex++] = ants_d[position].nextCity;
				ants_d[position].tourLength += distances_d[ants_d[position].curCity + (ants_d[position].nextCity * CITIES)];
				if(ants_d[position].pathIndex == CITIES){
					ants_d[position].tourLength += distances_d[ants_d[position].path[CITIES -1] + (ants_d[position].path[0]*CITIES)];
				}
				ants_d[position].curCity = ants_d[position].nextCity;
			}
		}
	}
}

__forceinline__ __device__ double antProduct(int from, int to, double *hormone_d, float *distances_d){
    int index = from + to * CITIES;
    return __fmul_rn(__powf(hormone_d[index], ALPHA), __powf((1.0f / distances_d[index]), BETA));
}

__forceinline__ __device__ int NextCity(struct ant *ants_d, int pos, float *distances_d, double *hormone_d, curandState *state_d ){
	int to, from;
	double denom = 0.0;
	from =  ants_d[pos].curCity;

	for (to = 0; to < CITIES; to ++) {
		if(ants_d[pos].visited[to] == 0) denom += antProduct(from, to, hormone_d, distances_d);
	}

	assert(denom != 0.0);

	to++;
	int count = CITIES - ants_d[pos].pathIndex;

	do{
		double p;
		to++;

		if(to >= CITIES)
			to = 0;

		if(ants_d[pos].visited[to] == 0){
			p = __fdiv_rn(antProduct(from, to, hormone_d, distances_d), denom);
      double rem = curand(&state_d[pos]) % 100000000;
			double x = __fdiv_rn(rem, 100000000.0);
			if(x < p){
				break;
			}
			count--;
			if(count == 0){
				break;
			}
		}
	}while(1);

	__syncthreads();

	return to;
}