#include <cstdio>
#include <vector>
#include <random>
#include <chrono>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/fill.h>

#define MAX_DIM 100
#define MAX_N 2000000

struct saxpy {
    const int a;
    saxpy(int _a) : a(_a) { }
    __host__ __device__
    double operator()(const double &x, const int &y) const {
        //generate random number
        thrust::minstd_rand rng;
        thrust::uniform_real_distribution<double> dist(y-a,y+1);
        return dist(rng);
    }
};

int main(){
    
    //gpu
    printf("GPU START!\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    thrust::host_vector<double> H;
    H.resize(MAX_N*MAX_DIM);
    thrust::device_vector<double> D(H);
    
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    int ms = duration.count();
    printf("INIT USED: %d ms.\n", ms);
    start = std::chrono::high_resolution_clock::now();
    
    thrust::counting_iterator<int> first(0);
    for(int i=0; i<MAX_N; ++i){
        thrust::transform(D.begin()+i*MAX_DIM, D.begin()+i*MAX_DIM+MAX_DIM, first, D.begin(), saxpy(i));
    }
    
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    ms = duration.count();
    printf("COMPUTE USED: %d ms.\n\n", ms);
    
    //cpu
    printf("CPU START!\n");
    start = std::chrono::high_resolution_clock::now();
    
    std::vector<double> V;
    V.resize(MAX_N*MAX_DIM);
    
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    ms = duration.count();
    printf("INIT USED: %d ms.\n", ms);
    
    start = std::chrono::high_resolution_clock::now();
    
    for(int i=0; i<MAX_N; ++i){
        for(int j=0; j<MAX_DIM; ++j){
            //generate random number
            static thread_local std::random_device rd;
            static thread_local std::mt19937* generator = new std::mt19937( rd() );
            std::uniform_real_distribution<double> distribution(j-i, j+1);
            V[i*MAX_DIM+j] = distribution(*generator);
        }
    }
    
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    ms = duration.count();
    printf("COMPUTE USED: %d ms.\n\n", ms);
    std::vector<double>().swap(V);
    
    
    return 0;
}


