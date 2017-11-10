#ifndef __NN_h
#define __NN_h

#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <string>
using namespace std;

struct output {
    vector<double> weights;
};

struct Problem {

    vector<vector<int>> inputs;
    vector<int> targets;
    int num_inputs;
    int map_size;

};



class NN {
public:
    NN(double learningRate, Problem train_prob, Problem test_prob,
       int numOutputs, int maxEpochs, int numSymbols);
    double test();
    void train();
    vector<int> compression_vector;

private:
    int num_train_inputs;
    int num_test_inputs;
    int num_symbols;
    int num_outputs;
    int map_size;
    int max_epochs;
    double learning_rate;

    vector<vector<int>> train_inputs;
    vector<vector<int>> test_inputs;
    vector<vector<int>> compressed_train_inputs;
    vector<vector<int>> compressed_test_inputs;
    vector<int> train_targets;
    vector<int> test_targets;
    vector<output> outputs;
    

    void initialize_weights();
    void reset();
    void compress_maps();
    void update_weights(int output_index, int input_index, double g, double g_prime, double target);
    double activation_function(double x);
    double ddx_activation_function(double x);


};

#endif
