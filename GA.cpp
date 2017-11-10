#include "GA.h"

#include <ctype.h>
#include <algorithm>
#include <iostream>
#include <random>
#include <cmath>
#include <time.h>
#include <string>
#include <algorithm>
#include <set>

using namespace std;

// GA constructor with initializer list
GA::GA(int populationSize, string selectionType, string crossoverType,
       double crossoverProbability, double mutationProbability,
       int generationNumber, int numSymbols, NN_Parameters nnParameters) :
net(nnParameters.learning_rate, nnParameters.train_prob,
    nnParameters.test_prob, nnParameters.num_outputs,
    nnParameters.max_epochs, numSymbols)
{

    srand(time(NULL)); //seed rand

    this->population_size = populationSize;
    this->crossover_probability = crossoverProbability;
    this->crossover_points = 20; // For N-Point Crossover
    this->mutation_probability = mutationProbability;
    this->generations = generationNumber;
    this->best_generation = 0;
    this->map_size = 1024; // HARDCODED: 32x32 INPUT
    this->num_symbols = numSymbols;
    this->nn_parameters = nnParameters;

    this->population = generate_initial_population();

    if(selectionType == "ts" ) {
        this->selection_type = TOURNAMENT;
    } else if(selectionType == "bs") {
        this->selection_type = BOLTZMANN;
    } else if(selectionType == "rs") {
        this->selection_type = RANK;
    } else {
        cout << "Selection type not recognized" << endl;
        exit(1);
    }

    if(crossoverType == "uc") {
        this->crossover_type = UNIFORM;
    } else if(crossoverType == "1c") {
        this->crossover_type = ONEPOINT;
    } else if(crossoverType.substr(0,2) == "nc") {
        this->crossover_type = NPOINT;
	try {
	  this->crossover_points = stoi(crossoverType.substr(2));
	} catch(std::invalid_argument& e){
	  cerr << "Invalid n-point crossover parameter.\nTo call "
	    "N-Point crossover, append an integer to the end of "
	    "'nc':\n ie: nc42 is 42 point crossover."  << endl;
	  exit(1);
	}
    } else {
        cout << "Crossover type not recognized" << endl;
        exit(1);
    }
}

Result GA::runGA() {
    Result results;
    start_time = clock();
    srand(time(NULL));

    int streak = 0;
    double og_mutation = mutation_probability;

    Individual best_overall_individual = population[0];
    Individual best_after_update;

    for(int i = 0; i < generations; i++) {
        fitness();  // evaluate fitness of population

        ++streak;
        mutation_probability = 0.5 * (mutation_probability - og_mutation) + og_mutation;

        // Re-add the best individual from the last generation (only
        // do this after there IS a "last generation)
        if(i > 0) elitism(best_after_update);

        srand(time(NULL)); // re-seed rand, because NN uses a fixed seed
        best_after_update = get_best();

        //print every 50, so we dont clog up the terminal
        if(i % 50 == 0) print_individual(best_after_update);
        cout << endl << "best " <<  best_after_update.number_correct << endl;


        // for measuring which generation produce the best Individual
        if(best_after_update.number_correct > best_overall_individual.number_correct) {
            best_overall_individual = best_after_update;
            best_generation = i+1;
            streak = 0;
        }
        //cout << "3" << endl;
        // perform selection
        switch(selection_type) {
            case TOURNAMENT:
                tournament_selection();
                break;
            case BOLTZMANN:
                boltzmann_selection();
                break;
            case RANK:
                rank_selection();
                break;
        }

        for(int j = 0; j < population_size-1; j++) { //size of breeding population
            Individual ind;

            switch(crossover_type) {
                case UNIFORM:
                    ind = uniform_crossover(breeding_population[2*j],
                                            breeding_population[2*j+1]);
                    break;
                case NPOINT:
                    ind = n_point_crossover(crossover_points, breeding_population[2*j],
                                            breeding_population[2*j+1]);
                    break;
                case ONEPOINT:
                    ind = one_point_crossover(breeding_population[2*j],
                                              breeding_population[2*j+1]);
                    break;
            }

            population[j] = ind;

        }//for pop
        if(streak > 5) {
            mutation_probability = og_mutation * 8;
        }
        mutation();
        results.num_correct.push_back(best_overall_individual.number_correct);
    }//for gen
    end_time = clock();
    extract_and_print_answer(best_overall_individual);
    results.run_time = (end_time-start_time)/CLOCKS_PER_SEC;
    results.best_compression_vector = best_overall_individual.compression_vector;
    return results;
}


// void GA::debug_print() {
//   cout << endl << endl;
//   fitness();
//   for(int i = 0; i < population_size; i++) {
//     for(int j = 0; j < map_size+1; j++) {
//       cout << population[i].compression_vector[j];
//     }
//     cout << "   " << population[i].number_correct << endl;
//   }
// }

void GA::print_individual(Individual ind) {
    vector <int> compression_vector = ind.compression_vector;
    for(int i = 0; i < 32; i++) {
        for(int j = 0; j < 32; j++) {
            cout << compression_vector[i*32 + j] << "\t";
        }
        cout << endl;
    }
}
void GA::extract_and_print_answer(Individual best_individual) {

    fitness();//get final fitness update
    Individual ind = get_best();
    if(ind.number_correct > best_individual.number_correct) {
        best_individual = ind;
        best_generation = generations;
    }

    int num_train_inputs = nn_parameters.test_prob.inputs.size();
    cout << " Number of symbols: " << num_symbols << endl;
    cout << " Number of tests: " << nn_parameters.test_prob.inputs.size() << endl;
    cout << " Number of tests satisfied: " << best_individual.number_correct << endl;
    cout << " Percentage of tests satisfied: " << 100.0*best_individual.number_correct/(num_train_inputs*1.0)<< "%" << endl;
    cout << " GA result: " << endl;
    print_individual(best_individual);
    cout << endl;
    cout << " Generation where best assignment was found: " << best_generation << endl;
    cout << " Time to solve: " << (end_time-start_time)/CLOCKS_PER_SEC << endl;
    cout << endl;


}


/*
 * Gets the CURRENT best individual from the population
 */
Individual GA::get_best() {
    int index_of_fittest = 0;
    for(int i = 0; i < population_size; i++) {
        if(population[index_of_fittest].number_correct < population[i].number_correct)
            index_of_fittest = i;
    }
    return population[index_of_fittest];
}

vector <Individual> GA::generate_initial_population() {


    vector<Individual> population; // to return

    for(int i = 0; i < population_size; i++) {
        vector<int> compressionVector;
        for(int j = 0; j <= map_size; j++) {
            int random;
            random = rand() % num_symbols;
            compressionVector.push_back(random);

        }
        Individual ind;
        ind.number_correct = -1;
        ind.compression_vector = compressionVector;
        population.push_back(ind);
    }
    return population;
}

void GA::fitness() {


    // for every Individual in the population
    for(int i = 0; i < population_size; i++) {


        net.compression_vector = population[i].compression_vector;

        net.train();

        population[i].number_correct = net.test();
    }//for pop

}//func

/*
 * STILL NOT GREAT FOR LARGE FITNESS VALUES
 */
void GA::boltzmann_selection() {

    breeding_population.clear();
    vector<long double> boltzmann_weights;  //declare size equal to popultion size
    long double boltzmann_sum = 0;  // DISCRETE DISTRUBTION RANDOM GENERATOR

    // exponentiate each fitness, that is the individual's weight
    for (int i = 0; i < population_size; ++i) {
        //1800 is approximate max fitness, 180 is 10% of that
        double scaled_fitness = double(population[i].number_correct)/double(180);
        long double k = expl(scaled_fitness);
        boltzmann_weights.push_back(k);
        boltzmann_sum += k;
        //    cout << "individual boltz weight is " << k << endl;
    }

    // select individual, using weights above
    for(int i = 0; i < population_size * 2; ++i) {
        // gen random value between 0 and total boltzmann_sum
        long double random_weight;
        random_weight = boltzmann_sum*double(rand())/RAND_MAX;
        //cout << "random weight is " << random_weight << endl;


        // get index corresponding to random_weight, by subtracting
        // successive boltzmann weights off of random_weight until
        // random_weight is 0
        int boltzmann_index = 0;
        while(random_weight > 0) {
            random_weight -= boltzmann_weights[boltzmann_index];
            boltzmann_index++;

        }
        boltzmann_index--;  //because we went one past

        //cout << "boltzmann selected " << boltzmann_index << endl;
        breeding_population.push_back(population[boltzmann_index]);
    }
}


void GA::tournament_selection() {
    breeding_population.clear();

    int first_random_index;
    int second_random_index;

    for(int i = 0; i < population_size*2; ++i) {

        // get two random indices
        first_random_index = rand() % (population_size-1);
        second_random_index = rand() % (population_size-1);
        if(population[first_random_index].number_correct > population[second_random_index].number_correct) {
            breeding_population.push_back(population[first_random_index]);
        }
        else {
            breeding_population.push_back(population[second_random_index]);
        }
    }
}


// comparison overload fucntion for Individuals
bool compare_individual_satisfication(Individual a, Individual b){
    return a.number_correct < b.number_correct;
}


void GA::rank_selection() {

    //cout << "in rs " << endl;
    breeding_population.clear();

    // sorts the population in ascending order
    sort(population.begin(), population.end(), compare_individual_satisfication);
    // sum from 1 to <population size>
    int gaussian_sum = population_size * (population_size + 1) / 2;

    // weight probabilities by index from least
    for(int i = 0; i < population_size * 2; i++) {
        int random_number;
        //Generate a random value from 1 to <population size>
        random_number = 1+rand() % (gaussian_sum);

        // Find the index that corresponds to that random value rank
        int index = ceil((-1 + sqrt(1 + 8 * random_number)) / 2)-1;

        breeding_population.push_back(population[index]);
    }

}

void GA::mutation() {
    for(int i = 0; i < population_size; i++) { //all Individuals
        for(int j = 0; j < map_size; j++) { //first bit is trash
            double random; // = distribution(generator);
            random = double(rand())/RAND_MAX;
            if(random <= mutation_probability) { // if rand is less than prob[0,1]
                int random_symbol = rand() % num_symbols;
                population[i].compression_vector[j] = random_symbol;
            }
        }
    }
}

void GA::elitism(Individual best_individual) {
    //find the weakest individual to replace with our elite individual
    int index_weakest = 0;
    for(int i = 0; i < population_size; i++) {
        if(population[index_weakest].number_correct > population[i].number_correct) {
            index_weakest = i;
        }
    }

    //replace the worst individual with our elite individual
    population[index_weakest].number_correct = best_individual.number_correct;
    population[index_weakest].compression_vector = best_individual.compression_vector;
}

Individual GA::one_point_crossover(Individual parent_a, Individual parent_b) {

    // for crossover probability -- determines if crossover is even going to happen
    double random_number = double(rand()) / RAND_MAX;

    Individual new_Individual;
    new_Individual.number_correct = -1;

    // DO crossover
    if(random_number < crossover_probability) {
        int random_index;
        random_index = rand() % map_size;
        for(int i = 0; i < random_index; ++i) {
            new_Individual.compression_vector.push_back(parent_a.compression_vector[i]);
        }
        for(int j = random_index; j < parent_a.compression_vector.size(); ++j) {
            new_Individual.compression_vector.push_back(parent_b.compression_vector[j]);
        }
    } else { // don't crossover at all
        new_Individual.compression_vector = parent_a.compression_vector;
    }

    return new_Individual;
}

Individual GA::n_point_crossover(int points,
                                 Individual parent_a, Individual parent_b) {
    // for crossover probability -- determines if crossover is even going to happen
    double random_number = double(rand()) / RAND_MAX;
    set<int> crossover_points;
    set<int>::iterator iter;

    Individual new_individual;
    new_individual.number_correct = -1;

    // do crossover
    if(random_number < crossover_probability) {
        // Step 1: create crossover index set
        for(int i = 0; i < points; i++) {
            do {
                int random_index = rand() % map_size;

                // iter will point to the MySet.end() if rand_index NOT in set
                iter = crossover_points.find(random_index);

                //if the index is NOT in the xover_points set, add the point
                if( iter == crossover_points.end())
                    crossover_points.insert(random_index);

                // loop while we can find the index in our xover points set:
                // iter will NOT point to the end if we find() a match
            } while(iter != crossover_points.end());
        }

        // Step 2: alternate grabbing segments of genes from each parent
        set<int>::iterator end = crossover_points.end();
        bool toggle = false;
        Individual active_parent = parent_a;
        iter = crossover_points.begin();

        for(int i = 0; i < map_size; i++) {
            new_individual.compression_vector.push_back(
                                                        active_parent.compression_vector[i]);

            // if current index is a crossover point, and we still have
            // crossover points to hit, toggle parent
            if(i == *iter && iter != end) {
                if(toggle == false) {
                    active_parent = parent_b;
                }
                else {
                    active_parent = parent_a;
                }
                toggle = !toggle;
                iter++;
            }
        }
    }

    else { // don't crossover at all
        new_individual.compression_vector = parent_a.compression_vector;
    }

    return new_individual;
}


Individual GA::uniform_crossover(Individual parent_a, Individual parent_b) {
    //cout << "in uc " << endl;
    ;
    // initialize new Individual
    Individual new_Individual;
    new_Individual.number_correct = -1;
    double random1;
    random1 = double(rand())/RAND_MAX;
    // crossover
    if(random1 < crossover_probability) {
        for(int i = 0; i < parent_a.compression_vector.size(); ++i) {// for parent selection
            // pick parent
            int random2;
            random2 = rand() % 2;
            if(random2 == 0) {
                new_Individual.compression_vector.push_back(parent_a.compression_vector[i]);
            } else {
                new_Individual.compression_vector.push_back(parent_b.compression_vector[i]);
            }
        }
    } else { // don't crossover
        new_Individual.compression_vector = parent_a.compression_vector;
    }
    return new_Individual;
}
