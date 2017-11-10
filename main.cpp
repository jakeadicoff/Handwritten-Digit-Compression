#include "main.h"

using namespace std;

int main(int argc, char** argv) {

  if(argc != 8) {
    cout << "ERROR: Missing required parameters \n"
	 << "population_size\n"
	 << "selection_type\n"
	 << "crossover_type\n"
	 << "crossover_probability\n"
	 << "mutation_probability\n"
	 << "num_generations\n"
	 << "num_symbols"
	 << endl;
    exit(1);
  }


  int population_size = stoi(argv[1]);
  string selection_type = argv[2];
  string crossover_type = argv[3];
  double crossover_probability = stod(argv[4]);
  double mutation_probability = stod(argv[5]);
  int num_generations = stoi(argv[6]);
  int num_symbols = stoi(argv[7]);

  Problem train = read_file("optdigits-32x32.tra");
  Problem test = read_file("optdigits-32x32.tes");


  NN_Parameters nn_params;
  nn_params.learning_rate = .01;
  nn_params.train_prob = train;
  nn_params.test_prob = test;
  nn_params.num_outputs = 10;
  nn_params.max_epochs = 10;

  string run_command;
  cout << "Run all the tests? (y/n)" << endl;
  cin >> run_command;


  Result results;

  if(run_command == "n") {
    GA rat_king(population_size, selection_type, crossover_type,
		crossover_probability, mutation_probability,
		num_generations, num_symbols, nn_params);

    results = rat_king.runGA();
  }

  else {
    string nn_filename;
    ofstream output_file;
    cout << "name output file (with .csv): " << endl;
    cin >> nn_filename;

    string output_string = "";
    output_string += "Command line:\n";
    for(int i = 0; i < argc; i++){
      output_string += argv[i];
      output_string += " ";
    }
    output_string += "\nRat King\ngeneration,num_symbols,"
      "mutation_probability,num_correct,run_time\n";

    //    vector<double> symbol_vector {8, 16, 32, 64};

    //    for(int i = 0; i < symbol_vector.size(); ++i) {
    // run it four times
    for(int i = 0; i < 4; ++i) {
      //      num_symbols = symbol_vector[i];
      GA rat_king(population_size, selection_type, crossover_type,
		  crossover_probability, mutation_probability,
		  num_generations, num_symbols, nn_params);

      results = rat_king.runGA();

      for(int gen = 0; gen < results.num_correct.size(); ++gen) {
	output_string += to_string(gen) + "," + to_string(num_symbols) + "," +
	  to_string(mutation_probability) + "," + to_string(results.num_correct[gen]) +
	  "," + to_string(results.run_time) + "\n";
      }

      output_string += "\n\n";
    }

    output_file.open(nn_filename);
    output_file << output_string;
    output_file.close();

  }
  return 0;
}

// reads in sets of digit recognition problems from files
Problem read_file(string problem_file_name) {

  Problem digit_recognition;
  digit_recognition.map_size = 0;
  digit_recognition.num_inputs = 0;
  vector<int> dummy_inputs;
  digit_recognition.inputs.push_back(dummy_inputs);

  string curr_number = "";
  char curr_char;

  ifstream problem_stream;
  problem_stream.open(problem_file_name.c_str(), ios::in);

  if(!problem_stream.good()) {
    cout << "Error: not able to open file" << endl;
  }

  // read the first line and determine map size
  if(problem_stream.peek() == 'B') {
    for(int i = 0; i < 3; i++){
      problem_stream.ignore(999999999,'\n');
    }
    int inputs_index = 0;
    while(problem_stream.peek() != EOF) {
      for(int i = 0; i < 32; i++) {
	for(int j = 0; j < 32; j++) {
	  problem_stream.get(curr_char);
	  curr_number =  curr_char;
	  digit_recognition.inputs[inputs_index].push_back(stoi(curr_number));
	}
	problem_stream.get(curr_char); // get rid of \n char
      }
      // get rid of ' ' and set target
      problem_stream.get(curr_char);
      problem_stream.get(curr_char);
      curr_number = curr_char;
      digit_recognition.targets.push_back(stoi(curr_number));
      problem_stream.get(curr_char);
      inputs_index++;
      vector <int> dummy_vector;
      digit_recognition.inputs.push_back(dummy_vector);
    }

    digit_recognition.map_size = 32*32;
    digit_recognition.num_inputs = inputs_index;
    return digit_recognition;


  } else {
    while(problem_stream.peek() != '\n') {
      problem_stream.get(curr_char);

      // if comma, save number and move on
      if(curr_char == ',') {
	digit_recognition.inputs[digit_recognition.num_inputs].push_back(stoi(curr_number));
	curr_number.clear();
	++digit_recognition.map_size;
      }

      // else, add char to curr number string
      else {
	curr_number += curr_char;
      }
    }

    while(problem_stream.peek() != EOF) {

      problem_stream.get(curr_char);

      // if comma, save number and move on
      if(curr_char == ',') {
	digit_recognition.inputs[digit_recognition.num_inputs].push_back(stoi(curr_number));
	curr_number.clear();
      }

      // if new line, cap off current digit image and move on to next one
      else if(curr_char == '\n') {
	digit_recognition.targets.push_back(stoi(curr_number));
	curr_number.clear();
	++digit_recognition.num_inputs;
	//cout << digit_recognition.num_inputs << endl;
	digit_recognition.inputs.push_back(dummy_inputs);
      }

      // else, add char to curr number string
      else {
	curr_number += curr_char;
      }
    }

    problem_stream.close();

    return digit_recognition;
  }

}
