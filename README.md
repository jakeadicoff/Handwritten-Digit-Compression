# Handwritten Digit Compression
Final Project for NIC: GA / Perceptron hybrid for image compression and digit recognition.

After making the project, run

> ./compress

with the following parameters, in this order:

__Population:__
  [Positive integer]
  
  
__Selection Type:__
  [String]
  
  "bs" - Botlzmann Selection
  
  "ts" - Tournament Selection
  
  "rs" - Rank Selection // This is the best one
  
  
__Crossover Type:__
  [String]
  
  "uc" - Uniform Crossover
  
  "1c" - 1-Point Crossover
  
  "nc30" - 30-Point Crossover // Replace '30' with any positive integer for any N-Point crossover
  
  
__Crossover Probability:__
  [Double, between 0 and 1]
  
  
__Mutation Probability:__
  [Double, between 0 and 1]
  
  
__Number of Generations/Iterations:__
  [Positive integer]
  
  
__Number of Symbols:__
  [Positive integer]
  
  
Example command, pretty good results:

./compress 100 rs uc .5 .01 200 32
