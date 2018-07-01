#include <stdlib.h>  
#include <time.h>      
#include <math.h>
#include <iostream>
#include "Exercise5.h"


int iterations = 0; 
int L = 3; // For simplicity's sake, xyz-coordinates

void Parent_Selection() { // Unused, but written anyways in accordance with the instructions
  //Inheritance();
}

void Inheritance(population* p, int pop, int maxIndex) {

  for (int i = 0; i < pop; i++) { // overwrite every individual in the population with the individual's data with the highest fitness in previous step. Makes the math easy in the Mutation step.
    if (i != maxIndex) {
      for (int j = 0; j < L; j++) {
        p[i].g[j] = p[maxIndex].g[j];
      }
      p[i].fitness = p[maxIndex].fitness;
    }
  }

  Mutation(p, pop);
}

void Mutation(population* p, int pop) {

  // Since every individual in the population is a clone of the parent
  // Mutation skips first in population, designating that the 'parent' and mutates all the others.
  for (int i = 1; i < pop; i++) { 
      for (int j = 0; j < L; j++) {
        p[i].g[j] = p[i].g[j] + rng();
      }
  }

  Fitness_Eval(p, pop);
}

void Fitness_Eval(population* p, int pop) {

  for (int i = 0; i < pop; i++) {
    double fitness = (-1)*pow((p[i].g[0] - 6),2) + (-1)*pow((p[i].g[1] + 3),2) + (-1)*pow((p[i].g[2] - 4),2) + 50;  // Fitness function was not specified in the assignment, so I made my own.
    p[i].fitness = fitness;
  }

  std::cout << "Iteration: " << iterations << "\n";
  std::cout << "P[0] Fitness: " << p[0].fitness << "\n";
  std::cout << "P[1] Fitness: " << p[1].fitness << "\n";

  External_Selection(p, pop);
}

void External_Selection(population* p, int pop) {

  int maxIndex = 0; 
  for (int i = 0; i < pop; i++) {
    if (p[maxIndex].fitness < p[i].fitness) { // find the individual in the population with the highest fitness...
      maxIndex = i;
    }
  }


 // Check if finished, if not,
  if (iterations < 500) { // 500 iterations should be enough to get reasonably close to the optimal solution
    iterations++;
    //Parent_Selection(); Omitted as per assignment, but has a function definition anyway for future use
    Inheritance(p, pop, maxIndex);
  }
  else {
    std::cout << "----------- Finished -----------\n";
    std::cout << "Final Fitness: " << p[maxIndex].fitness << "\n";
    std::cout << "Maximized Genome: P[" << maxIndex << "] = ";
    for (int i = 0; i < L; i++) {
      std::cout << p[maxIndex].g[i] << " ";
    }
    std::cout << "\n";
  }
  

}

int rng() {
  return (rand() % 20)-10; // Random numbers between -10 and 10. This is used for the mutation step, and for initial values.
}

void init(population *p, int initPop) {
  srand(time(NULL));

  for (int i = 0; i < initPop; i++) {
    p[i].g = new int[L];
    p[i].fitness = 0;
    for (int j = 0; j < L; j++) {
      p[i].g[j] = rng(); // Set initial population genome to random values
    }
  }
    
}

int main()
{
  int initPop = 2;
  population* p = new population[initPop];

  init(p, initPop);

  Fitness_Eval(p, initPop);

  // cleaning up pointers after all is said and done.
  for (int i = 0; i < initPop; i++) {
    delete p[i].g;
  }
  delete p;

  return 0;
}