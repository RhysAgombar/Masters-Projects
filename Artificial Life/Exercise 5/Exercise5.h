#pragma once

class population {
public:
  int* g;
  int fitness;
};

void Parent_Selection();
void Inheritance(population* p, int pop);
void Mutation(population* p, int pop);
void Fitness_Eval(population* p, int pop);
void External_Selection(population* p, int pop);
int rng();
void init(population* p, int L);
