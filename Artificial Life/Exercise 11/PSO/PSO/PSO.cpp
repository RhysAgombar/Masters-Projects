// PSO.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <math.h>
#include <random>
#include <time.h>
#include <stdio.h>
#include <tchar.h>

using namespace std;

const int P = 10;
const int W = 9;

class Particle {
public:
  float w[W]; // velocity
  float pcp[W]; // personal current position
  float pb = 0.0; // personal best value
  float pbp[W]; // personal best position
};

class Group {
public:
  Particle p[P]; // Population of particles
  float gb = 0.0; // group best value
  float gbp[W]; // group best position
};

float gb = 0.0; // global best value
float gbp[W]; // global best position

////////////////////////////////////////////////////////////////////////
// These are our helper functions for dealing with array arithmetic
////////////////////////////////////////////////////////////////////////
float* arraySub(float* a1, float* a2, int size) {
  float* out = new float[size];
  for (int i = 0; i < size; i++) {
    out[i] = a1[i] - a2[i];
  }
  return out;
}

float* arrayAdd(float* a1, float* a2, int size) {
  float* out = new float[size];
  for (int i = 0; i < size; i++) {
    out[i] = a1[i] + a2[i];
  }
  return out;
}

float* arrayMult(float a1, float* a2, int size) {
  float* out = new float[size];
  for (int i = 0; i < size; i++) {
    out[i] = a1 * a2[i];
  }
  return out;
}
////////////////////////////////////////////////////////////////////////

float f(float z) {
  float out = 1 / (1 + exp(-z));
  return out;
}

float G(float* w)
{
  float out = 0.0;

  out = pow((0.0 - f(w[6] + w[7] * f(w[0]) + w[8] * f(w[3]))), 2);
  out += pow((1.0 - f(w[6] + w[7] * f(w[0] + w[1]) + w[8] * f(w[3] + w[4]))), 2);
  out += pow((1.0 - f(w[6] + w[7] * f(w[0] + w[2]) + w[8] * f(w[3] + w[5]))), 2);
  out += pow((0.0 - f(w[6] + w[7] * f(w[0] + w[1] + w[2]) + w[8] * f(w[3] + w[4] + w[5]))), 2);

  return out;
}

float R() { // Random number function
  float out = rand() % 100;
  out = out / 100.0;
  return out;
}

Group updateV(Group Gr, float omega, float alpha, float beta, float gamma) { // Algorithm from Slide 55

  for (int i = 0; i < P; i++) {

    float* holder1 = arrayMult(omega, Gr.p[i].w, W);
    float* holder2 = arrayMult(alpha * R(), arraySub(Gr.p[i].pbp, Gr.p[i].pcp, W),W);
    float* holder3 = arrayMult(beta * R(), arraySub(gbp, Gr.p[i].pcp, W), W);
    float* holder4 = arrayMult(gamma * R(), arraySub(Gr.gbp, Gr.p[i].pcp, W), W);

    float* holder0 = arrayAdd(arrayAdd(arrayAdd(holder1, holder2, W), holder3, W), holder4, W);

    for (int j = 0; j < W; j++) {
      Gr.p[i].w[j] = holder0[j];
    }
  }

  return Gr;

}

Group init(Group Gr) {
  gb = 999;  // Minimizing, so global, group and personal bests are all set to high numbers
  Gr.gb = 999;
  
  for (int i = 0; i < P; i++) {
    Gr.p[i].pb = 999;
    for (int j = 0; j < W; j++) {

      float rn1 = (R() * 10) - 5;

      Gr.p[i].pcp[j] = rn1; // Initial starting position is between +,-5 in all w

      float rn2 = (R() * 10) - 5;

      Gr.p[i].w[j] = rn2; // random starting velocities of the same magnitude as above
    }
  }

  return Gr;
}

Group processG(Group Gr) {
  for (int i = 0; i < P; i++) {

    float out = G(Gr.p[i].w);

    if (out < gb) { // if the current output of G is less than the global best, update the global best and global best position
      gb = out;
      for (int j = 0; j < W; j++) {
        gbp[j] = Gr.p[i].pcp[j];
      }
    }

    if (out < Gr.gb) { // "" group best, group best position ""
      Gr.gb = out;
      for (int j = 0; j < W; j++) {
        Gr.gbp[j] = Gr.p[i].pcp[j];
      }
    }

    if (out < Gr.p[i].pb) { // "" personal best, personal best position ""
      Gr.p[i].pb = out;
      for (int j = 0; j < W; j++) {
        Gr.p[i].pbp[j] = Gr.p[i].pcp[j];
      }
    }
  }

  return Gr;
}

Group updatePos(Group Gr) {
  for (int i = 0; i < P; i++) {
    float* holder = arrayAdd(Gr.p[i].pcp, Gr.p[i].w, W);
    for (int j = 0; j < W; j++) {
      Gr.p[i].pcp[j] = holder[j];
    }
  }

return Gr;
}

const int numGroups = 25;
int main()
{
  
  srand(time(NULL));
  Group Gr[numGroups];

  for (int i = 0; i < numGroups; i++) {
    Gr[i] = init(Gr[i]);
  }


  for (int count = 0; count < 100; count++) {
    for (int i = 0; i < numGroups; i++) {
      Gr[i] = processG(Gr[i]);
      Gr[i] = updatePos(Gr[i]);
      Gr[i] = updateV(Gr[i], 0.4, 1.5, 1.5, 0.5);

    }
    printf("Iteration %3d: Minimal G Value = %f \n", count, gb);

  }
  printf("-------------------\n");
  printf("Minimal G Value = %f \n", gb);
  printf("W = \n");
  for (int i = 0; i < W; i++) {
    printf("%f\n",gbp[i]);
  }

  printf("-------------------\n");
  printf("The minimal value of G that can be reached fluctuates.\nIt seems there are local minima that exist and prevent the algorithm from reaching the lowest possible value.\nThrough multiple experiments, the lowest possible value seems to be 0.");

  return 0;
}

