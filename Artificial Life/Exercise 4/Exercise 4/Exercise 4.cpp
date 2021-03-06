// Exercise 4.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"


void PPAI(double *x, double *y, double *params, int leng) {

  for (int i = 1; i < leng; i++) {
    x[i] = x[i - 1] + params[0] * x[i - 1] + params[1] * y[i - 1] + params[2] * x[i - 1] * x[i - 1];
    y[i] = y[i - 1] + params[3] * x[i - 1] + params[4] * y[i - 1] + params[5] * y[i - 1] * y[i - 1];;
  }

}


int main()
{
  double x[100];
  double y[100];

  for (int i = 0; i < 100; i++) {
    x[i] = 0.0;
    y[i] = 0.0;
  }

  x[0] = 10.0;
  y[0] = 5.0;

  double params[6];
  params[0] = 1;
  params[1] = -20;
  params[2] = 1;
  params[3] = 0.2;
  params[4] = 0;
  params[5] = 0.5;


  PPAI(x, y, params, 100);


    return 0;
}

