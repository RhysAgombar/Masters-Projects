// Exercise2.cpp : Defines the entry point for the console application.
//

#include <stdio.h>
#include <tchar.h>
#include <stdlib.h>
#include "time.h"

const int cellSize = 83;
const int seedCell = 42;
int cCells[cellSize], pCells[cellSize]; // Current cells and previous cells

int init(char condition, int r) {

  for (int i = 0; i < cellSize; i++) {
    cCells[i] = 0;
    pCells[i] = 0;
  }

  if (r != 0 && r != 1) {
    printf("Invalid value for r.\n");
    return 0;
  }

  if (condition == 'S' or condition == 's') {
    printf("Seed, r=%d\n", r);

    pCells[seedCell] = 1;
  }
  else if (condition == 'R' or condition == 'r') {
    printf("Random, r=%d\n", r);

    for (int i = 2; i < cellSize - 1; i++) { // Skips the first and last two cells to create a boundary.
      int chance = rand() % 10;
      if (chance < 5) {
        pCells[i] = 0;
      }
      else {
        pCells[i] = 1;
      }
    }

  }
  else {
    printf("Invalid starting condition.\n");
    return 0;
  }

  for (int i = 0; i < cellSize + 1; i++) {
    printf("%d ", pCells[i]);
  }

  printf("\n");

  return 1;

}

void step(int r) {

}

int main()
{
  srand(time(NULL));
  char cond = 'r';
  int r = 1;

  int res = init(cond, r);
  if (res == 0) {
    return 0;
  }

  for (int i = 0; i < 20; i++) {
    step(r);
  }

  return 0;
}

