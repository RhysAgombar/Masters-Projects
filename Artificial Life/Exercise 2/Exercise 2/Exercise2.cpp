// Exercise2.cpp : Defines the entry point for the console application.
//

#include <stdio.h>
#include <tchar.h>
#include <stdlib.h>
#include "time.h"

const int cellSize = 83;
const int seedCell = 42;
int cCells[cellSize], pCells[cellSize]; // Current cells and previous cells

int init(char condition) {

  for (int i = 0; i < cellSize; i++) {
    cCells[i] = 0;
    pCells[i] = 0;
  }

  if (condition == 'S' or condition == 's') {
    printf("Seed\n");

    pCells[seedCell] = 1;
  }
  else if (condition == 'R' or condition == 'r') {
    printf("Random\n");

    for (int i = 0; i < cellSize + 1; i++) {
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

int main()
{
  srand(time(NULL));
  char cond = 'r';

  int res = init(cond);
  if (res == 0) {
    return 0;
  }

  return 0;
}

