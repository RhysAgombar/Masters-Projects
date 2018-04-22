// Exercise2.cpp : Defines the entry point for the console application.
//

#include <stdio.h>
#include <tchar.h>
#include <stdlib.h>
#include "time.h"
#include <string>
#include <bitset>
#include <cmath>

using namespace std;

const int cellSize = 83;
const int seedCell = 42;
const int startPos = 2;
const int endPos = 80;
int cCells[cellSize], pCells[cellSize]; // Current cells and previous cells
int rsLen = 0;
string *ruleSet;
string ruleString = "";

string ruleToString(int rule, int r) {

  int bitnum = pow(2, (2 * r + 1));
  string out = "";

  int ruleHolder = rule;
for (int i = bitnum - 1; i >= 0; i--) {
  int val = pow(2, i);
  if (ruleHolder - val >= 0) {
    ruleHolder -= val;
    out.append("1");
  }
  else {
    out.append("0");
  }
}

return out;
}

void initRuleSet(int r) {
  int bitnum = 2 * r + 1;
  int bitlen = pow(2, bitnum);
  rsLen = bitlen;
  ruleSet = new string[rsLen];

  for (int j = 1; j <= bitlen; j++) {
    int holder = bitlen - j;
    string curRule = "";
    for (int i = bitnum - 1; i >= 0; i--) {
      int val = pow(2, i);
      if (holder - val >= 0) {
        holder -= val;
        curRule.append("1");
      }
      else {
        curRule.append("0");
      }
    }

    ruleSet[j - 1] = curRule;
  }

  for (int i = 0; i < rsLen; i++) {
    printf(ruleSet[i].c_str());
    printf("\n");
  }

  return;

}

int init(int rule, int r, char condition) {

  ruleString = ruleToString(rule, r);
  initRuleSet(r);

  for (int i = 0; i < cellSize; i++) {
    cCells[i] = 0;
    pCells[i] = 0;
  }

  if (r != 1 && r != 2) {
    printf("Invalid value for r.\n");
    return 0;
  }

  if (condition == 'S' or condition == 's') {
    printf("Seed, r=%d\n", r);

    pCells[seedCell] = 1;
  }
  else if (condition == 'R' or condition == 'r') {
    printf("Random, r=%d\n", r);

    for (int i = startPos; i <= endPos; i++) { // Skips the first and last two cells to create a boundary.
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

  for (int i = 0; i < cellSize; i++) {
    printf("%d ", pCells[i]);
  }

  printf("\n");

  return 1;
}

int evalRules(int pos, int r) {
  string state = "";
  for (int i = pos - r; i <= pos + r; i++) {
    state.append(to_string(pCells[i]));
  }

  for (int i = 0; i < rsLen; i++){
    if (ruleSet[i] == state) {
      return (int)ruleString.at(i) - '0';
    }
  }

  return 1;
}

void step(int r) {
  for (int i = startPos; i <= endPos; i++) {
    cCells[i] = evalRules(i, r);
  }

  for (int i = 0; i < cellSize; i++) {
    printf("%d ", cCells[i]);
    pCells[i] = cCells[i];
    cCells[i] = 0;
  }
  printf("\n");
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Invalid Arguments. Radius (1 or 2), CA Rule (eg. 150) and Starting Conditions (s or r) required.\n");
    return 0;
  }

  srand(time(NULL));

  int r = 0;
  int rule = 0;
  char cond = 'r';

  try {
    string rS = argv[1];
    string ruleS = argv[2];
    string condS = argv[3];

    r = stoi(rS);
    rule = stoi(ruleS);
    cond = condS[0];
  }
  catch (exception e) {
    printf("Invalid inputs. Please format the arguments as follows:\n");
    printf("Radius - Integer (1 or 2)\n");
    printf("Cellular Automata Rule - Integer (eg. 150)\n");
    printf("Starting Conditions - Character (r or s)\n");
    return 0;
  }

  
  int res = init(rule, r, cond);
  if (res == 0) {
    return 0;
  }

  for (int i = 0; i < 20; i++) {
    step(r);
  }

  return 0;
}

