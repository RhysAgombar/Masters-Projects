// Exercise 2 Programming Assignment - By: Rhys Agombar & Joanna Polewczyk

#include <stdio.h>
#include <tchar.h>
#include <stdlib.h>
#include "time.h"
#include <string>
#include <bitset>
#include <cmath>

using namespace std;

const int cellSize = 83; // Size of Cell Array
const int seedCell = 42; // Seed Cell Index
const int startPos = 2; // Start Position Index
const int endPos = 80; // End Position Index
int cCells[cellSize], pCells[cellSize]; // Current cells and previous cells
int ssLen = 0; // Length of the state set (this will be properly set in the initStateSet function
string *stateSet; // Uninitialized array of states
string ruleString = ""; // The binary representation of the CA Rule Number

/// <summary>
/// Converts the CA number to a binary string which will be used in the evalStates function to return the correct new state of a cell.
/// </summary>
/// <param name="rule">The rule number to be converted</param>
/// <param name="r">The radius of the CA</param>
/// <returns></returns>
string ruleToString(int rule, int r) {

  int bitnum = pow(2, (2 * r + 1)); //Calculates the number of bits the string will contain
  string out = "";

  int ruleHolder = rule;
  for (int i = bitnum - 1; i >= 0; i--) { // Counting down from the highest bit to the lowest bit...
    int val = pow(2, i); // Find the value of that bit (2^7, 2^6, 2^5, etc...)
    if (ruleHolder - val >= 0) { // If the value fits into the rule number (150 - 128 = 22, etc.)...
      ruleHolder -= val;
      out.append("1"); // Update the value and append a 1 to the binary representation
    }
    else {
      out.append("0"); // If not, append a 0 to the representation as the current bit does not fit.
    }
  }

return out;
}

/// <summary>
/// Initializes the state set. Each state is stored as a set of binary bits, representing the states of the previous cells (11111, 101, 100, 001, etc...)
/// </summary>
/// <param name="r">The radius of the CA</param>
/// <returns></returns>
void initStateSet(int r) {
  int bitnum = 2 * r + 1; // Number of bits in each state. For r=1, there are a total of 3 bits in each state. 111 = 0, 110 = 1, etc.
  int bitlen = pow(2, bitnum); // Length of the stateset
  ssLen = bitlen;
  stateSet = new string[ssLen]; // Creating an array of strings

  for (int j = 1; j <= bitlen; j++) { // The same process as in ruleToString
    int holder = bitlen - j;
    string curState = "";
    for (int i = bitnum - 1; i >= 0; i--) {
      int val = pow(2, i);
      if (holder - val >= 0) {
        holder -= val;
        curState.append("1");
      }
      else {
        curState.append("0");
      }
    }

    stateSet[j - 1] = curState; // Save the state
  }
  return;
}

/// <summary>
/// Initializes the program. 
/// </summary>
/// <param name="rule">The CA Number</param>
/// <param name="r">The radius of the CA</param>
/// <param name="condition">The starting condition</param>
/// <returns></returns>
int init(int rule, int r, char condition) {

  ruleString = ruleToString(rule, r); // Sets the ruleString
  initStateSet(r); // Initializes the state set.

  for (int i = 0; i < cellSize; i++) { // Initialize arrays to 0
    cCells[i] = 0;
    pCells[i] = 0;
  }

  if (r != 1 && r != 2) { // Check for invalid radius
    printf("Invalid value for r.\n");
    return 0;
  }

  if (condition == 'S' or condition == 's') { // If seed starting condition, only set seed cell to be 1
    printf("Seed, r=%d, rule=%d\n", r, rule);

    pCells[seedCell] = 1;
  }
  else if (condition == 'R' or condition == 'r') {
    printf("Random, r=%d, rule=%d\n", r, rule);

    for (int i = startPos; i <= endPos; i++) { // Skips the first and last two cells to create a boundary.
      int chance = rand() % 10;
      if (chance < 5) { // 50% chance of the cell being a 0 or a 1
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

  for (int i = 0; i < cellSize; i++) { // print the initial starting condition of the cells
    printf("%d ", pCells[i]);
  }

  printf("\n");

  return 1;
}

/// <summary>
/// Determines the state that a cell should be in based on the rule set and previous cell states
/// </summary>
/// <param name="pos">The position of the cell to be evaluated</param>
/// <param name="r">The radius of the CA</param>
/// <returns></returns>
int evalStates(int pos, int r) {
  string state = "";
  for (int i = pos - r; i <= pos + r; i++) { // check the states of the cell neighbors from the previous time step
    state.append(to_string(pCells[i])); // construct a binary representation of the cell states eg. 110, 101, 11001, etc.
  }

  for (int i = 0; i < ssLen; i++){ // for every state in the stateset
    if (stateSet[i] == state) { // check if the state is relevant (the same as the cell state constructed in the above step)
      return (int)ruleString.at(i) - '0'; // return the result state as given by the binary representation of the CA number.
      // ruleString will be a string of binary, eg. 11001011. Each bit corresponds to the result of a set of previous states, therefore by returning the 
      // bit at the index of the previous state found, we get the result
    }
  }

  return -1; // If this gets hit, something has gone horribly wrong.
}

/// <summary>
/// Proceeds to the next timestep and generates the cell states based on the evaluation of the states.
/// </summary>
/// <param name="r">The radius of the CA</param>
/// <returns></returns>
void step(int r) {
  for (int i = startPos; i <= endPos; i++) { // For each cell in the current cell list
    cCells[i] = evalStates(i, r); // Set it's state equal to the result of the rule evaluation
  }

  for (int i = 0; i < cellSize; i++) { // for every cell...
    printf("%d ", cCells[i]); // Print the current cells to the console
    pCells[i] = cCells[i]; // Save the current values of cells to the previous cells list
    cCells[i] = 0; // set all current cells to state 0 in preparation for the next step.
  }
  printf("\n");
}


int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Invalid Arguments. Radius (1 or 2), CA Rule (eg. 150) and Starting Conditions (s or r) required.\n");
    return 0;
  }

  srand(time(NULL)); // Seeds the randomizer function with a value from the system clock

  int r = 0;
  int rule = 0;
  char cond = 'r';

  try { // Accept and format the input arguments
    string rS = argv[1];
    string ruleS = argv[2];
    string condS = argv[3];

    r = stoi(rS);
    rule = stoi(ruleS);
    cond = condS[0];
  }
  catch (exception e) { // If anything fails, something is wrong with the formatting. Display message to user.
    printf("Invalid inputs. Please format the arguments as follows:\n");
    printf("Radius - Integer (1 or 2)\n");
    printf("Cellular Automata Rule - Integer (eg. 150)\n");
    printf("Starting Conditions - Character (r or s)\n");
    return 0;
  }

  
  int res = init(rule, r, cond); // Initialize the system

  if (res == 0) { // If an error code (0) is returned, quit the program.
    return 0;
  }

  for (int i = 0; i < 20; i++) { 
    step(r); // This is the function that generates the next timestep
  }

  return 0;
}

