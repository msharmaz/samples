#include <iostream>
using namespace std;
 
int getUserInput()
{
    cout << "Please enter an integer: ";
    int value;
    cin >> value;
    return value;
}
 
int getMathematicalOperation()
{
    cout << "Please enter which operator you want: ";
 
    int op;
    cin >> op;

    return op;
}
 
int calculateResult(int x, int op, int y)
{
    
    if (op == 1) // if user chose addition (#1)
        return x + y; // execute this line
    if (op == 2) // if user chose subtraction (#2)
        return x - y; // execute this line
    if (op == 3) // if user chose multiplication (#3)
        return x * y; // execute this line
    if (op == 4) // if user chose division (#4)
        return x / y; // execute this line
	
    return -1; 
}
 
void printResult(int result)
{
    cout << "Your result is: " << result << endl;
}
 
int main()
{
  
    int input1 = getUserInput();
 
    int op = getMathematicalOperation();

    int input2 = getUserInput();
 
    int result = calculateResult(input1, op, input2 );
 
    printResult(result);
}
