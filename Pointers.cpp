#include <iostream>

using namespace std;

int main () {
   int  var = 10;
   int  *ptr;

   ptr = &var;

   cout << "Value of var variable: ";
   cout << var << endl;


   cout << "Address stored in ptr variable: ";
   cout << ptr << endl;


   cout << "Value of *ptr variable: ";
   cout << *ptr << endl;

   return 0;
}
