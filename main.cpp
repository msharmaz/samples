#include <iostream>
#include <cstdio>
using namespace std;

int main() {
    // Complete the code.
    int a;
    long b;
    long long c;
    char ch;
    float e;
    double d;
    scanf("%d %ld %lld %c %f %lf", &a, &b, &c, &ch, &e, &d);
    printf ("%d \n%ld\n%lld\n%c\n%.2f\n%.5lf\n", a, b, c, ch, e, d);
    return 0;
}
