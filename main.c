#include "value.h"

int main() {
	Value* a = newValue(-4.0);
	Value* b = newValue(2.0);
	Value* c = vAdd(a, b);
	Value* auxA = newValue(3.0);
	Value* d = vAdd(vMul(a, b), vPow(b, auxA));
	
	backward(d);

	printValue(*a);
	printValue(*b);

	return 0;
}
