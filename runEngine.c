#include "engine.h"

int main() {
	Value* a = newValue(-4.0);
	Value* b = newValue(2.0);
	Value* c = vAdd(a, b);
	Value* d = vAdd(vMul(a, b), vPowFloat(b, 3.0));
	c = vAddFloat(vAdd(c, c), 1.0);
	c = vAdd(vAdd(vAddFloat(c, 1), vNeg(a)), c);
	d = vAdd(d, vAdd(vMulFloat(d, 2), vRelu(vAdd(b, a))));
	d = vAdd(d, vAdd(vMulFloat(d, 3), vRelu(vSub(b, a))));
	Value* e = vSub(c, d);
	Value* f = vPowFloat(e, 2);
	Value* g = vDivFloat(f, 2.0);
	g = vAdd(g, vDiv(newValue(10.0), f));
	
	backward(g);

	printValue(*a);
	printValue(*b);

	freeTopo(g);

	return 0;
}

