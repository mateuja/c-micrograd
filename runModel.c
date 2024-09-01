#include <stdlib.h>
#include <time.h>

#include "engine.h"
#include "nn.h"

int main() {
	// Seed the random number generator
	srand(time(NULL));

	ValueArray* x = (ValueArray*)malloc(sizeof(ValueArray));
	for (int i=0; i < 5; i++) {
		writeValueArray(x, newValue((float)(i * 2)));
	}
	
	MLP* mlp = newMLP(2, 5, 1);
	Value* res = forwardMLP(mlp, x);

	printValue(*res);
}
