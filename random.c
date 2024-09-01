#include <stdlib.h>
#include "random.h"

float randomUniform(float min, float max) {
	return min + ((float)rand() / RAND_MAX) * (max - min);
}
