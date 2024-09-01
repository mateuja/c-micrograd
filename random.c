#include <stdlib.h>
#include "random.h"

float randomUniform(float min, float max) {
	return min + ((float)rand() / RAND_MAX) * (max - min);
}

static int randomUniformInt(int min, int max) {
	return rand() % (max - min + 1) + min;
}

static void swap(int *a, int* b) {
	int temp = *a;
	*a = *b;
	*b = temp;
}

int* fisherYatesShuffle(int size) {
	int* array = (int*)malloc(size * sizeof(int));
	for (int i=0; i < size; i++) {
		array[i] = i;
	}
	
	for (int i=0; i < size; i++) {
		swap(&array[i], &array[randomUniformInt(0, size-1)]);
	}

	return array;
}

