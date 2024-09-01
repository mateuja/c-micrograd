#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "engine.h"
#include "nn.h"
#include "random.h"

ValueArray* readCsv(const char* path, int rows, int cols) {
	FILE *file = fopen(path, "r");
	if (file == NULL) {
		perror("Unable to open file!");
		exit(1);
	}
	
	ValueArray* data;
	if (cols == 1) {
		data = (ValueArray*)malloc(sizeof(ValueArray));
		data->values = (Value**)malloc(rows*sizeof(Value*));
		data->count = rows;
		data->capacity = rows;
	} else {
		data = (ValueArray*)malloc(rows * sizeof(ValueArray));
		if (data == NULL) {
			perror("Memory allocation failed!");
			fclose(file);
			exit(1);
		}

		for (int i=0; i < rows; i++) {
			data[i].values = (Value**)malloc(cols * sizeof(Value*));
			data[i].count = cols;
			data[i].capacity = cols;
		}
	}

	// Read the data into the array
	int r = 0;
	char buffer[1024];
	while (fgets(buffer, sizeof(buffer), file)) {
		char* token = strtok(buffer, ",");
		if (cols == 1) {
			if (token) {
				data->values[r] = newValue(atof(token));
			}
		} else {
			for (int c=0; c < cols; c++) {
				if (token) {
					data[r].values[c] = newValue(atof(token));
					token = strtok(NULL, ",");
				}
			}
		}
		r++;
	}

	fclose(file);

	return data;
}

void printValueArray(ValueArray* data, int rows, int cols) {
	for (int i=0; i < rows; i++) {
		if (cols == 1) {
			printValue(*data->values[i]);
		} else {
			for (int j=0; j < cols; j++) {
				printValue(*data[i].values[j]);
			}
		}
	}
}

Value* computeDataLoss(ValueArray* labels, ValueArray* scores) {
	Value* totalLoss = newValue(0.0);
	for (int i=0; i < scores->count; i++) {
		Value* label = labels->values[i];
		Value* score = scores->values[i]; 
		
		totalLoss = vAdd(
			totalLoss,
			vRelu(vAddFloat(vMul(vNeg(label), score), 1.0))
		);
	}
	return vDiv(totalLoss, newValue((float)scores->count));
}

Value* computeRegLoss(ValueArray* params, float alpha) {
	Value* loss = newValue(0.0);
	for (int i=0; i < params->count; i++) {
		loss = vAdd(
			loss,
			vMul(params->values[i], params->values[i])
		);
	}
	return vMulFloat(loss, alpha);
}

float accuracy(ValueArray* labels, ValueArray* scores) {
	float accuracy = 0;
	for (int i=0; i < scores->count; i++) {
		accuracy += (int)((labels->values[i]->data > 0) == (scores->values[i]->data > 0));
	}
	return accuracy / scores->count;
}

void updateParams(ValueArray* params, float lr) {
	for (int i=0; i < params->count; i++) {
		params->values[i]->data -= lr * params->values[i]->grad;
	}
}

void printParams(ValueArray* params) {
	printf("Params:\n");
	for (int i=0; i < params->count; i++) {
		printValue(*params->values[i]);
	}
	printf("\n");
}

int main() {
	// Seed the random number generator
	srand(time(NULL));

	int nRows = 100;
	int nFeats = 2;
	
	// Read inputs
	ValueArray* inputs = readCsv("data/X.csv", nRows, nFeats);
	ValueArray* labels = readCsv("data/y.csv", nRows, 1);
	
	// Initialize model
	MLP* model = newMLP(4, 2, 16, 16, 1);

	// Optimization
	int nEpoch = 100;
	for (int epoch=0; epoch < nEpoch; epoch++) {
		// Compute scores
		ValueArray* scores = (ValueArray*)malloc(sizeof(ValueArray));
		scores->values = (Value**)malloc(nRows * sizeof(Value*));
		scores->count = nRows;
		scores->capacity = nRows;

		for (int i=0; i < nRows; i++) {
			scores->values[i] = forwardMLP(model, &inputs[i]);
		}

		Value* dataLoss = computeDataLoss(labels, scores);
		float alpha = 1e-4;
		Value* regLoss = computeRegLoss(paramsMLP(model), alpha);
		Value* loss = vAdd(dataLoss, regLoss);

		float acc = accuracy(labels, scores);

		// backward
		zeroGrad(paramsMLP(model));
		backward(loss);
		
		float lr = 1.0 - 0.9 * epoch / nEpoch;
		updateParams(paramsMLP(model), lr);
		 
		printf("step %d loss %f, accuracy %.4f \n", epoch, loss->data, acc);
	}
}
