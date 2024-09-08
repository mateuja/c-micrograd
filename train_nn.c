#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "engine.h"
#include "nn.h"

ValueArray* readCsv(const char* path, size_t rows, size_t cols) {
	FILE *file = fopen(path, "r");
	if (file == NULL) {
		perror("Unable to open file!");
		exit(1);
	}
	
	ValueArray* data;
	if (cols == 1) {
		data = (ValueArray*)malloc(sizeof(ValueArray));
		data->items = (Value**)malloc(rows*sizeof(Value*));
		data->count = rows;
		data->capacity = rows;
	} else {
		data = (ValueArray*)malloc(rows * sizeof(ValueArray));
		if (data == NULL) {
			perror("Memory allocation failed!");
			fclose(file);
			exit(1);
		}

		for (size_t i=0; i < rows; i++) {
			data[i].items = (Value**)malloc(cols * sizeof(Value*));
			data[i].count = cols;
			data[i].capacity = cols;
		}
	}

	// Read the data into the array
	size_t r = 0;
	char buffer[1024];
	while (fgets(buffer, sizeof(buffer), file)) {
		char* token = strtok(buffer, ",");
		if (cols == 1) {
			if (token) {
				data->items[r] = newValue(atof(token));
			}
		} else {
			for (size_t c=0; c < cols; c++) {
				if (token) {
					data[r].items[c] = newValue(atof(token));
					token = strtok(NULL, ",");
				}
			}
		}
		r++;
	}

	fclose(file);

	return data;
}

void freeInputs(ValueArray* data, size_t rows) {
	for (size_t i=0; i < rows; i++) {
		for (size_t j=0; j < 2; j++) {
			freeValue(&data[i].items[j]);
			assert(data[i].items[j] == NULL);
		}
		free(data[i].items);
	}
	free(data);
}

void freeLabels(ValueArray* data, size_t rows) {
	for (size_t i=0; i < rows; i++) {
		freeValue(&data->items[i]);
		assert(data->items[i] == NULL);
	}
	free(data->items);
	free(data);
}

void printValueArray(ValueArray* data, size_t rows, size_t cols) {
	for (size_t i=0; i < rows; i++) {
		if (cols == 1) {
			printValue(*data->items[i]);
		} else {
			for (size_t j=0; j < cols; j++) {
				printValue(*data[i].items[j]);
			}
		}
	}
}

Value* computeDataLoss(ValueArray* labels, ValueArray* scores) {
	Value* totalLoss = newValue(0.0);
	for (size_t i=0; i < scores->count; i++) {
		Value* label = labels->items[i];
		Value* score = scores->items[i]; 
		
		totalLoss = vAdd(
			totalLoss,
			vRelu(vAddFloat(vMul(vNeg(label), score), 1.0))
		);
	}
	return vDiv(totalLoss, newValue((float)scores->count));
}

Value* computeRegLoss(ValueArray* params, float alpha) {
	Value* loss = newValue(0.0);
	for (size_t i=0; i < params->count; i++) {
		loss = vAdd(
			loss,
			vMul(params->items[i], params->items[i])
		);
	}
	return vMulFloat(loss, alpha);
}

float accuracy(ValueArray* labels, ValueArray* scores) {
	float accuracy = 0;
	for (size_t i=0; i < scores->count; i++) {
		accuracy += (int)((labels->items[i]->data > 0) == (scores->items[i]->data > 0));
	}
	return accuracy / scores->count;
}

void updateParams(ValueArray* params, float lr) {
	for (size_t i=0; i < params->count; i++) {
		params->items[i]->data -= lr * params->items[i]->grad;
	}
}

int main() {
	// Seed the random number generator
	srand(1337);

	size_t nRows = 100;
	size_t nFeats = 2;
	
	// Read inputs
	ValueArray* inputs = readCsv("data/X.csv", nRows, nFeats);
	ValueArray* labels = readCsv("data/y.csv", nRows, 1);
	
	// Initialize model
	MLP* model = newMLP(4, 2, 16, 16, 1);

	// Get params
	ValueArray* params = paramsMLP(model);
	
	// Initialize scores
	ValueArray* scores = (ValueArray*)malloc(sizeof(ValueArray));
	scores->items = (Value**)malloc(nRows * sizeof(Value*));
	scores->count = nRows;
	scores->capacity = nRows;

	// Optimization
	size_t nEpoch = 100;
	for (size_t epoch=0; epoch < nEpoch; epoch++) {
		int startEpochId = ID_COUNTER;	

		// forward
		for (size_t i=0; i < nRows; i++) {
			scores->items[i] = forwardMLP(model, &inputs[i]);
		}

		Value* dataLoss = computeDataLoss(labels, scores);
		Value* regLoss = computeRegLoss(params, 1e-5);
		Value* loss = vAdd(dataLoss, regLoss);

		float acc = accuracy(labels, scores);

		// backward
		zeroGrad(params);
		backward(loss);
		
		float lr = 1.0 - 0.9 * epoch / nEpoch;
		updateParams(params, lr);
		 
		printf("step %lu loss %f, accuracy %.4f \n", epoch, loss->data, acc);
		
		// free all nodes that were created in the epoch
		freeDAG(loss, startEpochId);
	}
	freeValueArray(&params);
	freeValueArray(&scores);
	freeMLP(&model);
	freeLabels(labels, nRows);
	freeInputs(inputs, nRows);
}
