#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "memory.h"
#include "engine.h"

int ID_COUNTER = 1;

void initValueArray(ValueArray* array) {
	array->values = NULL;
	array->capacity = 0;
	array->count = 0;
}

void writeValueArray(ValueArray* array, Value* value) {
	if (array->capacity < array->count + 1) {
		int oldCapacity = array->capacity;
		array->capacity = GROW_CAPACITY(oldCapacity);
		array->values = GROW_ARRAY(Value*, array->values, oldCapacity, array->capacity);
	}
	array->values[array->count] = value;
	array->count++;
}

void freeValueArray(ValueArray** array) {
	if (array == NULL || *array == NULL) {
		return; // Do nothing if the array is already null
	}

	ValueArray* tempArray = *array;
	
	free(tempArray->values);
	tempArray->values = NULL;
	free(tempArray);
	tempArray = NULL;
	*array = NULL;
}

Value* newValue(float data) {
	Value* val = (Value*)malloc(sizeof(Value));
	if (val == NULL) {
		fprintf(stderr, "Memory allocation failed\n");
		exit(EXIT_FAILURE);
	}
	val->data = data;
	val->grad = 0;
	val->id = ID_COUNTER;
	val->backward = BW_NULL;
	ID_COUNTER++;
	initValueArray(&val->prev);

	return val;
}

void freeValue(Value** val) {
	if (val == NULL || *val == NULL) {
		return;
	}
	Value* tempValue = *val;	
	
	free(tempValue->prev.values);
	tempValue->prev.values = NULL;
	free(tempValue);
	*val = NULL;
}

void printValue(Value value) {
	printf("Value(id=%d, data=%.4f, grad=%.4f)\n", value.id, value.data, value.grad);
}

static void backwardAdd(Value* out, Value* fst, Value* snd)  {
	fst->grad += out->grad;
	snd->grad += out->grad;
}

Value* vAdd(Value* self, Value* other) {
	Value* out = newValue(self->data + other->data);
	out->backward = BW_ADD;
	writeValueArray(&out->prev, self);
	writeValueArray(&out->prev, other);
	return out;
}

Value* vAddFloat(Value* self, float other) {
	return vAdd(self, newValue(other));
}

static void backwardMul(Value* out, Value* fst, Value* snd) {
	fst->grad += snd->data * out->grad;
	snd->grad += fst->data * out->grad;
}

Value* vMul(Value* self, Value* other) {
	Value* out = newValue(self->data * other->data);
	out->backward = BW_MUL;
	writeValueArray(&out->prev, self);
	writeValueArray(&out->prev, other);
	return out;
}

Value* vMulFloat(Value* self, float other) {
	return vMul(self, newValue(other));
}

static void backwardPow(Value* out, Value* fst, Value* snd) {
	fst->grad += snd->data * pow(fst->data, snd->data - 1) * out->grad;
}

Value* vPow(Value* self, Value* other) {
	Value* out = newValue(pow(self->data, other->data));
	out->backward = BW_POW;
	writeValueArray(&out->prev, self);
	writeValueArray(&out->prev, other);
	return out;
}

Value* vPowFloat(Value* self, float other) {
	return vPow(self, newValue(other));
}

Value* vFloatPow(float self, Value* other) {
	return vPow(newValue(self), other);
}

static void backwardRelu(Value* out, Value* fst) {
	fst->grad += (out->data > 0 ? out->grad : 0);
}

Value* vRelu(Value* self) {
	Value* out = newValue(self->data < 0 ? 0 : self->data);
	out->backward = BW_RELU;
	writeValueArray(&out->prev, self);
	return out;
}

Value* vNeg(Value* self) {
	return vMulFloat(self, -1.0);
}

Value* vSub(Value* self, Value* other) {
	return vAdd(self, vNeg(other));
}

Value* vSubFloat(Value* self, float other) {
	return vAddFloat(self, -other);
}

Value* vDiv(Value* self, Value* other) {
	return vMul(self, vPowFloat(other, -1.0));
}

Value* vDivFloat(Value* self, float other) {
	return vDiv(self, newValue(other));
}

static void buildTopo(Value* val, ValueArray* topo, bool* visited) {
	if (!visited[val->id - 1]) {
		visited[val->id - 1] = true;
		for (int i=0; i < val->prev.count; i++) {
			buildTopo(val->prev.values[i], topo, visited);
		}
		writeValueArray(topo, val);
	}
}

void freeDAG(Value* last, int fromId) {
	ValueArray* topo = (ValueArray*)malloc(sizeof(ValueArray));
	initValueArray(topo);
	bool* visited = (bool*)calloc((ID_COUNTER), sizeof(bool));

	buildTopo(last, topo, visited);

	for (int i=0; i < topo->count; i++) {
		if (topo->values[i] != NULL && topo->values[i]->id >= fromId) {
			freeValue(&topo->values[i]);
			assert(topo->values[i] == NULL);
		}
	}

	
	freeValueArray(&topo);
	assert(topo == NULL);
	
	free(visited);
}

void backward(Value* val) {
	ValueArray* topo = (ValueArray*)malloc(sizeof(ValueArray));
	initValueArray(topo);
	bool* visited = (bool*)calloc((ID_COUNTER), sizeof(bool));

	buildTopo(val, topo, visited);

	val->grad = 1;
	
	Value* currentVal;
	for (int i=topo->count-1; i >= 0; i--) {
		currentVal = topo->values[i];

		switch (currentVal->backward) {
			case BW_NULL:
				continue;
			case BW_ADD:
				backwardAdd(
					currentVal, currentVal->prev.values[0], currentVal->prev.values[1]
				);
				continue;
			case BW_MUL:
				backwardMul(
					currentVal, currentVal->prev.values[0], currentVal->prev.values[1]
				);
				continue;
			case BW_POW:
				backwardPow(
					currentVal, currentVal->prev.values[0], currentVal->prev.values[1]
				);
				continue;
			case BW_RELU:
				backwardRelu(
					currentVal, currentVal->prev.values[0]
				);
				continue;
		}
	}
	free(visited);
	freeValueArray(&topo);
}

