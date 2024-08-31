#ifndef micrograd_value_h
#define micrograd_value_h

#include <stdio.h>

// Forward declaration to use it in the BackwardFunc function pointer type
typedef struct Value Value;

typedef void (*BackwardFunc)(Value*, Value*, Value*);

typedef struct {
	int capacity;
	int count;
	Value** values;
} ValueArray;

struct Value {
	float data;
	float grad;
	BackwardFunc backward;
	ValueArray prev;
	int id;
};

void initValueArray(ValueArray* array);
void writeValueArray(ValueArray* array, Value* value);
void freeValueArray(ValueArray* array);

Value* newValue(float data);
void freeValue(Value* value);

Value* vAdd(Value* self, Value* other);
Value* vMul(Value* self, Value* other);
Value* vPow(Value* self, Value* other);
Value* vRelu(Value* self);

void backward(Value* val);

void printValue(Value value);

#endif
