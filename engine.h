#ifndef micrograd_engine_h
#define micrograd_engine_h

#include <stdbool.h>
#include <stdio.h>

#include "dynarray.h"

extern int ID_COUNTER;

// Forward declaration to use it in the BackwardFunc function pointer type
typedef struct Value Value;

typedef enum {
	BW_NULL,
	BW_ADD,
	BW_MUL,
	BW_POW,
	BW_RELU,
} BackwardFunc;

DEFINE_DYNAMIC_ARRAY(Value);

struct Value {
	float data;
	float grad;
	BackwardFunc backward;
	ValueArray prev;
	int id;
};

void initValueArray(ValueArray* array);
void freeValueArray(ValueArray** array);

Value* newValue(float data);
void freeValue(Value** value);

Value* vAdd(Value* self, Value* other);
Value* vAddFloat(Value* self, float other);
Value* vMul(Value* self, Value* other);
Value* vMulFloat(Value* self, float other);
Value* vPow(Value* self, Value* other);
Value* vPowFloat(Value* self, float other);
Value* vRelu(Value* self);
Value* vNeg(Value* self);
Value* vSub(Value*self, Value* other);
Value* vSubFloat(Value* self, float other);
Value* vDiv(Value* self, Value* snd);
Value* vDivFloat(Value* self, float other);

void backward(Value* val);

void freeDAG(Value* last, int fromId);

void printValue(Value value);

#endif
