#ifndef micrograd_dynamic_array_h
#define micrograd_dynamic_array_h

#include <stdlib.h>

#define GROW_CAPACITY(capacity) \
  ((capacity) < 1 ? 1 : (capacity) * 2)

#define GROW_ARRAY(type, pointer, oldCount, newCount) \
  (type*)reallocate( \
     pointer, sizeof(type) * (oldCount), sizeof(type) * (newCount) \
  )

#define DEFINE_DYNAMIC_ARRAY(type) \
  typedef struct { \
    size_t capacity; \
    size_t count; \
    type** items; \
  } type##Array;

#define APPEND_ARRAY(array, element) \
    do { \
        typeof(array) _arr = (array); \
        if (_arr->count >= _arr->capacity) { \
            size_t oldCapacity = _arr->capacity; \
            _arr->capacity = GROW_CAPACITY(oldCapacity); \
            _arr->items = GROW_ARRAY(typeof(*(_arr->items)), _arr->items, \
                                    oldCapacity, _arr->capacity); \
        } \
        _arr->items[_arr->count] = (element); \
        _arr->count++; \
    } while (0)

void* reallocate(void* pointer, size_t oldSize, size_t newSize);

#endif
