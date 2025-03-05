#ifndef SHUFFLE_H
#define SHUFFLE_H

typedef struct Shuffle {
    int *indices;
    int current;
    int count;
} Shuffle;

Shuffle *shuffle_create(int samples_count);
void shuffle_free(Shuffle *shuffle);
void shuffle_shuffle(Shuffle *shuffle);
int shuffle_next(Shuffle *shuffle);
int shuffle_end(Shuffle *shuffle);

#endif
