#include <stdlib.h>
#include <math.h>
#include "shuffle.h"

Shuffle *shuffle_create(int samples_count) {
    Shuffle *shuffle = (Shuffle *)malloc(sizeof(Shuffle));
    shuffle->indices = (int *)malloc(samples_count * sizeof(int));
    shuffle->current = 0;
    shuffle->count = samples_count;
    for (int i = 0; i < samples_count; i++) {
        shuffle->indices[i] = i;
    }
    return shuffle;
}

void shuffle_free(Shuffle *shuffle) {
    free(shuffle->indices);
    free(shuffle);
}

void shuffle_shuffle(Shuffle *shuffle) {
    for (int i = 0; i < shuffle->count; i++) {
        int j = rand() % shuffle->count;
        if (i == j) {
            continue;
        }
        int temp = shuffle->indices[i];
        shuffle->indices[i] = shuffle->indices[j];
        shuffle->indices[j] = temp;
    }
    shuffle->current = 0;
}

int shuffle_next(Shuffle *shuffle) {
    return shuffle->indices[shuffle->current++];
}

int shuffle_end(Shuffle *shuffle) {
    return shuffle->current >= shuffle->count;
}
