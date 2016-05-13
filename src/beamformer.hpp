//
// Created by lessju on 12/05/2016.
//

#ifndef BEAMFORMER_BEAMFORMER_H
#define BEAMFORMER_BEAMFORMER_H

typedef struct complex64
{
    float r, i;
} complex64;

extern "C" void beamform(complex64 *input, complex64 *weights, complex64 *output,
                         int nsamp, int nchans, int nbeams, int nants, int nthreads);

#endif //BEAMFORMER_BEAMFORMER_H
