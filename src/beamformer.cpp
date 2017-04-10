//
// Created by lessju on 12/05/2016.
//

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>

#include "beamformer.hpp"


void beamform(complex64 *input, complex64 *weights, complex64 *output,
                int nsamp, int nchans, int nbeams, int nants, int npols, int nthreads)

{
    // Loop over all polarizations
    for(unsigned p = 0; p < npols; p++)
    {
        // Loop over all channels
        for(unsigned c = 0; c < nchans; c++)
        {
	        // Create openmp threads which will split up beams between them
	        #pragma omp parallel for shared(input, weights, output) schedule(dynamic) num_threads(nthreads)
	        for (unsigned b = 0; b < nbeams; b++)
	        {
	            // Loop over samples
	            for (unsigned s = 0; s < nsamp; s++)
	            {
	                // Get pointer to required segment of weights and create temporary result
	                complex64 *wp = weights + c * nants * nbeams + b * nants;

	                // Get pointer to required segment of input buffer
	                complex64 *ip = input + p * nsamp * nants * nchans + c * nsamp * nants + s * nants;

	                complex64 res = { 0, 0};

	                // Loop over antennas
	                for (unsigned a = 0; a < nants; a++)
	                {
	                    // Perform complex multiply and add to result
	                    res.r += ip -> r * wp -> r;
	                    res.i += ip -> r * wp -> i;
	                    res.r -= ip -> i * wp -> i;
	                    res.i += ip -> i * wp -> r;

	                    // Advance input and weight pointers
	                    ip ++; wp ++;
	                }

	                // Save to output
	                output[p * nbeams * nchans * nsamp + b * nchans * nsamp + c * nsamp + s] = res;
	            }
	        }
        }
    }
}


int main()
{
    int nsamp = 1024*1024*4;
    int npols = 2;
    int nchans = 1;
    int nbeams = 32;
    int nants = 32;

    complex64 *input = (complex64 *) malloc(npols * nchans * nsamp * nants * sizeof(complex64));
    complex64 *weights = (complex64 *) malloc(npols * nchans * nbeams * nants * sizeof(complex64));
    complex64 *output = (complex64 *) malloc(npols * nbeams * nchans * nsamp * sizeof(complex64));

    struct timeval start, end;
    long mtime, seconds, useconds;
    gettimeofday(&start, NULL);

    beamform(input, weights, output, nsamp, nchans, nbeams, nants, npols, 2);

    gettimeofday(&end, NULL);
    seconds  = end.tv_sec  - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec;

    mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
    printf("Time: %ld ms\n", mtime);
}
