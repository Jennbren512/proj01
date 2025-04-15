#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#ifndef F_PI
#define F_PI		(float)M_PI
#endif

#ifndef DEBUG
#define DEBUG		false
#endif

#ifndef NUMT
#define NUMT		2
#endif

#ifndef NUMTRIALS
#define NUMTRIALS	50000
#endif

#ifndef NUMTRIES
#define NUMTRIES	30
#endif

const float GMIN = 10.0;
const float GMAX = 20.0;
const float HMIN = 20.0;
const float HMAX = 30.0;
const float DMIN = 10.0;
const float DMAX = 20.0;
const float VMIN = 20.0;
const float VMAX = 30.0;
const float THMIN = 70.0;
const float THMAX = 80.0;

const float GRAVITY = -9.8;
const float TOL = 5.0;

float Ranf(float low, float high) {
    float r = (float)rand();
    float t = r / (float)RAND_MAX;
    return low + t * (high - low);
}

void TimeOfDaySeed() {
    time_t now;
    time(&now);
    struct tm n;
#ifdef _WIN32
    localtime_s(&n, &now);
#else
    n = *localtime(&now);
#endif
    srand((unsigned int)(1000 * difftime(now, mktime(&n))));
}

inline float Radians(float degrees) {
    return (F_PI / 180.f) * degrees;
}

int main() {
#ifndef _OPENMP
    fprintf(stderr, "No OpenMP support!\n");
    return 1;
#endif

    TimeOfDaySeed();
    omp_set_num_threads(NUMT);

    float *vs = new float[NUMTRIALS];
    float *ths = new float[NUMTRIALS];
    float *gs = new float[NUMTRIALS];
    float *hs = new float[NUMTRIALS];
    float *ds = new float[NUMTRIALS];

    for (int n = 0; n < NUMTRIALS; n++) {
        vs[n] = Ranf(VMIN, VMAX);
        ths[n] = Ranf(THMIN, THMAX);
        gs[n] = Ranf(GMIN, GMAX);
        hs[n] = Ranf(HMIN, HMAX);
        ds[n] = Ranf(DMIN, DMAX);
    }

    double maxPerformance = 0.;
    int numHits = 0;

    for (int tries = 0; tries < NUMTRIES; tries++) {
        double time0 = omp_get_wtime();
        int hits = 0;

        #pragma omp parallel for default(none) shared(vs, ths, gs, hs, ds) reduction(+:hits)
        for (int n = 0; n < NUMTRIALS; n++) {
            float v = vs[n];
            float thr = Radians(ths[n]);
            float vx = v * cos(thr);
            float vy = v * sin(thr);
            float g = gs[n];
            float h = hs[n];
            float d = ds[n];

            float tGround = -vy / (0.5 * GRAVITY);
            float xGround = vx * tGround;
            if (xGround <= g)
                continue;

            float tCliff = g / vx;
            float yCliff = vy * tCliff + 0.5 * GRAVITY * tCliff * tCliff;
            if (yCliff <= h)
                continue;

            float A = 0.5 * GRAVITY;
            float B = vy;
            float C = -h;
            float disc = B * B - 4. * A * C;
            if (disc < 0.)
                continue;

            float sqrtDisc = sqrtf(disc);
            float t1 = (-B + sqrtDisc) / (2.f * A);
            float t2 = (-B - sqrtDisc) / (2.f * A);
            float tmax = fmax(t1, t2);

            float upperDist = vx * tmax - g;
            if (fabs(upperDist - d) <= TOL)
                hits++;
        }

        double time1 = omp_get_wtime();
        double megaTrialsPerSecond = (double)NUMTRIALS / (time1 - time0) / 1e6;
        if (megaTrialsPerSecond > maxPerformance) {
            maxPerformance = megaTrialsPerSecond;
            numHits = hits;
        }
    }

    float probability = (float)numHits / (float)NUMTRIALS;

#ifdef CSV
    // CSV line for spreadsheet use
    printf("%d,%d,%.2lf\n", NUMT, NUMTRIALS, maxPerformance);
#else
    // Verbose line for debugging
    fprintf(stderr, "%2d threads : %8d trials ; probability = %6.2f%% ; megatrials/sec = %6.2lf\n",
        NUMT, NUMTRIALS, 100. * probability, maxPerformance);
#endif

    delete[] vs;
    delete[] ths;
    delete[] gs;
    delete[] hs;
    delete[] ds;

    return 0;
}
