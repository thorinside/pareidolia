// Pareidolia - Spectral Voice Hallucination Plugin for disting NT
// A granular effect processor that exploits auditory pareidolia
// Copyright 2026 Nosuch
//
// Generates grain-based textures within formant-sensitive frequency ranges
// and applies spectral distortion to create "phantom choir" externalization effects.

#include <math.h>
#include <string.h>
#include <new>
#include <distingnt/api.h>

// ============================================================================
// Constants
// ============================================================================

// Sample rate — set from NT_globals in construct() and refreshed in step().
// File-scope so helper structs (SVF, OnePole, DCBlocker) can reference them.
// NOT initialised here because the NT plugin loader may not handle .data;
// construct() and step() always write these before first use.
static float gSampleRate;
static float gInvSampleRate;
static const float kTwoPi = 6.2831853071795864f;
static const float kPi = 3.1415926535897932f;

static const int kMaxGrains = 24;
static const float kMaxGrainMs = 2000.0f;  // max grain duration in ms
static const int kHannLUTSize = 4800;
static const int kAnalysisBufferSize = 2048;
static const int kCaptureBufferSize = 65536;
static const int kControlRateDiv = 32;     // control rate every 32 samples
static const int kNumAnalysisBands = 5;
// YIN constants
static const int kYinBufferSize = 2048;
static const float kYinThreshold = 0.15f;
static const float kMinPitchHz = 50.0f;
static const float kMaxPitchHz = 500.0f;
static const int kYinTausPerStep = 4;   // amortize diff function across step calls
static const float kResonatorModeBoost = 3.0f;
static const float kOutputSoftCeilingVolts = 8.0f; // ~16Vpp soft ceiling
static const float kGainSmoothingTimeSec = 0.005f; // 5ms smoothing
static const float kMaxSpawnPeriodSec = 8.0f; // allow sparse, rhythmically slow grains

// ============================================================================
// DSP Utilities
// ============================================================================

// Fast xorshift32 PRNG
struct PRNG {
    uint32_t state;

    void seed(uint32_t s) {
        state = s ? s : 0xDEADBEEF;
    }

    uint32_t next() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }

    // Returns float in [0, 1)
    float nextFloat() {
        return (next() & 0x7FFFFF) / (float)0x800000;
    }

    // Returns float in [-1, 1)
    float nextBipolar() {
        return nextFloat() * 2.0f - 1.0f;
    }

    // Returns float in [lo, hi)
    float nextRange(float lo, float hi) {
        return lo + nextFloat() * (hi - lo);
    }
};

// One-pole smoother for parameter interpolation
struct OnePole {
    float value;
    float coeff;

    void init(float initial, float timeConstantMs) {
        value = initial;
        setTimeConstant(timeConstantMs);
    }

    void setTimeConstant(float timeConstantMs) {
        float controlRateHz = gSampleRate / (float)kControlRateDiv;
        float timeConstantSec = timeConstantMs * 0.001f;
        coeff = 1.0f - exp(-1.0f / (timeConstantSec * controlRateHz));
    }

    float process(float target) {
        value += (target - value) * coeff;
        return value;
    }

    void set(float v) { value = v; }
};

// State Variable Filter (SVF)
// Chamberlin topology - simultaneous LP/BP/HP outputs
struct SVF {
    float ic1eq;  // integrator state 1
    float ic2eq;  // integrator state 2
    float g;      // frequency coefficient
    float k;      // damping (1/Q)
    float a1, a2, a3;  // pre-computed coefficients

    void reset() {
        ic1eq = 0.0f;
        ic2eq = 0.0f;
    }

    void setFreqQ(float freqHz, float Q) {
        g = tan(kPi * freqHz * gInvSampleRate);
        k = 1.0f / Q;
        a1 = 1.0f / (1.0f + g * (g + k));
        a2 = g * a1;
        a3 = g * a2;
    }

    // Returns bandpass output
    float processBandpass(float v0) {
        float v3 = v0 - ic2eq;
        float v1 = a1 * ic1eq + a2 * v3;
        float v2 = ic2eq + a2 * ic1eq + a3 * v3;
        ic1eq = 2.0f * v1 - ic1eq;
        ic2eq = 2.0f * v2 - ic2eq;
        return v1;
    }

    // Returns lowpass output
    float processLowpass(float v0) {
        float v3 = v0 - ic2eq;
        float v1 = a1 * ic1eq + a2 * v3;
        float v2 = ic2eq + a2 * ic1eq + a3 * v3;
        ic1eq = 2.0f * v1 - ic1eq;
        ic2eq = 2.0f * v2 - ic2eq;
        return v2;
    }

    // Returns highpass output
    float processHighpass(float v0) {
        float v3 = v0 - ic2eq;
        float v1 = a1 * ic1eq + a2 * v3;
        float v2 = ic2eq + a2 * ic1eq + a3 * v3;
        ic1eq = 2.0f * v1 - ic1eq;
        ic2eq = 2.0f * v2 - ic2eq;
        float hp = v0 - k * v1 - v2;
        return hp;
    }

    // Allpass output (LP - BP + HP = allpass for SVF)
    float processAllpass(float v0) {
        float v3 = v0 - ic2eq;
        float v1 = a1 * ic1eq + a2 * v3;
        float v2 = ic2eq + a2 * ic1eq + a3 * v3;
        ic1eq = 2.0f * v1 - ic1eq;
        ic2eq = 2.0f * v2 - ic2eq;
        float hp = v0 - k * v1 - v2;
        return v2 - k * v1 + hp;  // LP - k*BP + HP = allpass
    }

    // Process and return all outputs (LP, BP, HP)
    void processAll(float v0, float &lp, float &bp, float &hp) {
        float v3 = v0 - ic2eq;
        float v1 = a1 * ic1eq + a2 * v3;
        float v2 = ic2eq + a2 * ic1eq + a3 * v3;
        ic1eq = 2.0f * v1 - ic1eq;
        ic2eq = 2.0f * v2 - ic2eq;
        bp = v1;
        lp = v2;
        hp = v0 - k * v1 - v2;
    }
};

// High-shelf filter using SVF
// DC blocker - single-pole highpass
struct DCBlocker {
    float x1;
    float y1;
    float coeff;  // typically ~0.998 for ~15Hz at 48kHz

    void init(float cutoffHz) {
        x1 = y1 = 0.0f;
        coeff = 1.0f - (kTwoPi * cutoffHz * gInvSampleRate);
    }

    float process(float x) {
        y1 = x - x1 + coeff * y1;
        x1 = x;
        return y1;
    }
};

static inline float clampf(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static inline float lerpf(float a, float b, float t) {
    return a + (b - a) * t;
}

// Soft clipper (tanh approximation)
static inline float softClip(float x) {
    if (x > 1.5f) return 1.0f;
    if (x < -1.5f) return -1.0f;
    return x - (x * x * x) / 6.75f;
}

// Soft-clip with configurable ceiling. E.g. ceiling=2.5 -> ~5Vpp max.
static inline float softClipScaled(float x, float ceiling) {
    if (ceiling <= 0.0f) return 0.0f;
    return ceiling * softClip(x / ceiling);
}

// ============================================================================
// Grain Structures
// ============================================================================

enum GrainSourceMode {
    kGrainNoise = 0,
    kGrainInputSeeded = 1,
    kGrainResonator = 2,
};

struct Grain {
    bool active;
    int startDelay;     // samples to wait before grain starts outputting
    int position;       // current sample position within grain
    int duration;       // total grain duration in samples
    float hannPos;        // current LUT position
    float hannIncrement;  // LUT increment per sample
    int channel;        // 0 = left, 1 = right

    // Mode A: noise → SVF bandpass
    SVF noiseBPF;

    // Mode B: input-seeded
    int captureReadPos;     // starting position in capture buffer
    float resampleRatio;    // pitch shift ratio

    // Mode C: resonator
    SVF resoF1;
    SVF resoF2;
    SVF resoF3;
    float impulsePhase;
    float impulseRate;
    float f2Gain;
    float f3Gain;

    GrainSourceMode mode;
    GrainSourceMode prevMode;  // for detecting mode changes on slot reuse

    // Formant frequencies frozen at grain onset
    float f1Hz;
    float f2Hz;
    float f3Hz;
    float grainQ;
};

// ============================================================================
// Vowel targets
// ============================================================================

struct VowelTarget {
    float f1;
    float f2;
};

static const VowelTarget kVowelTargets[5] = {
    { 300.0f, 2300.0f },   // /i/ "see"
    { 700.0f, 1200.0f },   // /a/ "father"
    { 400.0f, 2000.0f },   // /e/ "say"
    { 600.0f, 1000.0f },   // /o/ "saw"
    { 300.0f,  900.0f },   // /u/ "who"
};

// Band center frequencies for spectral analysis
static const float kBandCenters[kNumAnalysisBands] = {
    200.0f,   // F1 region
    800.0f,   // F1 upper / F2 lower
    1800.0f,  // F2 region (primary voice)
    3750.0f,  // F3 / presence
    8000.0f,  // Air / spatial cues
};

// ============================================================================
// DTC (hot-path data in tightly-coupled memory)
// ============================================================================

struct _pareidolia_DTC {
    // DC blockers (L/R)
    DCBlocker dcBlockL;
    DCBlocker dcBlockR;

    // Active grain count
    int activeGrainCount;

    // Control-rate sample counter
    int controlCounter;
};

// ============================================================================
// Main Algorithm Structure (in SRAM)
// ============================================================================

struct _pareidoliaAlgorithm : public _NT_algorithm {
    _pareidoliaAlgorithm() {}
    ~_pareidoliaAlgorithm() {}

    // Pointer to DTC data
    _pareidolia_DTC* dtc;

    // PRNG
    PRNG rng;

    // Grain pool
    Grain grains[kMaxGrains];

    // Hann window LUT
    float hannLUT[kHannLUTSize];

    // Analysis buffer (mono-summed input for YIN)
    float analysisBuffer[kAnalysisBufferSize];
    // Snapshot of analysis buffer for amortized YIN (avoids overwrite during computation)
    float yinSnapshot[kAnalysisBufferSize];
    // YIN difference function buffer
    float yinDiffBuf[kYinBufferSize / 2];
    int analysisWritePos;
    bool analysisReady;

    // Amortized YIN state
    int yinTauProgress;     // how many tau values computed so far
    bool yinInProgress;     // true while amortized computation is running
    bool yinDiffDone;       // true when diff function is complete

    // Amortized spectral features state
    int spectralBandProgress;       // which band to compute next
    bool spectralInProgress;        // true while amortized computation is running
    float spectralTotalEnergy;      // accumulated across bands
    float spectralWeightedFreq;     // accumulated across bands

    // Analysis outputs
    float pitchEstimate;       // Hz
    float pitchConfidence;     // 0-1
    float heldPitchHz;         // smoothed fallback pitch when confidence drops
    float spectralCentroid;    // Hz
    float spectralFlux;        // 0-1
    float inputEnergy;         // linear RMS
    float prevBandEnergies[kNumAnalysisBands];
    float curBandEnergies[kNumAnalysisBands];

    // Vowel-space walk state
    int currentVowelTarget;
    float f1Current;
    float f2Current;
    float f1Final;
    float f2Final;
    float f3Final;
    float lfoPhase;

    // Grain scheduling state
    float grainAccumL;     // accumulated time since last grain (L)
    float grainAccumR;     // accumulated time since last grain (R)
    float nextGrainPeriodL;
    float nextGrainPeriodR;

    // Parameter smoothers
    OnePole smoothDensity;
    OnePole smoothGrainLength;
    OnePole smoothFormantCenter;
    OnePole smoothFormantDrift;
    OnePole smoothInputAtten;
    OnePole smoothDryWet;
    OnePole smoothCoherence;
    OnePole smoothInputTracking;

    // Feature toggle smoother (click-free enable/disable of pitch tracking)
    OnePole smoothPitchTrackEnable;
    float pPitchTrackEnable;

    // Analysis output smoothers (prevent clicks from stepped analysis updates)
    OnePole smoothCentroid;
    OnePole smoothFlux;


    // Smoothed parameter values (updated at control rate)
    float pDensity;
    float pGrainLength;
    float pFormantCenter;
    float pFormantDrift;
    float pInputAtten;
    float pDryWet;
    float pCoherence;
    float pInputTracking;

    // Clock input state
    bool clockHigh;            // Schmitt trigger state
    int clockBurstCount;       // total grains in this burst
    int clockBurstFired;       // grains fired so far
    int clockBurstElapsed;     // samples elapsed since trigger
    int clockBurstDelays[4];   // sorted fire times (samples after trigger)
    int clockPeriodSamples;    // measured period between rising edges (0 = invalid)
    int clockSampleCounter;    // samples since last rising edge

    // CV inputs
    float formantCVVolts;   // V/Oct CV for formant center

    // Effective parameters (after input tracking modulation)
    float effFormantCenter;
    float effDensity;
    float effInputGain;
    float effDryWet;

    // Audio-rate smoothed values to avoid clicks from control-rate jumps
    float smoothedOverlapComp;
    float smoothedInputGain;
    float smoothedDryWet;

    // --- DRAM pointers ---
    // Input capture buffer for Mode B (4096 × 2ch)
    float* captureBuffer;    // [kCaptureBufferSize * 2] interleaved L/R
    int captureWritePos;
};

// ============================================================================
// Parameter Definitions
// ============================================================================

enum {
    kParamInputL,
    kParamInputR,
    kParamOutputL,
    kParamOutputR,
    kParamOutputMode,
    kParamGrainSource,
    kParamGrainDensity,
    kParamGrainLength,
    kParamFormantCenter,
    kParamFormantDrift,
    kParamInputAtten,
    kParamDryWetMix,
    kParamCoherence,
    kParamInputTracking,
    kParamFormantCV,
    kParamPitchTrackEnable,
    kParamFreeze,
    kParamClockCV,
    kNumParams,
};

static const char* grainSourceStrings[] = {
    "Noise",
    "Input",
    "Resonator",
    NULL,
};

static const char* onOffStrings[] = {
    "Off",
    "On",
    NULL,
};

static const _NT_parameter parameters[] = {
    NT_PARAMETER_AUDIO_INPUT("Input L", 1, 1)
    NT_PARAMETER_AUDIO_INPUT("Input R", 1, 2)
    NT_PARAMETER_AUDIO_OUTPUT("Output L", 1, 13)
    NT_PARAMETER_AUDIO_OUTPUT("Output R", 1, 14)
    NT_PARAMETER_OUTPUT_MODE("Output mode")
    { .name = "Grain Source",   .min = 0, .max = 2,   .def = 2,  .unit = kNT_unitEnum,    .scaling = 0, .enumStrings = grainSourceStrings },
    { .name = "Grain Density",  .min = 0, .max = 100, .def = 35, .unit = kNT_unitPercent,  .scaling = 0, .enumStrings = NULL },
    { .name = "Grain Length",   .min = 0, .max = 100, .def = 25, .unit = kNT_unitPercent,  .scaling = 0, .enumStrings = NULL },
    { .name = "Formant Center", .min = 0, .max = 100, .def = 45, .unit = kNT_unitPercent,  .scaling = 0, .enumStrings = NULL },
    { .name = "Formant Drift",  .min = 0, .max = 100, .def = 35, .unit = kNT_unitPercent,  .scaling = 0, .enumStrings = NULL },
    { .name = "Input Atten",    .min = 0, .max = 100, .def = 25, .unit = kNT_unitPercent,  .scaling = 0, .enumStrings = NULL },
    { .name = "Dry/Wet Mix",    .min = 0, .max = 100, .def = 55, .unit = kNT_unitPercent,  .scaling = 0, .enumStrings = NULL },
    { .name = "Coherence",      .min = 0, .max = 100, .def = 60, .unit = kNT_unitPercent,  .scaling = 0, .enumStrings = NULL },
    { .name = "Input Tracking", .min = 0, .max = 100, .def = 70, .unit = kNT_unitPercent,  .scaling = 0, .enumStrings = NULL },
    NT_PARAMETER_CV_INPUT("Formant CV", 0, 0)
    { .name = "Pitch Track",   .min = 0, .max = 1, .def = 0, .unit = kNT_unitEnum, .scaling = 0, .enumStrings = onOffStrings },
    { .name = "Freeze",   .min = 0, .max = 1, .def = 0, .unit = kNT_unitEnum, .scaling = 0, .enumStrings = onOffStrings },
    NT_PARAMETER_CV_INPUT("Clock", 0, 0)
};

// Page definitions
static const uint8_t pageMain[] = {
    kParamGrainSource, kParamGrainDensity, kParamGrainLength,
    kParamFormantCenter, kParamFormantDrift,
    kParamInputAtten, kParamDryWetMix, kParamCoherence, kParamInputTracking,
    kParamPitchTrackEnable, kParamFreeze,
};

static const uint8_t pageRouting[] = {
    kParamInputL, kParamInputR, kParamOutputL, kParamOutputR,
    kParamOutputMode, kParamFormantCV, kParamClockCV,
};

static const _NT_parameterPage pages[] = {
    { .name = "Main",    .numParams = ARRAY_SIZE(pageMain),    .group = 0, .unused = {0,0}, .params = pageMain },
    { .name = "Routing", .numParams = ARRAY_SIZE(pageRouting), .group = 0, .unused = {0,0}, .params = pageRouting },
};

static const _NT_parameterPages parameterPages = {
    .numPages = ARRAY_SIZE(pages),
    .pages = pages,
};

// ============================================================================
// Forward declarations
// ============================================================================

static void updateControlRate(_pareidoliaAlgorithm* alg, int blockOffsetSamples);
static void updateFormantDrift(_pareidoliaAlgorithm* alg);
static void scheduleGrains(_pareidoliaAlgorithm* alg, int blockOffsetSamples);
static bool fireGrain(_pareidoliaAlgorithm* alg, int channel, int startDelaySamples);
static float processGrainSample(_pareidoliaAlgorithm* alg, Grain& grain);
static int chooseSpawnDelay(_pareidoliaAlgorithm* alg, float basePeriod, uint32_t& usedMask);
static void yinBegin(_pareidoliaAlgorithm* alg);
static void yinStepIncremental(_pareidoliaAlgorithm* alg);
static void yinFinalize(_pareidoliaAlgorithm* alg);
static void spectralFeaturesBegin(_pareidoliaAlgorithm* alg);
static void spectralFeaturesStep(_pareidoliaAlgorithm* alg);

// ============================================================================
// Hann window generation
// ============================================================================

static void generateHannLUT(float* lut, int size) {
    for (int i = 0; i < size; ++i) {
        float phase = (float)i / (float)(size - 1);
        lut[i] = 0.5f * (1.0f - cos(kTwoPi * phase));
    }
}

// ============================================================================
// YIN Pitch Detection
// ============================================================================

// Begin amortized YIN: snapshot the buffer and start computing
static void yinBegin(_pareidoliaAlgorithm* alg) {
    memcpy(alg->yinSnapshot, alg->analysisBuffer, sizeof(alg->analysisBuffer));
    alg->yinDiffBuf[0] = 0.0f;
    alg->yinTauProgress = 1;
    alg->yinInProgress = true;
    alg->yinDiffDone = false;
}

// Compute a chunk of the YIN difference function (called each step)
static void yinStepIncremental(_pareidoliaAlgorithm* alg) {
    if (!alg->yinInProgress || alg->yinDiffDone) return;

    const float* buf = alg->yinSnapshot;
    const int W = kYinBufferSize / 2;
    float* d = alg->yinDiffBuf;

    int end = alg->yinTauProgress + kYinTausPerStep;
    if (end > W) end = W;

    for (int tau = alg->yinTauProgress; tau < end; ++tau) {
        float sum = 0.0f;
        for (int j = 0; j < W; ++j) {
            float diff = buf[j] - buf[j + tau];
            sum += diff * diff;
        }
        d[tau] = sum;
    }

    alg->yinTauProgress = end;
    if (alg->yinTauProgress >= W) {
        alg->yinDiffDone = true;
    }
}

// Finalize YIN: CMND, threshold, parabolic interpolation (cheap, runs once)
static void yinFinalize(_pareidoliaAlgorithm* alg) {
    const int W = kYinBufferSize / 2;
    float* d = alg->yinDiffBuf;

    // Step 2: Cumulative mean normalized difference function (CMND)
    float runningSum = 0.0f;
    d[0] = 1.0f;
    for (int tau = 1; tau < W; ++tau) {
        runningSum += d[tau];
        if (runningSum > 0.0f)
            d[tau] = d[tau] * (float)tau / runningSum;
        else
            d[tau] = 1.0f;
    }

    // Step 3: Absolute threshold - find first dip below threshold
    int minTau = (int)(gSampleRate / kMaxPitchHz);
    int maxTau = (int)(gSampleRate / kMinPitchHz);
    if (minTau < 2) minTau = 2;
    if (maxTau >= W) maxTau = W - 1;

    int bestTau = -1;
    for (int tau = minTau; tau < maxTau; ++tau) {
        if (d[tau] < kYinThreshold) {
            while (tau + 1 < maxTau && d[tau + 1] < d[tau])
                ++tau;
            bestTau = tau;
            break;
        }
    }

    if (bestTau < 0) {
        float minVal = d[minTau];
        bestTau = minTau;
        for (int tau = minTau + 1; tau < maxTau; ++tau) {
            if (d[tau] < minVal) {
                minVal = d[tau];
                bestTau = tau;
            }
        }
        alg->pitchConfidence = clampf(1.0f - d[bestTau], 0.0f, 1.0f) * 0.5f;
    } else {
        alg->pitchConfidence = clampf(1.0f - d[bestTau], 0.0f, 1.0f);
    }

    // Step 4: Parabolic interpolation
    float betterTau = (float)bestTau;
    if (bestTau > 0 && bestTau < W - 1) {
        float s0 = d[bestTau - 1];
        float s1 = d[bestTau];
        float s2 = d[bestTau + 1];
        float denom = 2.0f * s1 - s2 - s0;
        if (denom > 0.0001f) {
            betterTau = (float)bestTau + (s0 - s2) / (2.0f * denom);
        }
    }

    if (betterTau > 0.0f) {
        alg->pitchEstimate = clampf(gSampleRate / betterTau, kMinPitchHz, kMaxPitchHz);
    }

    alg->yinInProgress = false;
}

// ============================================================================
// Spectral Feature Computation (using filterbank energies)
// ============================================================================

// Begin amortized spectral feature computation
static void spectralFeaturesBegin(_pareidoliaAlgorithm* alg) {
    alg->spectralBandProgress = 0;
    alg->spectralInProgress = true;
    alg->spectralTotalEnergy = 0.0f;
    alg->spectralWeightedFreq = 0.0f;
}

// Compute one band per call (amortized across step() calls)
static void spectralFeaturesStep(_pareidoliaAlgorithm* alg) {
    if (!alg->spectralInProgress) return;

    int b = alg->spectralBandProgress;

    if (b < kNumAnalysisBands) {
        // Goertzel for this band
        const float* buf = alg->yinSnapshot;
        const int N = kAnalysisBufferSize;
        float freq = kBandCenters[b] * gInvSampleRate;
        float coeff = 2.0f * cos(2.0f * kPi * freq);
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
        for (int i = 0; i < N; ++i) {
            s0 = buf[i] + coeff * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        float energy = sqrt((s1 * s1 + s2 * s2 - coeff * s1 * s2) / (float)(N * N));
        alg->curBandEnergies[b] = energy;
        alg->spectralTotalEnergy += energy;
        alg->spectralWeightedFreq += energy * kBandCenters[b];
        alg->spectralBandProgress++;
    } else {
        // All bands done — finalize centroid, flux, energy
        if (alg->spectralTotalEnergy > 0.00001f)
            alg->spectralCentroid = alg->spectralWeightedFreq / alg->spectralTotalEnergy;
        else
            alg->spectralCentroid = 1500.0f;

        float flux = 0.0f;
        for (int i = 0; i < kNumAnalysisBands; ++i) {
            float diff = alg->curBandEnergies[i] - alg->prevBandEnergies[i];
            flux += fabs(diff);
            alg->prevBandEnergies[i] = alg->curBandEnergies[i];
        }
        flux /= (float)kNumAnalysisBands;
        alg->spectralFlux = clampf(flux * 20.0f, 0.0f, 1.0f);

        // RMS energy
        const float* buf = alg->yinSnapshot;
        const int N = kAnalysisBufferSize;
        float rms = 0.0f;
        for (int i = 0; i < N; ++i) {
            rms += buf[i] * buf[i];
        }
        alg->inputEnergy = sqrt(rms / (float)N);

        alg->spectralInProgress = false;
    }
}

// ============================================================================
// Grain Scheduling and Firing
// ============================================================================

static int findFreeGrainSlot(_pareidoliaAlgorithm* alg) {
    // Find an inactive slot.
    for (int i = 0; i < kMaxGrains; ++i) {
        if (!alg->grains[i].active) return i;
    }
    // If all slots are occupied, drop this grain spawn instead of hard-stealing
    // an active grain. This avoids discontinuities that cause pops.
    return -1;
}

// Choose a spawn delay within the current control block while avoiding
// same-sample collisions for grains launched on the same channel/tick.
static int chooseSpawnDelay(_pareidoliaAlgorithm* alg, float basePeriod, uint32_t& usedMask) {
    if (basePeriod >= (float)(kControlRateDiv * 1.5f)) {
        return 0;
    }
    if (usedMask == 0xFFFFFFFFu) {
        return 0;
    }

    int start = (int)(alg->rng.nextFloat() * (float)kControlRateDiv);
    if (start < 0) start = 0;
    if (start >= kControlRateDiv) start = kControlRateDiv - 1;

    for (int i = 0; i < kControlRateDiv; ++i) {
        int idx = (start + i) % kControlRateDiv;
        uint32_t bit = 1u << idx;
        if ((usedMask & bit) == 0u) {
            usedMask |= bit;
            return idx;
        }
    }
    return 0;
}

static bool fireGrain(_pareidoliaAlgorithm* alg, int channel, int startDelaySamples) {
    int slot = findFreeGrainSlot(alg);
    if (slot < 0) return false;
    Grain& g = alg->grains[slot];

    g.active = true;
    if (startDelaySamples < 0) startDelaySamples = 0;
    g.startDelay = startDelaySamples;
    g.position = 0;
    g.channel = channel;

    // Current grain source mode
    g.mode = (GrainSourceMode)alg->v[kParamGrainSource];

    // Freeze formant frequencies at grain onset
    g.f1Hz = alg->f1Final;
    g.f2Hz = alg->f2Final;
    g.f3Hz = alg->f3Final;

    // Grain duration from Grain Length param (0%=10ms, 100%=2000ms).
    // Use a linear mapping so mid/high knob positions produce clearly longer grains.
    float durMs = lerpf(10.0f, 2000.0f, alg->pGrainLength);
    g.duration = (int)(durMs * 0.001f * gSampleRate);
    if (g.duration < 48) g.duration = 48;
    if (g.duration > (int)(kMaxGrainMs * 0.001f * gSampleRate)) g.duration = (int)(kMaxGrainMs * 0.001f * gSampleRate);

    // Grain Q from density
    g.grainQ = lerpf(5.0f, 15.0f, alg->effDensity);

    switch (g.mode) {
        case kGrainNoise: {
            // Mode A: noise -> formant-focused bandpass.
            // Bias toward F2/F3 to preserve intelligible "vocal air" content.
            float sel = alg->rng.nextFloat();
            float bpFreq;
            if (sel < 0.25f)
                bpFreq = g.f1Hz;
            else if (sel < 0.85f)
                bpFreq = g.f2Hz;
            else
                bpFreq = g.f3Hz;
            bpFreq *= alg->rng.nextRange(0.96f, 1.04f);
            if (g.prevMode != kGrainNoise) {
                g.noiseBPF.reset();
            }
            g.noiseBPF.setFreqQ(clampf(bpFreq, 180.0f, 4200.0f), g.grainQ * 1.15f);
            break;
        }
        case kGrainInputSeeded: {
            // Mode B: input-seeded
            // Subtle pitch shift: formant center controls ±1 octave (0.5x to 2.0x)
            g.resampleRatio = lerpf(0.5f, 2.0f, alg->effFormantCenter);
            // Clamp duration so grain doesn't wrap past capture buffer
            int maxDuration = (int)((float)kCaptureBufferSize / g.resampleRatio);
            if (g.duration > maxDuration) g.duration = maxDuration;
            // Keep read head safely behind the live write head for the entire grain.
            // This avoids ratio-dependent discontinuities when resampling faster than 1x.
            float maxReadAdvance = (float)(g.duration - 1) * g.resampleRatio + 2.0f;
            int minLag = (int)maxReadAdvance + 16;
            int maxLag = kCaptureBufferSize - 2;
            if (minLag < 2) minLag = 2;
            if (minLag > maxLag) minLag = maxLag;
            int lagRange = maxLag - minLag;
            int readOffset = minLag;
            if (lagRange > 0) {
                readOffset += (int)(alg->rng.nextFloat() * (float)(lagRange + 1));
            }
            g.captureReadPos = (alg->captureWritePos - readOffset + kCaptureBufferSize) & (kCaptureBufferSize - 1);
            // Bandpass tracks shifted pitch
            float bpFreq = (alg->pitchConfidence > 0.5f)
                ? alg->pitchEstimate * g.resampleRatio
                : lerpf(300.0f, 2000.0f, alg->effFormantCenter);
            if (g.prevMode != kGrainInputSeeded) {
                g.noiseBPF.reset();
            }
            g.noiseBPF.setFreqQ(clampf(bpFreq, 100.0f, 4000.0f), g.grainQ * 0.3f);
            break;
        }
        case kGrainResonator: {
            // Resonator benefits from slightly longer grains to read as
            // "whispered voice" rather than short noisy ticks.
            g.duration = (int)((float)g.duration * 1.75f);
            int minResoDur = (int)(0.035f * gSampleRate);  // 35ms
            if (g.duration < minResoDur) g.duration = minResoDur;
            if (g.duration > (int)(kMaxGrainMs * 0.001f * gSampleRate)) g.duration = (int)(kMaxGrainMs * 0.001f * gSampleRate);

            // Mode C: impulse-excited resonator
            // Only reset filter state on mode change — allow natural ring-up
            // when reusing a slot that was already in resonator mode
            if (g.prevMode != kGrainResonator) {
                g.resoF1.reset();
                g.resoF2.reset();
                g.resoF3.reset();
            }
            // Lower-Q tuning reduces narrow resonant spikes on specific notes.
            g.resoF1.setFreqQ(g.f1Hz, lerpf(6.0f, 8.0f, alg->effDensity));
            g.resoF2.setFreqQ(g.f2Hz, lerpf(7.0f, 10.0f, alg->effDensity));
            g.resoF3.setFreqQ(g.f3Hz, 10.0f);
            g.f2Gain = 0.562f;   // -5 dB
            g.f3Gain = 0.251f;   // -12 dB

            // Impulse rate
            float basePitch = (alg->pitchConfidence >= 0.35f)
                ? alg->pitchEstimate
                : alg->heldPitchHz;
            if (basePitch < 60.0f)
                basePitch = alg->rng.nextRange(110.0f, 140.0f);
            g.impulseRate = basePitch * alg->rng.nextRange(0.98f, 1.02f);
            g.impulsePhase = 0.0f;
            break;
        }
    }

    // Compute envelope stride after all mode-dependent duration adjustments.
    g.hannPos = 0.0f;
    g.hannIncrement = (g.duration > 1)
        ? (float)(kHannLUTSize - 1) / (float)(g.duration - 1)
        : 0.0f;

    g.prevMode = g.mode;
    return true;
}

static float processGrainSample(_pareidoliaAlgorithm* alg, Grain& grain) {
    float sample = 0.0f;

    switch (grain.mode) {
        case kGrainNoise: {
            // Generate white noise and filter
            float noise = alg->rng.nextBipolar();
            sample = grain.noiseBPF.processBandpass(noise);
            break;
        }
        case kGrainInputSeeded: {
            // Read from capture buffer with resampling
            float readPos = (float)grain.captureReadPos + (float)grain.position * grain.resampleRatio;
            int readPosInt = (int)readPos;
            float frac = readPos - (float)readPosInt;
            int idx = readPosInt & (kCaptureBufferSize - 1);
            int idx2 = (idx + 1) & (kCaptureBufferSize - 1);
            int chOffset = grain.channel * kCaptureBufferSize;
            float s0 = alg->captureBuffer[chOffset + idx];
            float s1 = alg->captureBuffer[chOffset + idx2];
            float raw = s0 + frac * (s1 - s0);
            // Shape with bandpass
            sample = grain.noiseBPF.processBandpass(raw);
            break;
        }
        case kGrainResonator: {
            // Generate impulse train
            grain.impulsePhase += grain.impulseRate * gInvSampleRate;
            float impulse = 0.0f;
            if (grain.impulsePhase >= 1.0f) {
                grain.impulsePhase -= 1.0f;
                impulse = 1.0f;
            }
            // Parallel resonators
            float f1out = grain.resoF1.processBandpass(impulse);
            float f2out = grain.resoF2.processBandpass(impulse);
            float f3out = grain.resoF3.processBandpass(impulse);
            sample = f1out + f2out * grain.f2Gain + f3out * grain.f3Gain;
            // De-emphasize higher impulse rates where ringing tends to spike.
            float pitchDamp = clampf(180.0f / grain.impulseRate, 0.55f, 1.0f);
            sample *= pitchDamp;
            sample *= kResonatorModeBoost;
            break;
        }
    }

    // Apply Hann envelope
    int lutIdx = (int)grain.hannPos;
    if (lutIdx >= kHannLUTSize) lutIdx = kHannLUTSize - 1;
    if (lutIdx < 0) lutIdx = 0;
    float env = alg->hannLUT[lutIdx];
    sample *= env;
    grain.hannPos += grain.hannIncrement;

    return sample;
}

static void scheduleGrains(_pareidoliaAlgorithm* alg, int blockOffsetSamples) {
    uint32_t usedDelayMaskL = 0u;
    uint32_t usedDelayMaskR = 0u;

    bool clocked = (alg->clockPeriodSamples > 0);

    // Legacy edge-burst path is disabled while clocked; clock sync now relies
    // on period-based scheduling so Grain Length remains perceptually useful.
    if (!clocked) {
        while (alg->clockBurstFired < alg->clockBurstCount &&
               alg->clockBurstElapsed >= alg->clockBurstDelays[alg->clockBurstFired]) {
            if (!fireGrain(alg, 0, blockOffsetSamples)) break;  // no free slots
            if (alg->rng.nextFloat() < alg->pCoherence) {
                fireGrain(alg, 1, blockOffsetSamples);  // paired R grain
            }
            alg->clockBurstFired++;
        }
        if (alg->clockBurstFired < alg->clockBurstCount) {
            alg->clockBurstElapsed += kControlRateDiv;
        }
    } else {
        alg->clockBurstCount = 0;
        alg->clockBurstFired = 0;
        alg->clockBurstElapsed = 0;
    }

    float density = alg->effDensity;
    if (density < 0.01f) density = 0.01f;

    float basePeriod;
    float jitterAmt;

    if (clocked) {
        // Clock-synced mode:
        // density selects rhythmic multiplier including slower-than-clock rates.
        static const float kClockRateMult[8] = {
            0.25f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f
        };
        int rateIdx = (int)(density * 7.999f);  // 0..7
        if (rateIdx < 0) rateIdx = 0;
        if (rateIdx > 7) rateIdx = 7;
        basePeriod = (float)alg->clockPeriodSamples / kClockRateMult[rateIdx];
        jitterAmt = 0.0f;
    } else {
        float syncPitch = (alg->pitchConfidence >= 0.35f) ? alg->pitchEstimate : alg->heldPitchHz;
        if (alg->pInputTracking > 0.1f && syncPitch > 60.0f) {
            basePeriod = gSampleRate / (syncPitch * density);
            jitterAmt = 0.12f;
        } else {
            basePeriod = gSampleRate / (100.0f * density);
            jitterAmt = 0.24f;
        }
    }

    // Limit to reasonable range
    float maxPeriod = gSampleRate * kMaxSpawnPeriodSec;
    basePeriod = clampf(basePeriod, 24.0f, maxPeriod);

    float coherence = alg->pCoherence;

    // Advance accumulators once per control tick.
    alg->grainAccumL += (float)kControlRateDiv;
    alg->grainAccumR += (float)kControlRateDiv;

    // Left channel scheduling with carry-over timing (no hard reset to 0).
    int lFiredCount = 0;
    while (alg->grainAccumL >= alg->nextGrainPeriodL) {
        int startDelay = chooseSpawnDelay(alg, basePeriod, usedDelayMaskL);
        if (!fireGrain(alg, 0, blockOffsetSamples + startDelay)) {
            // Hold near threshold and retry on next control tick.
            alg->grainAccumL = alg->nextGrainPeriodL - 1.0f;
            break;
        }
        lFiredCount++;
        alg->grainAccumL -= alg->nextGrainPeriodL;
        float jitter = alg->rng.nextRange(-jitterAmt, jitterAmt) * basePeriod;
        alg->nextGrainPeriodL = clampf(basePeriod + jitter, 24.0f, maxPeriod);
    }

    // Paired R grains for coherence (one chance per L onset event).
    for (int i = 0; i < lFiredCount; ++i) {
        if (alg->rng.nextFloat() < coherence) {
            int startDelay = chooseSpawnDelay(alg, basePeriod, usedDelayMaskR);
            if (fireGrain(alg, 1, blockOffsetSamples + startDelay)) {
                // Nudge R accumulator back to avoid immediate double-fire.
                alg->grainAccumR *= 0.5f;
                float jitter = alg->rng.nextRange(-jitterAmt, jitterAmt) * basePeriod;
                alg->nextGrainPeriodR = clampf(basePeriod + jitter, 24.0f, maxPeriod);
            } else {
                alg->grainAccumR = alg->nextGrainPeriodR - 1.0f;
                break;
            }
        }
    }

    // Independent R schedule with carry-over timing.
    while (alg->grainAccumR >= alg->nextGrainPeriodR) {
        int startDelay = chooseSpawnDelay(alg, basePeriod, usedDelayMaskR);
        if (!fireGrain(alg, 1, blockOffsetSamples + startDelay)) {
            alg->grainAccumR = alg->nextGrainPeriodR - 1.0f;
            break;
        }
        alg->grainAccumR -= alg->nextGrainPeriodR;
        float jitter = alg->rng.nextRange(-jitterAmt, jitterAmt) * basePeriod;
        alg->nextGrainPeriodR = clampf(basePeriod + jitter, 24.0f, maxPeriod);
    }
}

// ============================================================================
// Formant Drift (Vowel-Space Walk)
// ============================================================================

static void updateFormantDrift(_pareidoliaAlgorithm* alg) {
    float drift = alg->pFormantDrift;
    float centerParam = alg->effFormantCenter;

    // Stochastic vowel target switching
    float switchProb = 0.02f * drift;  // higher drift = more frequent switches
    if (alg->rng.nextFloat() < switchProb) {
        int newTarget;
        int attempts = 0;
        do {
            newTarget = (int)(alg->rng.nextFloat() * 5.0f);
            if (newTarget >= 5) newTarget = 4;
        } while (newTarget == alg->currentVowelTarget && drift > 0.01f && ++attempts < 10);
        alg->currentVowelTarget = newTarget;
    }

    // Smooth interpolation toward current target
    const VowelTarget& target = kVowelTargets[alg->currentVowelTarget];
    float interpRate = 0.005f + 0.02f * drift;
    alg->f1Current += (target.f1 - alg->f1Current) * interpRate;
    alg->f2Current += (target.f2 - alg->f2Current) * interpRate;

    // LFO modulation (0.5-3Hz, depth scaled by drift)
    float lfoRate = lerpf(0.5f, 3.0f, drift);
    float controlRateHz = gSampleRate / (float)kControlRateDiv;
    alg->lfoPhase += lfoRate / controlRateHz;
    if (alg->lfoPhase > 1.0f) alg->lfoPhase -= 1.0f;

    // Fast parabolic sine approximation (good enough for LFO)
    // Maps phase [0,1] to sine [-1,1]
    float p1 = alg->lfoPhase * 2.0f - 1.0f;  // [-1, 1]
    float lfoSin = 4.0f * p1 * (1.0f - fabs(p1));  // parabolic approx
    float p2 = (alg->lfoPhase * 1.1f);
    p2 -= (int)p2;  // wrap to [0,1]
    p2 = p2 * 2.0f - 1.0f;
    float lfoCos = 4.0f * p2 * (1.0f - fabs(p2));
    float lfoDepth = 80.0f * drift;

    // Bias formant targets based on Formant Center parameter
    // centerParam 0→F1 region, 0.5→F2, 1.0→F3
    float f1Bias = lerpf(1.2f, 0.8f, centerParam);
    float f2Bias = lerpf(0.7f, 1.3f, centerParam);

    alg->f1Final = alg->f1Current * f1Bias + lfoSin * lfoDepth;
    alg->f2Final = alg->f2Current * f2Bias + lfoCos * lfoDepth;

    // Clamp to safe ranges
    alg->f1Final = clampf(alg->f1Final, 200.0f, 800.0f);
    alg->f2Final = clampf(alg->f2Final, 700.0f, 2500.0f);

    // F3 derived from F2 + 1000Hz + stochastic offset
    float stochOffset = alg->rng.nextRange(-50.0f, 50.0f) * drift;
    alg->f3Final = alg->f2Final + 1000.0f + stochOffset;
    alg->f3Final = clampf(alg->f3Final, 2000.0f, 4000.0f);
}

// ============================================================================
// Control Rate Update
// ============================================================================

static void updateControlRate(_pareidoliaAlgorithm* alg, int blockOffsetSamples) {
    // Smooth parameters
    float rawDensity       = (float)alg->v[kParamGrainDensity] / 100.0f;
    float rawGrainLength   = (float)alg->v[kParamGrainLength] / 100.0f;
    float rawFormantCenter = (float)alg->v[kParamFormantCenter] / 100.0f;
    float rawFormantDrift  = (float)alg->v[kParamFormantDrift] / 100.0f;
    float rawInputAtten    = (float)alg->v[kParamInputAtten] / 100.0f;
    float rawDryWet        = (float)alg->v[kParamDryWetMix] / 100.0f;
    float rawCoherence     = (float)alg->v[kParamCoherence] / 100.0f;
    float rawTracking      = (float)alg->v[kParamInputTracking] / 100.0f;

    alg->pDensity        = alg->smoothDensity.process(rawDensity);
    alg->pGrainLength    = alg->smoothGrainLength.process(rawGrainLength);
    alg->pFormantCenter  = alg->smoothFormantCenter.process(rawFormantCenter);
    alg->pFormantDrift   = alg->smoothFormantDrift.process(rawFormantDrift);
    alg->pInputAtten     = alg->smoothInputAtten.process(rawInputAtten);
    if (rawDryWet <= 0.0001f) {
        alg->smoothDryWet.set(0.0f);
        alg->pDryWet = 0.0f;
    } else {
        alg->pDryWet = alg->smoothDryWet.process(rawDryWet);
    }
    alg->pCoherence      = alg->smoothCoherence.process(rawCoherence);
    alg->pInputTracking  = alg->smoothInputTracking.process(rawTracking);

    // Pitch tracking toggle smoother
    alg->pPitchTrackEnable = alg->smoothPitchTrackEnable.process((float)alg->v[kParamPitchTrackEnable]);

    // Input attenuation: 0% = unity, 100% = heavy attenuation.
    // Knob is squared for finer low-attenuation control.
    float attenShape = alg->pInputAtten * alg->pInputAtten;
    alg->effInputGain = lerpf(1.0f, 0.10f, attenShape);

    // Hold last valid pitch estimate (only when pitch tracking is active)
    if (alg->pPitchTrackEnable > 0.01f) {
        if (alg->pitchConfidence >= 0.45f) {
            alg->heldPitchHz += (alg->pitchEstimate - alg->heldPitchHz) * 0.35f;
        } else {
            alg->heldPitchHz += (140.0f - alg->heldPitchHz) * 0.01f;
        }
        alg->heldPitchHz = clampf(alg->heldPitchHz, 80.0f, 320.0f);
    }

    // Compute effective parameters with input tracking modulation
    float tracking = alg->pInputTracking;
    float cvOffset = alg->formantCVVolts * (1.0f / 4.3f);

    // Smooth analysis outputs to prevent stepped clicks
    float smoothedCentroid = alg->smoothCentroid.process(alg->spectralCentroid);
    float smoothedFlux = alg->smoothFlux.process(alg->spectralFlux);
    // 1. Formant center bias from spectral centroid + V/Oct CV
    float centroidOffset = clampf((smoothedCentroid - 1500.0f) / 3000.0f, -0.3f, 0.3f);
    alg->effFormantCenter = clampf(alg->pFormantCenter + centroidOffset * tracking + cvOffset, 0.0f, 1.0f);

    // 2. Grain density scaling from spectral flux
    float fluxScale = lerpf(1.0f, lerpf(0.3f, 1.5f, smoothedFlux), tracking);
    alg->effDensity = clampf(alg->pDensity * fluxScale, 0.01f, 1.0f);

    alg->effDryWet = alg->pDryWet;

    // Update formant drift
    updateFormantDrift(alg);

    // Schedule grains
    scheduleGrains(alg, blockOffsetSamples);

}

// ============================================================================
// Factory Functions
// ============================================================================

static void calculateRequirements(_NT_algorithmRequirements& req, const int32_t* specifications) {
    req.numParameters = ARRAY_SIZE(parameters);
    req.sram = sizeof(_pareidoliaAlgorithm);
    req.dram = kCaptureBufferSize * 2 * sizeof(float);  // capture buffer (L+R)
    req.dtc = sizeof(_pareidolia_DTC);
    req.itc = 0;
}

static _NT_algorithm* construct(const _NT_algorithmMemoryPtrs& ptrs,
                                 const _NT_algorithmRequirements& req,
                                 const int32_t* specifications) {
    _pareidoliaAlgorithm* alg = new (ptrs.sram) _pareidoliaAlgorithm();
    alg->parameters = parameters;
    alg->parameterPages = &parameterPages;

    // Read actual sample rate from hardware/emulator
    gSampleRate = (float)NT_globals.sampleRate;
    gInvSampleRate = 1.0f / gSampleRate;

#if defined(__ARM_ARCH_7EM__)
    // Enable flush-to-zero and default-NaN to prevent denormal slowdown
    uint32_t fpscr;
    __asm__ volatile("vmrs %0, fpscr" : "=r"(fpscr));
    fpscr |= (1 << 24) | (1 << 25);  // FTZ + DNZ
    __asm__ volatile("vmsr fpscr, %0" : : "r"(fpscr));
#endif

    // Setup DTC
    alg->dtc = (_pareidolia_DTC*)ptrs.dtc;

    // Setup DRAM: capture buffer
    alg->captureBuffer = (float*)ptrs.dram;
    alg->captureWritePos = 0;

    // Initialize PRNG
    alg->rng.seed(0x50617265);  // "Pare"

    // Generate Hann LUT
    generateHannLUT(alg->hannLUT, kHannLUTSize);

    // Initialize all grains as inactive
    for (int i = 0; i < kMaxGrains; ++i) {
        alg->grains[i].active = false;
        alg->grains[i].prevMode = kGrainNoise;
    }

    // Initialize analysis
    memset(alg->analysisBuffer, 0, sizeof(alg->analysisBuffer));
    alg->analysisWritePos = 0;
    alg->analysisReady = false;
    alg->yinInProgress = false;
    alg->yinDiffDone = false;
    alg->yinTauProgress = 0;
    alg->spectralBandProgress = 0;
    alg->spectralInProgress = false;
    alg->spectralTotalEnergy = 0.0f;
    alg->spectralWeightedFreq = 0.0f;
    alg->pitchEstimate = 150.0f;
    alg->pitchConfidence = 0.0f;
    alg->heldPitchHz = 140.0f;
    alg->spectralCentroid = 1500.0f;
    alg->spectralFlux = 0.0f;
    alg->inputEnergy = 0.0f;
    memset(alg->prevBandEnergies, 0, sizeof(alg->prevBandEnergies));
    memset(alg->curBandEnergies, 0, sizeof(alg->curBandEnergies));

    // Initialize vowel-space walk
    alg->currentVowelTarget = 0;
    alg->f1Current = kVowelTargets[0].f1;
    alg->f2Current = kVowelTargets[0].f2;
    alg->f1Final = alg->f1Current;
    alg->f2Final = alg->f2Current;
    alg->f3Final = alg->f2Current + 1000.0f;
    alg->lfoPhase = 0.0f;

    // Initialize grain scheduling
    alg->grainAccumL = 0.0f;
    alg->grainAccumR = 0.0f;
    alg->nextGrainPeriodL = 480.0f;
    alg->nextGrainPeriodR = 480.0f;

    // Initialize parameter smoothers at default values (faster response).
    alg->smoothDensity.init(0.35f, 6.0f);
    alg->smoothGrainLength.init(0.25f, 8.0f);
    alg->smoothFormantCenter.init(0.45f, 10.0f);
    alg->smoothFormantDrift.init(0.35f, 9.0f);
    alg->smoothInputAtten.init(0.25f, 7.0f);
    alg->smoothDryWet.init(0.55f, 6.0f);
    alg->smoothCoherence.init(0.60f, 9.0f);
    alg->smoothInputTracking.init(0.70f, 9.0f);

    // Feature toggle smoother (30ms crossfade, default off)
    alg->smoothPitchTrackEnable.init(0.0f, 30.0f);
    alg->pPitchTrackEnable = 0.0f;

    // Analysis output smoothers
    alg->smoothCentroid.init(1500.0f, 35.0f);
    alg->smoothFlux.init(0.0f, 20.0f);

    // Set smoothed values to defaults
    alg->pDensity = 0.35f;
    alg->pGrainLength = 0.25f;
    alg->pFormantCenter = 0.45f;
    alg->pFormantDrift = 0.35f;
    alg->pInputAtten = 0.25f;
    alg->pDryWet = 0.55f;
    alg->pCoherence = 0.60f;
    alg->pInputTracking = 0.70f;

    alg->clockHigh = false;
    alg->clockBurstCount = 0;
    alg->clockBurstFired = 0;
    alg->clockBurstElapsed = 0;
    memset(alg->clockBurstDelays, 0, sizeof(alg->clockBurstDelays));
    alg->clockPeriodSamples = 0;
    alg->clockSampleCounter = 0;

    alg->smoothedOverlapComp = 1.0f;
    alg->smoothedInputGain = 0.95f;
    alg->smoothedDryWet = 0.55f;

    alg->formantCVVolts = 0.0f;
    alg->effFormantCenter = 0.45f;
    alg->effDensity = 0.35f;
    alg->effInputGain = lerpf(1.0f, 0.10f, alg->pInputAtten * alg->pInputAtten);
    alg->effDryWet = 0.55f;

    // Initialize DTC data
    _pareidolia_DTC* dtc = alg->dtc;

    // Initialize DC blockers
    dtc->dcBlockL.init(15.0f);
    dtc->dcBlockR.init(15.0f);

    dtc->activeGrainCount = 0;
    dtc->controlCounter = 0;

    // Zero capture buffer
    memset(alg->captureBuffer, 0, kCaptureBufferSize * 2 * sizeof(float));

    return alg;
}

static void parameterChanged(_NT_algorithm* self, int p) {
    // Parameters are smoothed in control rate, no immediate action needed
    (void)self;
    (void)p;
}

// ============================================================================
// Main Audio Processing (step)
// ============================================================================

static void step(_NT_algorithm* self, float* busFrames, int numFramesBy4) {
    _pareidoliaAlgorithm* alg = (_pareidoliaAlgorithm*)self;
    _pareidolia_DTC* dtc = alg->dtc;

    int numFrames = numFramesBy4 * 4;

    gSampleRate = (float)NT_globals.sampleRate;
    gInvSampleRate = 1.0f / gSampleRate;

    // Get bus pointers
    int inBusL  = alg->v[kParamInputL] - 1;
    int inBusR  = alg->v[kParamInputR] - 1;
    int outBusL = alg->v[kParamOutputL] - 1;
    int outBusR = alg->v[kParamOutputR] - 1;
    bool replace = alg->v[kParamOutputMode];
    bool freeze = (alg->v[kParamFreeze] != 0);

    const float* inL = busFrames + inBusL * numFrames;
    const float* inR = busFrames + inBusR * numFrames;
    float* outL = busFrames + outBusL * numFrames;
    float* outR = busFrames + outBusR * numFrames;

    // Read Formant CV (V/Oct) — average across block
    int cvBus = alg->v[kParamFormantCV] - 1;
    if (cvBus >= 0) {
        const float* cvIn = busFrames + cvBus * numFrames;
        float cvSum = 0.0f;
        for (int i = 0; i < numFrames; ++i) cvSum += cvIn[i];
        alg->formantCVVolts = cvSum / (float)numFrames;
    } else {
        alg->formantCVVolts = 0.0f;
    }

    // Clock CV edge detection (per-sample Schmitt trigger)
    int clockBus = alg->v[kParamClockCV] - 1;
    if (clockBus < 0) {
        alg->clockPeriodSamples = 0;
    } else {
        const float* clockIn = busFrames + clockBus * numFrames;
        alg->clockSampleCounter += numFrames;
        for (int i = 0; i < numFrames; ++i) {
            float s = clockIn[i];
            if (!alg->clockHigh && s > 2.5f) {
                alg->clockHigh = true;
                // Measure clock period (ignore very short intervals as debounce)
                if (alg->clockSampleCounter > 480) {
                    alg->clockPeriodSamples = alg->clockSampleCounter;
                }
                alg->clockSampleCounter = 0;
            } else if (alg->clockHigh && s < 0.5f) {
                alg->clockHigh = false;
            }
        }
        // Timeout: no edge for 2s → revert to free-running
        if (alg->clockSampleCounter > (int)(gSampleRate * 2.0f)) {
            alg->clockPeriodSamples = 0;
        }
    }

    // Analysis path (disabled while frozen)
    if (freeze) {
        // Freeze holds last analyzed/captured state and skips analysis CPU work.
        alg->yinInProgress = false;
        alg->yinDiffDone = false;
        alg->spectralInProgress = false;
        alg->analysisReady = false;
    } else if (alg->pPitchTrackEnable > 0.01f) {
        // YIN pitch tracking (expensive — gated by toggle)
        if (alg->analysisReady && !alg->yinInProgress) {
            alg->analysisReady = false;
            yinBegin(alg);  // snapshots the buffer
        }
        if (alg->yinInProgress && !alg->yinDiffDone) {
            yinStepIncremental(alg);
        }
        if (alg->yinDiffDone) {
            yinFinalize(alg);
            spectralFeaturesBegin(alg);
        }
    } else {
        // Cancel any in-progress YIN and clear stale pitch state
        alg->yinInProgress = false;
        alg->yinDiffDone = false;
        alg->pitchConfidence = 0.0f;

        // Spectral-only: snapshot buffer directly when ready
        if (alg->analysisReady && !alg->spectralInProgress) {
            alg->analysisReady = false;
            memcpy(alg->yinSnapshot, alg->analysisBuffer, sizeof(alg->analysisBuffer));
            spectralFeaturesBegin(alg);
        }
    }

    // Goertzel spectral features (cheap, always on)
    if (alg->spectralInProgress) {
        spectralFeaturesStep(alg);
    }

    // Accumulate into analysis and capture buffers (unless frozen).
    if (!freeze) {
        for (int i = 0; i < numFrames; ++i) {
            float wetInL = inL[i] * alg->effInputGain;
            float wetInR = inR[i] * alg->effInputGain;
            float mono = (wetInL + wetInR) * 0.5f;
            alg->analysisBuffer[alg->analysisWritePos] = mono;
            alg->analysisWritePos++;
            if (alg->analysisWritePos >= kAnalysisBufferSize) {
                alg->analysisWritePos = 0;
                alg->analysisReady = true;
            }

            // Capture buffer for Mode B
            alg->captureBuffer[alg->captureWritePos] = inL[i];
            alg->captureBuffer[kCaptureBufferSize + alg->captureWritePos] = inR[i];
            alg->captureWritePos = (alg->captureWritePos + 1) & (kCaptureBufferSize - 1);
        }
    }

    // Control-rate processing with exact intra-block tick offsets.
    // This avoids scheduling multiple control ticks as if they occurred at block start.
    int controlCounter = dtc->controlCounter;
    int sampleOffset = 0;
    int samplesRemaining = numFrames;
    while (samplesRemaining > 0) {
        int toNextTick = kControlRateDiv - controlCounter;
        if (toNextTick > samplesRemaining) {
            controlCounter += samplesRemaining;
            break;
        }
        sampleOffset += toNextTick;
        samplesRemaining -= toNextTick;
        controlCounter = 0;
        updateControlRate(alg, sampleOffset);
    }
    dtc->controlCounter = controlCounter;

    // ---- Grain synthesis (overlap-add) ----
    // Use full-size buffers so spectral distortion runs once, not per-chunk
    float grainBufL[128];
    float grainBufR[128];
    for (int i = 0; i < numFrames; ++i) {
        grainBufL[i] = 0.0f;
        grainBufR[i] = 0.0f;
    }

    int voiceCount[128];
    for (int i = 0; i < numFrames; ++i) {
        voiceCount[i] = 0;
    }

    for (int g = 0; g < kMaxGrains; ++g) {
        Grain& grain = alg->grains[g];
        if (!grain.active) continue;

        float* dst = (grain.channel == 0) ? grainBufL : grainBufR;
        int frame = 0;
        while (frame < numFrames && grain.active) {
            if (grain.startDelay > 0) {
                int skip = grain.startDelay;
                int left = numFrames - frame;
                if (skip > left) skip = left;
                grain.startDelay -= skip;
                frame += skip;
                continue;
            }

            if (grain.position >= grain.duration) {
                grain.active = false;
                break;
            }

            dst[frame] += processGrainSample(alg, grain);
            voiceCount[frame]++;
            grain.position++;
            frame++;
        }

        if (grain.position >= grain.duration) {
            grain.active = false;
        }
    }

    int activeCount = 0;
    for (int g = 0; g < kMaxGrains; ++g) {
        if (alg->grains[g].active) activeCount++;
    }

    // Per-sample smoothing for input gain and overlap compensation to avoid clicks.
    float smoothCoeff = 1.0f - exp(-1.0f / (kGainSmoothingTimeSec * gSampleRate));
    for (int i = 0; i < numFrames; ++i) {
        int overlapVoices = voiceCount[i] - 3;
        if (overlapVoices < 0) overlapVoices = 0;
        float targetOverlapComp = 1.0f / sqrt(1.0f + 0.20f * (float)overlapVoices);
        alg->smoothedOverlapComp += (targetOverlapComp - alg->smoothedOverlapComp) * smoothCoeff;
        alg->smoothedInputGain += (alg->effInputGain - alg->smoothedInputGain) * smoothCoeff;
        float gain = alg->smoothedOverlapComp * alg->smoothedInputGain;
        grainBufL[i] *= gain;
        grainBufR[i] *= gain;
    }

    // Leave headroom before spectral processing; avoid early saturation.

    // ---- DC Blocking ----
    for (int i = 0; i < numFrames; ++i) {
        grainBufL[i] = dtc->dcBlockL.process(grainBufL[i]);
        grainBufR[i] = dtc->dcBlockR.process(grainBufR[i]);
    }

    // ---- Output gain + soft output ceiling ----
    for (int i = 0; i < numFrames; ++i) {
        grainBufL[i] *= 2.0f;
        grainBufR[i] *= 2.0f;
        // Soft-limit near ±5V so transients are preserved and hard squashing is reduced.
        grainBufL[i] = softClipScaled(grainBufL[i], kOutputSoftCeilingVolts);
        grainBufR[i] = softClipScaled(grainBufR[i], kOutputSoftCeilingVolts);
    }

    // ---- Dry/Wet Mix (per-sample smoothed to avoid clicks) ----
    float targetDryWet = clampf(alg->effDryWet, 0.0f, 1.0f);
    bool outSameAsInL = (outBusL == inBusL);
    bool outSameAsInR = (outBusR == inBusR);
    // Reuse same smoothing coefficient for dry/wet.
    float dwSmoothCoeff = smoothCoeff;
    if (replace) {
        for (int i = 0; i < numFrames; ++i) {
            alg->smoothedDryWet += (targetDryWet - alg->smoothedDryWet) * dwSmoothCoeff;
            float wet = alg->smoothedDryWet;
            float dry = 1.0f - wet;
            outL[i] = dry * inL[i] + wet * grainBufL[i];
            outR[i] = dry * inR[i] + wet * grainBufR[i];
        }
    } else {
        for (int i = 0; i < numFrames; ++i) {
            alg->smoothedDryWet += (targetDryWet - alg->smoothedDryWet) * dwSmoothCoeff;
            float wet = alg->smoothedDryWet;
            float dry = 1.0f - wet;
            float mixedL = dry * inL[i] + wet * grainBufL[i];
            float mixedR = dry * inR[i] + wet * grainBufR[i];
            // In add mode, avoid doubling dry signal when input/output share a bus.
            outL[i] += outSameAsInL ? (wet * grainBufL[i]) : mixedL;
            outR[i] += outSameAsInR ? (wet * grainBufR[i]) : mixedR;
        }
    }

    dtc->activeGrainCount = activeCount;
}

// ============================================================================
// Draw
// ============================================================================

static bool draw(_NT_algorithm* self) {
    _pareidoliaAlgorithm* alg = (_pareidoliaAlgorithm*)self;

    // Draw a simple visualization
    // Show active grain count as a bar
    int activeCount = alg->dtc->activeGrainCount;

    // Title
    NT_drawText(128, 16, "PAREIDOLIA", 15, kNT_textCentre, kNT_textNormal);

    // Active grains indicator
    char buf[32];
    NT_intToString(buf, activeCount);
    NT_drawText(10, 30, "Grains:", 10, kNT_textLeft, kNT_textTiny);
    NT_drawText(50, 30, buf, 15, kNT_textLeft, kNT_textTiny);

    // Grain bar
    int barWidth = (activeCount * 200) / kMaxGrains;
    if (barWidth > 0) {
        NT_drawShapeI(kNT_rectangle, 10, 35, 10 + barWidth, 40, 12);
    }

    // Formant info
    char f1Buf[16], f2Buf[16];
    NT_intToString(f1Buf, (int)alg->f1Final);
    NT_intToString(f2Buf, (int)alg->f2Final);
    NT_drawText(10, 45, "F1:", 8, kNT_textLeft, kNT_textTiny);
    NT_drawText(30, 45, f1Buf, 12, kNT_textLeft, kNT_textTiny);
    NT_drawText(70, 45, "F2:", 8, kNT_textLeft, kNT_textTiny);
    NT_drawText(90, 45, f2Buf, 12, kNT_textLeft, kNT_textTiny);

    // Pitch info
    if (alg->pitchConfidence > 0.5f) {
        char pitchBuf[16];
        NT_intToString(pitchBuf, (int)alg->pitchEstimate);
        NT_drawText(140, 45, "P:", 8, kNT_textLeft, kNT_textTiny);
        NT_drawText(155, 45, pitchBuf, 12, kNT_textLeft, kNT_textTiny);
    }

    // Mode indicator
    const char* modeStr = "?";
    switch (alg->v[kParamGrainSource]) {
        case 0: modeStr = "NOISE"; break;
        case 1: modeStr = "INPUT"; break;
        case 2: modeStr = "RESO"; break;
    }
    NT_drawText(200, 30, modeStr, 15, kNT_textLeft, kNT_textTiny);
    if (alg->v[kParamFreeze]) {
        NT_drawText(200, 42, "FRZ", 14, kNT_textLeft, kNT_textTiny);
    }

    return false;
}

// ============================================================================
// Factory
// ============================================================================

static const _NT_factory factory = {
    .guid = NT_MULTICHAR('N', 's', 'P', 'a'),
    .name = "Pareidolia",
    .description = "Spectral voice hallucination - granular phantom choir effect",
    .numSpecifications = 0,
    .specifications = NULL,
    .calculateStaticRequirements = NULL,
    .initialise = NULL,
    .calculateRequirements = calculateRequirements,
    .construct = construct,
    .parameterChanged = parameterChanged,
    .step = step,
    .draw = draw,
    .midiRealtime = NULL,
    .midiMessage = NULL,
    .tags = kNT_tagEffect,
    .hasCustomUi = NULL,
    .customUi = NULL,
    .setupUi = NULL,
    .serialise = NULL,
    .deserialise = NULL,
    .midiSysEx = NULL,
    .parameterUiPrefix = NULL,
    .parameterString = NULL,
};

// ============================================================================
// Plugin Entry Point
// ============================================================================

extern "C"
uintptr_t pluginEntry(_NT_selector selector, uint32_t data) {
    switch (selector) {
        case kNT_selector_version:
            return kNT_apiVersionCurrent;
        case kNT_selector_numFactories:
            return 1;
        case kNT_selector_factoryInfo:
            return (uintptr_t)((data == 0) ? &factory : NULL);
    }
    return 0;
}
