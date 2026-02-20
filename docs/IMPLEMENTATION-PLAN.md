# Pareidolia - distingNT Plugin Implementation Plan

## Context

Pareidolia is a stereo granular effect processor that exploits auditory pareidolia - the brain's tendency to perceive voice-like patterns in ambiguous stimuli. It generates grain-based textures within formant-sensitive frequency ranges and applies spectral distortion to create "phantom choir" externalization effects.

The spec (`docs/PAREIDOLIA-v1.1.md`) defines 4 processing stages: Analysis, Granular Synthesis, Spectral Distortion, and Dry/Wet Mix. This plan implements it as a distingNT C++ plugin, testable via nt_emu in VCV Rack.

## Architecture Overview

**Single file**: `pareidolia.cpp` (~2000-2500 lines estimated)
**Build**: Makefile from template, targeting both `hardware` (ARM .o) and `test` (.dylib for nt_emu)
**GUID**: `NT_MULTICHAR('N', 's', 'P', 'a')` (Nosuch Pareidolia)

### Memory Layout

| Region | Contents | Size |
|--------|----------|------|
| SRAM | Algorithm struct, grain pool (24 structs), analysis buffer (2048), Hann LUT (4800), delay lines (5×2×64), vowel targets, parameter smoothers | ~45KB |
| DRAM | Input capture buffer (4096×2ch), overlap-add accumulation buffer | ~40KB |
| DTC | Hot-path state: filterbank SVFs (5×2ch), DC blocker, active grain count, control-rate counter | ~800B |

**Key insight**: Grains do NOT need pre-allocated content buffers. Each grain generates content sample-by-sample (noise→filter, resonator, or input-buffer read) during overlap-add. This eliminates ~900KB of grain content memory.

### Parameters (10 total)

| # | Name | Range | Unit | Notes |
|---|------|-------|------|-------|
| 0 | Input L | bus 1-64 | Audio Input | Default: 1 |
| 1 | Input R | bus 1-64 | Audio Input | Default: 2 |
| 2 | Output L | bus 1-64 | Audio Output + Mode | Default: 13 |
| 3 | Output R | bus 1-64 | Audio Output | Default: 14 |
| 4 | Output Mode | replace/add | | |
| 5 | Grain Source | A/B/C | Enum | Noise / Input-Seeded / Resonator |
| 6 | Grain Density | 0-100% | Percent | |
| 7 | Formant Center | 0-100% | Percent | F1→F3 sweep |
| 8 | Formant Drift | 0-100% | Percent | Vowel-space walk rate |
| 9 | Spectral Asymmetry | 0-100% | Percent | Externalization |
| 10 | Dry/Wet Mix | 0-100% | Percent | |
| 11 | Coherence | 0-100% | Percent | L/R grain correlation |
| 12 | Input Tracking | 0-100% | Percent | Analysis→grain modulation |

Pages: Main (6-12), Routing (0-5)

### Control Rate

Process control-rate updates every 64 samples (~1.3ms). Since step() typically receives 4-frame blocks on distingNT, accumulate a sample counter and run control-rate logic every 16 step() calls.

## Implementation Phases (Team Tasks)

### Phase 1: Scaffold + DSP Utilities
**Files**: `pareidolia.cpp`, `Makefile`

Create complete plugin structure with:
- All structs: `_pareidolia_DTC`, `_pareidoliaAlgorithm`, `Grain`, `SVF`, `OnePole`, `DelayLine`
- All parameters, pages, factory, entry point
- `calculateRequirements()` and `construct()` with full memory layout
- Stub `step()` that passes audio through with dry/wet mix
- DSP utility implementations:
  - **SVF filter**: `setFreq()`, `processBandpass()`, `processHighShelf()`, `processAllpass()` - used everywhere
  - **Hann window LUT**: Pre-computed at construction, 4800 entries (100ms max grain)
  - **One-pole parameter smoother**: 7 instances, configurable time constant
  - **DC blocker**: Single-pole highpass at 15Hz
  - **Fractional delay line**: For ITD processing (64-sample max per band per channel)
  - **PRNG**: Fast xorshift32 for noise generation and stochastic decisions
- Verify: builds for both targets, loads in nt_emu, passes audio through

### Phase 2: Grain Engine (Mode A - Noise Grains)
Build on Phase 1 scaffold:
- Grain pool: 24 pre-allocated slots with state (active, position, duration, envelope_idx, filter state, channel)
- Grain scheduling: free-running mode at 100Hz base × density, with ±30% jitter
- Mode A content: white noise (from PRNG) → SVF bandpass at formant center frequency
- Grain envelope: Hann window lookup with duration-based stride
- Overlap-add: accumulate active grains into output buffer
- Formant Center parameter: maps 0-1 to frequency via `lerp(300, 3500, param)`
- Grain Q: derived from density (5-15 range)
- Grain duration: `lerp(100ms, 10ms, formant_center_param)` in samples
- Stereo: L/R channels fire grains independently (Coherence param controls correlation)
- Verify: noise grains produce formant-shaped texture, density and formant center respond

### Phase 3: Formant Drift + Vowel-Space Walk
Build on Phase 2:
- 5 vowel attractor points: /i/, /a/, /e/, /o/, /u/ with F1/F2 coordinates
- Control-rate update: stochastic target switching (probability scaled by drift param)
- Smooth interpolation toward current target (one-pole)
- LFO modulation on top (0.5-3Hz, depth scaled by drift param) using sin/cos
- F3 derived: F2 + 1000Hz + small stochastic offset
- Feed f1_final/f2_final/f3_final into grain bandpass center frequency
- Verify: mid drift values produce organic voice-like movement

### Phase 4: YIN Pitch Detection + Analysis
Build on Phase 3:
- YIN algorithm operating on 2048-sample analysis buffer (mono-summed input)
- Run analysis every 2048 samples (~42ms) - accumulate input into analysis buffer
- Outputs: pitch_estimate (50-500Hz), pitch_confidence (0-1)
- Spectral centroid: weighted mean of 5-band filterbank energies (reuse spectral distortion filterbank)
- Spectral flux: frame-to-frame energy delta, normalized 0-1
- Input energy: RMS of analysis buffer
- Pitch-synchronous grain scheduling: when confidence >= 0.5, grain_period = 1/(pitch × density), jitter ±15%
- Input Tracking parameter: crossfade between analysis-derived values and defaults
  - Centroid → formant center bias
  - Flux → density scaling
  - Energy → wet mix gating
- Verify: grain timing locks to pitched input; free-runs on noise; tracking modulates effect

### Phase 5: Grain Modes B and C
Build on Phase 4:
- **Mode B (Input-Seeded)**: Circular input capture buffer (4096 samples × 2ch in DRAM). Grains read segments from capture buffer, pitch-shifted to formant target via sample-rate conversion (simple resampling ratio).
- **Mode C (Resonator)**: Impulse-excited parallel SVF bandpass pair (F1+F2) + optional F3.
  - Impulse train at pitch_estimate Hz (pitch-sync) or 100-150Hz (free-run)
  - F1 Q=10-12, F2 Q=12-15, F3 Q=15 (optional)
  - F1 at 0dB, F2 at -4dB, F3 at -9dB
  - Formant drift directly modulates resonator center frequencies
  - Filter state maintained across grain lifetime, reset on slot recycle
- Mode switching: active grains finish naturally, new grains use new mode
- Verify: Mode B produces uncanny familiar texture; Mode C produces vowel-like tones

### Phase 6: Spectral Distortion (Externalization)
Build on Phase 5:
- **5-band filterbank** (reduced from 8 for size budget - can expand later if within 64KB):
  - Band 1: 80-400Hz (F1 region)
  - Band 2: 400-1200Hz (F1 upper / F2 lower)
  - Band 3: 1200-2500Hz (F2 region - primary voice)
  - Band 4: 2500-5000Hz (F3 / presence / pinna cues)
  - Band 5: 5000-12000Hz (air / high-frequency spatial cues)
- Process grain output through filterbank (split → process → recombine)
- Per-band processing scaled by Spectral Asymmetry param:
  - **ILD**: Complementary L/R gain offsets, alternating sign per band
  - **ITD**: Frequency-dependent fractional micro-delays (max 32 samples), complementary L/R
  - **Phase offset** (asymmetry > 0.7): Random allpass phase per band between channels
  - **Pinna simulation**: Asymmetric high-shelf filtering (L: +2dB@4kHz, R: -1dB@5kHz)
- Band recombination: summation, verify energy preservation
- DC blocker on final output
- Verify: externalization clearly audible on headphones at asymmetry 0.5+

### Phase 7: Integration + Polish
- Parameter smoothing on all continuous parameters (one-pole, 10-20ms)
- Edge cases: silent input handling, grain source mode switching, extreme parameter combos
- Soft clipper on grain output stage (protection against filter instability)
- Startup initialization: all smoothers at target values, grain slots inactive, filters zeroed
- All 6 presets from spec verified by ear
- CPU profiling via `make size` and runtime testing in nt_emu
- Ensure within 64KB .text limit

## Design Decisions

- **Stereo I/O**: Separate L/R bus parameters (4 total) for maximum routing flexibility
- **All 7 parameters**: Implement Coherence and Input Tracking from the start
- **Size budget**: Start with 5-band filterbank (reduced from spec's 8). If we're well within 64KB after Phase 6, we can expand to 8 bands. The 5 bands still cover all critical formant and pinna-cue regions.

## Agent Team Structure

**3 agents, working sequentially** (single .cpp file requires sequential builds):

| Agent | Phases | Focus |
|-------|--------|-------|
| **foundation** | 1, 2, 3 | Scaffold, DSP utilities, grain engine, formant drift |
| **analysis** | 4, 5 | YIN pitch detection, input tracking, modes B/C |
| **externalization** | 6, 7 | Spectral distortion, integration, polish |

Each agent builds on the previous agent's work. After each phase, verify the plugin compiles and loads in nt_emu.

## Key Files

| File | Purpose |
|------|---------|
| `pareidolia.cpp` | Main plugin source (create new) |
| `Makefile` | Build config (create from template) |
| `distingNT_API/` | API submodule (needs to be added or symlinked) |
| `docs/PAREIDOLIA-v1.1.md` | Spec reference (exists) |

## Build & API Setup

The distingNT_API is available at `/Users/nealsanche/nosuch/nt_emu/external/distingNT_API/`. The Makefile `INCLUDES` should point there:
```
INCLUDES = -I. -I/Users/nealsanche/nosuch/nt_emu/external/distingNT_API/include
```

For nt_emu testing, build with `make test` and load the resulting `.dylib` in the nt_emu VCV Rack module.

## Verification

After each phase:
1. `make test` - compiles without errors/warnings
2. Load in nt_emu VCV Rack module - plugin appears and loads
3. Audio passes through (dry/wet at extremes)
4. Phase-specific perceptual checks from spec Section 6

Final verification:
- `make hardware` - ARM build succeeds
- `make size` - within 64KB .text limit
- All 6 presets produce expected perceptual character
- No clicks, pops, or DC offset
- Mono dry/wet bypass is bit-identical at mix=0
