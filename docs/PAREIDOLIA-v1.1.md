**PAREIDOLIA**

Spectral Voice Hallucination Plugin

*A Granular Effect Processor for Voice-Detection Psychoacoustics*

Technical Implementation Specification

Version 1.0

February 2026

Target Platform: ARM-based DSP (48 kHz, stereo)

Language: C++ (no dynamic allocation in audio path)

1\. Overview

Pareidolia is a stereo granular effect processor that exploits the human auditory system’s voice-detection neural pathways. It generates grain-based textures within formant-sensitive frequency ranges and applies spectral distortion to create convincing “phantom choir” externalization effects.

The plugin operates on the principle of auditory pareidolia — the brain’s tendency to perceive voice-like patterns in ambiguous stimuli. Rather than synthesizing convincing voices, the system generates stimuli that exist at the threshold of voice detection, keeping the listener’s phoneme-recognition circuitry (superior temporal sulcus) in a state of continuous near-activation.

1.1 Design Philosophy

The perceptual effect is strongest when all parameters occupy transitional zones — not clearly voice, not clearly not-voice. The system is designed so that mid-range parameter settings produce the most psychoacoustically potent results, with extremes serving as creative tools rather than optimal operating points.

1.2 Resource Budget

All processing must execute within a single stereo audio callback at 48 kHz. The design targets the following resource envelope:

|  |  |  |
|----|----|----|
| **Resource** | **Budget** | **Notes** |
| Max simultaneous grains | 24 | Pre-allocated grain pool, no dynamic alloc |
| Grain buffer memory | ~2.3M samples | 24 grains × 48000 samples (1000ms) × 2 ch |
| Filterbank bands | 8 fixed bands | SVF or Butterworth, hard-tuned to formant regions |
| FFT | None in audio path | Pitch detection via YIN; spectral analysis via filterbank |
| Control rate | 64–256 samples | Parameter interpolation and grain scheduling |
| Sample rate | 48 kHz | Stereo in, stereo out |

2\. Core Signal Flow

The processing chain consists of four sequential stages, each with clearly defined inputs and outputs.

**\[STEREO INPUT\]** → **\[ANALYSIS\]** → **\[GRANULAR SYNTHESIS\]** → **\[SPECTRAL DISTORTION\]** → **\[STEREO OUTPUT\]**

2.1 Analysis Stage

Purpose: Extract pitch, spectral shape, and energy information from the input signal to drive the granular synthesis engine reactively.

|  |  |  |
|----|----|----|
| **Output** | **Method** | **Range / Rate** |
| pitch_estimate | YIN algorithm (preferred over autocorrelation for cost) | 50–500 Hz fundamental |
| pitch_confidence | YIN confidence metric | 0.0–1.0; below 0.5 triggers free-run mode |
| spectral_centroid | Weighted mean of filterbank band energies | Continuous Hz value |
| spectral_flux | Frame-to-frame energy delta across filterbank | Normalized 0.0–1.0 |
| input_energy | RMS of input buffer | Linear amplitude |

2.1.1 Pitch Detection Details

Use the YIN algorithm rather than FFT-based autocorrelation. YIN is significantly cheaper and provides a built-in confidence metric. Buffer size for YIN: 1024–2048 samples (21–42 ms at 48 kHz). This latency is acceptable for a texture effect. Run the analysis on a separate slower timer and interpolate results to the grain engine at control rate.

When pitch confidence drops below the threshold (0.5), the grain timing engine must switch from pitch-synchronous mode to free-running stochastic mode. This prevents the grain engine from locking onto noise artifacts and producing glitchy output.

2.1.2 Spectral Flux as Modulation Source

Spectral flux measures frame-to-frame change in spectral shape across the filterbank. This provides a “movement” signal from the input:

**Low spectral flux** (static/quiet input) → sparse, whispery grain density

**High spectral flux** (active/changing input) → dense, responsive choir texture

Spectral flux should be available as an internal modulation source for grain density, formant drift depth, and other parameters. This makes the effect feel alive and input-tracking rather than simply overlaid.

2.1.3 Analysis Pseudocode

> // Per analysis frame (every 1024–2048 samples):
>
> pitch_estimate, pitch_confidence = yin_detect(input_buffer, min=50Hz, max=500Hz)
>
> // Spectral centroid via filterbank (cheaper than FFT)
>
> band_energies\[N_BANDS\] = compute_band_energies(input_buffer, filterbank)
>
> spectral_centroid = weighted_mean(band_center_freqs, band_energies)
>
> // Spectral flux
>
> spectral_flux = sum(abs(band_energies\[i\] - prev_band_energies\[i\])) / N_BANDS
>
> spectral_flux = normalize(spectral_flux, 0.0, 1.0)
>
> prev_band_energies = band_energies
>
> // RMS energy
>
> input_energy = rms(input_buffer)

2.2 Granular Voice Synthesis Stage

The core effect engine. Generates overlapping grains whose spectral content targets human formant frequency regions, creating stimuli that trigger voice-detection circuitry without resolving into recognizable speech.

2.2.1 Grain Source Modes

Three grain content generation modes are provided. These are selectable via a discrete mode switch (not a continuous parameter).

|  |  |  |
|----|----|----|
| **Mode** | **Content** | **Perceptual Character** |
| A: Noise Grains | Bandpass-filtered white noise centered on formant target frequency with adjustable Q | Phantom choir texture; nothing resolves into recognizable voice. Cleanest and most controllable. |
| B: Input-Seeded Grains | Short segments captured from input buffer, pitch-shifted to formant target range | Unsettling; brain detects familiar spectral fingerprints from source but cannot resolve them. “Hearing your own voice whispered back.” |
| C: Resonator Grains | Impulse-excited resonant bandpass filter pair (modeling F1+F2), excited at detected pitch rate | Stripped-down vocal tract model. Most convincingly voice-like for minimal CPU. Produces vowel-like tonal quality. |

Mode B requires a circular input capture buffer (recommended: 2048–4096 samples). Pitch-shifting should use simple sample-rate conversion or granular repitching — phase vocoder is too expensive here.

2.2.2 Mode C: Impulse-Excited Resonator (Detailed)

Mode C is the most perceptually potent grain source because it directly models the vocal tract’s resonant structure. Rather than filtering noise, it uses a pair of resonant bandpass filters excited by a periodic impulse train, producing vowel-like tonal output with minimal computational cost.

**Architecture:**

Two SVF bandpass filters in parallel (modeling F1 and F2), summed and optionally followed by a third filter (F3) for brightness. Each filter is excited by the same impulse train source.

|  |  |  |
|----|----|----|
| **Parameter** | **Value** | **Notes** |
| Impulse train rate | pitch_estimate Hz (pitch-sync) or 100–150 Hz (free-run) | Drives the perceived pitch of the resonator output. In free-run mode, use a rate that falls in the natural speech fundamental range. |
| F1 resonator Q | 8–15 | Higher Q = more tonal, ringing quality. Lower Q = breathier. Q of 10–12 is the sweet spot for speech-like formants. |
| F2 resonator Q | 10–18 | F2 typically has slightly higher Q than F1 in natural speech. Keep F2 Q ≥ F1 Q. |
| F3 resonator Q (optional) | 12–20 | Adds brightness and presence. Can be omitted for a darker, more ambiguous quality. |
| F1/F2 gain balance | F1: 0 dB, F2: -3 to -6 dB | F1 is typically louder in natural vowels. Attenuate F2 relative to F1 for realism. |
| F3 gain (if present) | -6 to -12 dB relative to F1 | F3 provides color but should not dominate. |
| Impulse amplitude | Normalized to match noise grain RMS | Ensure level parity across grain source modes for seamless switching. |

**Formant Drift Interaction:**

In Mode C, the Formant Drift parameter directly modulates the center frequencies of the F1 and F2 resonator filters via the vowel-space attractor walk. This is where Mode C becomes most powerful: as the resonator frequencies traverse vowel space, the output sounds like shifting, indistinct speech — as if someone is talking just beyond the range of comprehension. The Q values should remain relatively stable during drift to avoid instability; only the center frequencies move.

**Impulse Train Generation:**

> // Per sample in Mode C grain:
>
> impulse_phase += impulse_rate / sample_rate
>
> if (impulse_phase \>= 1.0):
>
> impulse_phase -= 1.0
>
> impulse_sample = 1.0 // single-sample impulse
>
> else:
>
> impulse_sample = 0.0
>
> // Feed impulse through parallel resonators
>
> f1_out = svf_bandpass(impulse_sample, f1_center, f1_Q)
>
> f2_out = svf_bandpass(impulse_sample, f2_center, f2_Q)
>
> f3_out = svf_bandpass(impulse_sample, f3_center, f3_Q) // optional
>
> // Mix with gain balance
>
> grain_sample = f1_out \* 1.0 + f2_out \* f2_gain + f3_out \* f3_gain
>
> // Apply grain envelope (Hann window) as usual
>
> output = grain_sample \* grain_envelope\[position\]

Note: The resonator filters maintain state across the grain’s lifetime. Do NOT reset filter state at grain onset — allow natural ring-up. This produces smoother, more organic grain attacks. Reset filter state only when a grain slot is recycled from a fully-released grain.

2.2.3 Grain Scheduling

Grains are scheduled from a pre-allocated pool of 24 grain slots. When all slots are occupied, the oldest grain is force-released.

**Pitch-synchronous mode** (pitch_confidence ≥ 0.5): Grain onset period derived from detected fundamental. grain_period = 1.0 / (pitch_estimate × grain_density_param). Random jitter of ±15% applied to onset time for naturalness.

**Free-running mode** (pitch_confidence \< 0.5): Grain onset period derived from a fixed base rate (default: 100 Hz) scaled by grain_density_param. Larger jitter (±30%) for more organic texture.

**Grain duration:** 10–1000 ms. Shorter at higher formant center frequencies, longer at lower. Suggested mapping: duration_ms = lerp(1000, 10, formant_center_param).

**Grain envelope:** Hann window (raised cosine). Ensures smooth amplitude transitions and COLA-compliant overlap-add. No other envelope shapes are needed for this application.

**Grain Q factor:** 5–15. Narrower Q (≈15) produces more tonal, pitched grain content. Broader Q (≈5) produces noisier, breathier texture. Can be derived from grain_density: higher density → narrower Q (more tonal).

2.2.4 Formant Region Targets

These are the hard-coded formant frequency centers and ranges used throughout the system:

|  |  |  |  |
|----|----|----|----|
| **Formant** | **Center Frequency** | **Range** | **Perceptual Role** |
| F1 | 300 Hz | 200–600 Hz | Vowel identity, lower register, “chest” quality |
| F2 | 1500 Hz | 700–2300 Hz | Vowel discrimination, presence, intelligibility |
| F3 | 2500 Hz | 2500–3500 Hz | Brightness, presence peak, “air” |

2.2.5 Formant Drift: Vowel-Space Walk

Rather than a purely random stochastic walk, the formant drift system is constrained to trajectories between vowel attractor points in F1/F2 frequency space. This is psychoacoustically more potent than random wandering because the brain’s phoneme detectors keep almost-triggering as formants pass through recognizable vowel regions.

**Vowel Attractor Points:**

|           |             |             |                  |
|-----------|-------------|-------------|------------------|
| **Vowel** | **F1 (Hz)** | **F2 (Hz)** | **Example Word** |
| /i/       | 300         | 2300        | “see”            |
| /ɑ/       | 700         | 1200        | “father”         |
| /e/       | 400         | 2000        | “say”            |
| /ɔ/       | 600         | 1000        | “saw”            |
| /u/       | 300         | 900         | “who”            |

Implementation: At each control-rate update, the drift system selects a target vowel and interpolates F1/F2/F3 toward it using a smoothed random walk. Target switching probability increases with the Formant Drift parameter value.

> // Vowel-space attractor walk
>
> vowel_targets\[\] = { {300,2300}, {700,1200}, {400,2000}, {600,1000}, {300,900} }
>
> // At control rate:
>
> if (random() \< target_switch_probability \* formant_drift_param):
>
> current_target = random_choice(vowel_targets)
>
> // Smooth interpolation toward target
>
> f1_current += (current_target.f1 - f1_current) \* interpolation_rate
>
> f2_current += (current_target.f2 - f2_current) \* interpolation_rate
>
> // Add LFO modulation on top (0.5–3.0 Hz, depth scaled by drift param)
>
> f1_final = f1_current + sin(time \* lfo_rate) \* lfo_depth \* formant_drift_param
>
> f2_final = f2_current + cos(time \* lfo_rate \* 1.1) \* lfo_depth \* formant_drift_param
>
> // F3 derived: F3 ≈ F2 + 1000 Hz (approximate)
>
> f3_final = f2_final + 1000.0 + stochastic_offset

At low drift values, formants orbit a single vowel. At high drift, they transition rapidly between vowels, producing the unmistakable perception of “someone is speaking but I can’t make out words.”

2.2.6 Granular Synthesis Pseudocode

> // Per control-rate block:
>
> update_formant_drift(formant_drift_param, formant_center_param)
>
> // Grain scheduling
>
> if (pitch_confidence \>= 0.5):
>
> grain_period = 1.0 / (pitch_estimate \* grain_density_param)
>
> jitter = random(-0.15, 0.15) \* grain_period
>
> else:
>
> grain_period = 1.0 / (100.0 \* grain_density_param)
>
> jitter = random(-0.30, 0.30) \* grain_period
>
> if (time_since_last_grain \>= grain_period + jitter):
>
> slot = acquire_grain_slot() // oldest slot if pool exhausted
>
> slot.duration = lerp(1000ms, 10ms, formant_center_param)
>
> slot.envelope = precomputed_hann\[slot.duration\]
>
> switch (grain_source_mode):
>
> case NOISE:
>
> slot.content = bandpass(white_noise, center=f1_final, Q=grain_Q)
>
> case INPUT_SEEDED:
>
> slot.content = pitch_shift(capture_buffer.read_segment(slot.duration), f1_final)
>
> case RESONATOR:
>
> slot.content = resonate(impulse_train(pitch_estimate), f1_final, f2_final)
>
> slot.active = true
>
> // Per sample: overlap-add all active grains
>
> for each active_grain in grain_pool:
>
> output_sample += active_grain.content\[pos\] \* active_grain.envelope\[pos\]
>
> advance(active_grain)

2.3 Spectral Distortion Stage (Externalization)

This stage creates inter-channel spectral asymmetries that cause the brain to perceive the synthesized voice texture as originating outside the listener’s head. The effect exploits three psychoacoustic localization cues: Interaural Level Difference (ILD), Interaural Time Difference (ITD), and pinna-related spectral shaping.

Processing is performed via a fixed filterbank of 8 bands. No FFT/IFFT is used — the entire stage operates as parallel SVF (state-variable filter) or Butterworth bandpass filters with per-band gain, delay, and phase manipulation.

2.3.1 Filterbank Design

|  |  |  |  |
|----|----|----|----|
| **Band** | **Low (Hz)** | **High (Hz)** | **Role** |
| 1 | 80 | 200 | Sub-formant; minimal processing |
| 2 | 200 | 450 | F1 lower region |
| 3 | 450 | 800 | F1 upper region |
| 4 | 800 | 1500 | F2 lower region |
| 5 | 1500 | 2300 | F2 upper / F3 lower |
| 6 | 2300 | 3500 | F3 region (primary externalization target) |
| 7 | 3500 | 6000 | Presence / pinna cue region |
| 8 | 6000 | 12000 | Air / high-frequency spatial cues |

Bands 2–6 are the primary targets for voice-formant asymmetry. Bands 7–8 carry pinna-filtering cues for externalization.

2.3.2 Per-Band Asymmetry Processing

Three complementary methods are applied per band, all scaled by the Spectral Asymmetry parameter:

**Method 1: ILD (Interaural Level Difference)**

Apply complementary gain offsets between L and R channels within each band. The offset direction alternates across bands so that no single channel is consistently louder.

> asymmetry_depth = spectral_asymmetry_param \* intensity_per_band\[band_idx\]
>
> sign = (band_idx % 2 == 0) ? 1.0 : -1.0
>
> band_L \*= (1.0 + sign \* asymmetry_depth)
>
> band_R \*= (1.0 - sign \* asymmetry_depth)

**Method 2: ITD (Interaural Time Difference)**

Apply frequency-dependent micro-delays between channels. Different bands receive different delay values, creating spatial cues that don’t correspond to any real-world source position.

> max_itd_samples = 32 // ~0.67 ms at 48kHz (natural ITD max ~0.6 ms)
>
> delay_L = spectral_asymmetry_param \* max_delay_by_band\[band_idx\]
>
> delay_R = -delay_L \* 0.5 // complementary, not symmetric
>
> band_L = fractional_delay(band_L, delay_L)
>
> band_R = fractional_delay(band_R, delay_R)

**Method 3: Phase Offset (high asymmetry only)**

At Spectral Asymmetry values above 0.7, apply random phase offsets per band between channels. This destroys spatial coherence entirely, creating the “impossible space” effect.

> if (spectral_asymmetry_param \> 0.7):
>
> phase = random_phase() \* (spectral_asymmetry_param - 0.7) / 0.3
>
> band_L = allpass_filter(band_L, +phase)
>
> band_R = allpass_filter(band_R, -phase)

**Method 4: Pinna Simulation (high-shelf asymmetry)**

Apply subtle high-frequency spectral shaping to simulate pinna (outer ear) filtering. This is the key cue that tells the brain “this sound is outside my head.” No full HRTF convolution is needed — a simple per-channel high-shelf with different parameters is sufficient.

> // Static or slowly modulated pinna cues
>
> L_shelf: +2 dB above 4 kHz, Q = 0.7
>
> R_shelf: -1 dB above 5 kHz, +1 dB notch at 8 kHz
>
> // Modulate shelf parameters slowly with asymmetry param
>
> // These tiny asymmetries push externalization hard without mono-collapse artifacts

2.3.3 Band Recombination

After per-band processing, recombine via simple summation (overlap-add of filterbank outputs). Ensure the filterbank is energy-preserving: the sum of all band outputs should approximately equal the original signal at unity gain settings. Verify this during development by sweeping a sine tone through the filterbank at zero asymmetry and confirming flat magnitude response ±0.5 dB.

2.4 Dry/Wet Mix

Simple linear crossfade applied post-processing. No special considerations beyond parameter smoothing.

> final_L = (1.0 - dry_wet_mix) \* input_L + dry_wet_mix \* processed_L
>
> final_R = (1.0 - dry_wet_mix) \* input_R + dry_wet_mix \* processed_R

3\. Parameter Specification

All continuous parameters are 0.0–1.0 normalized. All must have one-pole low-pass smoothing applied at control rate to prevent zipper noise (recommended time constant: 10–20 ms).

3.1 GRAIN DENSITY

|  |  |
|----|----|
| **Property** | **Value** |
| Range | 0.0 – 1.0 |
| Maps to | Grain onset frequency (grains/second); controls grain overlap factor |
| Low (0.0–0.3) | Sparse, ethereal whispers. Brain struggles to cohere pattern into voice. |
| Mid (0.3–0.7) | Dense choir texture. Rich, ambiguous vocal quality. Optimal for hallucination. |
| High (0.7–1.0) | Thick, near-continuous texture. Approaches vocal sustain. Less ambiguous. |
| Perceptual effect | Changes perceived “number of voices” and solidity of hallucination |
| Implementation note | Also influences grain Q: higher density → narrower Q (more tonal) |

3.2 FORMANT CENTER

|  |  |
|----|----|
| **Property** | **Value** |
| Range | 0.0 – 1.0 (mapped to frequency space) |
| 0.0 | Emphasize F1 region (~300 Hz): deeper, masculine voice cues |
| 0.5 | Emphasize F2 region (~1–2 kHz): midrange, neutral/female voice cues |
| 1.0 | Emphasize F3 region (~2.5–3.5 kHz): bright, “present” vocal quality |
| Maps to | Center frequency of granular synthesis bandpass filter |
| Perceptual effect | Completely shifts perceived voice character without changing structure |
| Interaction | Asymmetry at formant regions maximizes externalization; sweep this while asymmetry is high for dramatic effect |

3.3 FORMANT DRIFT

|  |  |
|----|----|
| **Property** | **Value** |
| Range | 0.0 – 1.0 |
| 0.0 | Static formants: monotone, slightly uncanny quality |
| 0.3–0.5 | Subtle drift: natural-sounding vibrato and shimmer |
| 0.7–1.0 | Aggressive drift: formants slide wildly, alien quality |
| Modulation | LFO (0.5–3.0 Hz) + vowel-space stochastic walk (see Section 2.2.4) |
| Perceptual effect | Creates motion and liveliness; shifts between “living choir” and “otherworldly” |
| Implementation | Controls both LFO depth and vowel-target switching probability |

3.4 SPECTRAL ASYMMETRY

|  |  |
|----|----|
| **Property** | **Value** |
| Range | 0.0 – 1.0 |
| 0.0 | Coherent stereo: symmetric, localized internally (in-head) |
| 0.3–0.6 | Subtle mismatches: pushes externalization without being obvious |
| 0.7–1.0 | Extreme asymmetry: dramatic externalization, “unreal” impossible spatial quality |
| Methods | ILD, ITD, phase offset, pinna simulation (see Section 2.3.2) |
| Perceptual effect | Moves sound from “inside head” to “definitely external but spatially impossible” |
| Caution | Above 0.7, phase methods activate — verify mono compatibility |

3.5 DRY/WET MIX

|                |                                                   |
|----------------|---------------------------------------------------|
| **Property**   | **Value**                                         |
| Range          | 0.0 – 1.0                                         |
| 0.0            | Pure dry input (bypass)                           |
| 0.5            | Balanced blend                                    |
| 1.0            | Pure effect (full granular + spectral distortion) |
| Implementation | Linear crossfade post-processing                  |

3.6 COHERENCE (Optional 6th Parameter)

|  |  |
|----|----|
| **Property** | **Value** |
| Range | 0.0 – 1.0 |
| 1.0 | Both channels fire grains simultaneously (mono-ish, internal) |
| 0.5 | Partially correlated grain timing |
| 0.0 | Fully independent grain timing per channel — brain hears two separate “speakers” |
| Implementation | Controls correlation of L/R grain onset timing (cheap to implement) |
| Perceptual effect | Dramatically increases externalization even without spectral asymmetry |
| Note | Interacts strongly with Spectral Asymmetry. Both at low values = maximum externalization and spatial confusion. |

3.7 INPUT TRACKING (Optional 7th Parameter)

Controls how much the analysis stage influences grain parameters versus free-running autonomous generation. This parameter determines whether the “voices” seem to respond to the input signal or exist independently.

|  |  |
|----|----|
| **Property** | **Value** |
| Range | 0.0 – 1.0 |
| 0.0 | Free-running: autonomous texture generator, independent of input |
| 1.0 | Fully tracked: grain density, timing, and formant center respond to input analysis |
| Implementation | Crossfade between analysis-derived values and fixed defaults for grain scheduling params |

**Modulation Routing:**

Input Tracking crossfades three analysis outputs into the grain engine. At tracking = 0.0, all three use their fixed defaults. At tracking = 1.0, all three are fully driven by the analysis stage.

|  |  |  |  |
|----|----|----|----|
| **Analysis Output** | **Modulation Target** | **Fixed Default (tracking=0)** | **Tracked Behavior (tracking=1)** |
| spectral_centroid | Formant Center bias | No bias (Formant Center param only) | Spectral centroid offsets the Formant Center param. If input is bright, formant target shifts up; if dark, shifts down. Mapping: offset = (centroid - 1500) / 3000, clamped to ±0.3 |
| spectral_flux | Grain Density scaling | No scaling (Density param only) | Flux multiplies effective grain density. Quiet/static input → density ×0.3. Active/changing input → density ×1.5. The effect breathes with the input. |
| input_energy | Dry/Wet mix modulation | No modulation (Mix param only) | Low input energy fades effect toward dry, high energy fades toward wet. Prevents the effect from producing phantom voices during silence. Mapping: energy_gate = smoothed_rms \> gate_threshold ? 1.0 : rms/threshold |

**Input Tracking Pseudocode:**

> // At control rate, compute effective parameters:
>
> tracking = input_tracking_param
>
> // 1. Formant Center bias from spectral centroid
>
> centroid_offset = clamp((spectral_centroid - 1500.0) / 3000.0, -0.3, 0.3)
>
> effective_formant_center = formant_center_param + (centroid_offset \* tracking)
>
> effective_formant_center = clamp(effective_formant_center, 0.0, 1.0)
>
> // 2. Grain Density scaling from spectral flux
>
> flux_scale = lerp(1.0, lerp(0.3, 1.5, spectral_flux), tracking)
>
> effective_grain_density = grain_density_param \* flux_scale
>
> effective_grain_density = clamp(effective_grain_density, 0.01, 1.0)
>
> // 3. Energy gating of wet signal
>
> energy_gate = smoothed_rms \> gate_threshold ? 1.0 : smoothed_rms / gate_threshold
>
> effective_wet_mix = lerp(dry_wet_param, dry_wet_param \* energy_gate, tracking)
>
> // Use effective\_\* values in place of raw params downstream

The energy gating is particularly important: without it, the effect generates phantom voices from the noise floor during silence, which breaks the illusion. With tracking enabled, the voices emerge from and recede with the input signal.

4\. Parameter Interaction & Sweet Spots

4.1 Synergies

|  |  |
|----|----|
| **Combination** | **Interaction** |
| Grain Density + Formant Drift | High density + low drift = stable choir. Low density + high drift = disembodied whispers. |
| Formant Center + Spectral Asymmetry | Asymmetry at formant regions maximizes externalization. Sweep formant center while asymmetry is high for dramatic sweeping spatial effect. |
| Coherence + Spectral Asymmetry | Both low = maximum spatial confusion. Both high = focused mono-ish effect. These are the two primary externalization controls. |
| Input Tracking + Grain Density | High tracking + high density = responsive choir that follows input dynamics. Low tracking + low density = autonomous ambient whisper texture. |
| All parameters + Dry/Wet | Start at 100% wet to hear full effect, then blend to taste. |

4.2 Recommended Presets

|  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|
| **Preset Name** | **Density** | **Formant Ctr** | **Drift** | **Asymmetry** | **Wet** | **Character** |
| Natural Choir | 0.5 | sweep | 0.4 | 0.5 | 0.7 | Floating, angelic voices; naturalistic |
| Alien Whispers | 0.2 | 0.3 | 0.8 | 0.9 | 0.9 | Unsettling externalized whispers; sci-fi |
| Angel Pad | 0.7 | 0.5 (F2) | 0.3 | 0.6 | 0.5 | Warm, present floating voices; ambient |
| EVP Static | 0.15 | 0.8 (F3) | 0.6 | 0.4 | 1.0 | Voices in static; ghost-box effect |
| Throat Singer | 0.8 | 0.0 (F1) | 0.1 | 0.2 | 0.6 | Deep, resonant, throat-singing quality |
| Phantom Room | 0.4 | 0.5 | 0.5 | 0.8 | 0.8 | Voices seem to come from the room itself |

4.3 Preset Tuning Rationale

Each preset targets a specific perceptual regime. Understanding why each value was chosen helps with intuitive tweaking and designing new presets.

**Natural Choir (0.5 / sweep / 0.4 / 0.5 / 0.7):** Density at 0.5 hits the perceptual sweet spot where grains overlap enough to cohere into “voices” but retain enough gaps for ambiguity. Drift at 0.4 keeps formants moving through vowel space at a natural-feeling rate — roughly matching the pace of conversational speech formant transitions (2–4 per second). Asymmetry at 0.5 produces clear externalization without the phase artifacts that emerge above 0.7. Formant Center sweeps slowly to shift the perceived “voice type” over time.

**Alien Whispers (0.2 / 0.3 / 0.8 / 0.9 / 0.9):** Low density (0.2) means isolated, sparse grains — the brain catches fragments rather than a continuous texture, producing a whispery quality. High drift (0.8) moves formants rapidly through vowel space, triggering phoneme detectors without settling. Extreme asymmetry (0.9) activates phase offsets, placing the whispers in impossible spatial locations. Formant Center biased low (0.3, near F1) gives a darker, more threatening timbre.

**Angel Pad (0.7 / 0.5 / 0.3 / 0.6 / 0.5):** High density (0.7) produces near-continuous texture approaching sustained vowel sounds. Formant Center at F2 (0.5) targets the frequency region most associated with human speech presence and intelligibility. Low drift (0.3) keeps the texture stable and warm. Moderate asymmetry (0.6) provides gentle externalization. Wet at 0.5 blends equally with input for a “layered” quality.

**EVP Static (0.15 / 0.8 / 0.6 / 0.4 / 1.0):** Very low density (0.15) produces rare, isolated grain events — mimicking the sporadic “voice fragments” in electronic voice phenomena recordings. High Formant Center (0.8, near F3) targets the presence/brightness region where voice detection is most sensitive. Moderate drift and low asymmetry keep the effect subtle enough that the listener questions whether they heard a voice at all. Full wet removes the dry signal, simulating a “static only” source.

**Throat Singer (0.8 / 0.0 / 0.1 / 0.2 / 0.6):** High density and Formant Center at F1 (0.0) produces a deep, resonant, droning quality reminiscent of Tuvan throat singing. Near-zero drift keeps the formant fixed, producing a monotone effect. Low asymmetry maintains internal localization, enhancing the meditative, centered quality.

**Phantom Room (0.4 / 0.5 / 0.5 / 0.8 / 0.8):** Balanced density and drift at the midpoints of their ranges — the transitional zone where the pareidolia effect is strongest. High asymmetry (0.8) pushes the voices firmly external, creating the sensation they exist in the physical space. High wet mix emphasizes the effect over the dry signal.

5\. Implementation Requirements

5.1 Memory Management

Zero dynamic allocation in the audio callback. All buffers must be pre-allocated at initialization:

|  |  |  |
|----|----|----|
| **Buffer** | **Size** | **Purpose** |
| Grain pool | 24 × grain_struct | Pre-allocated grain slot array (content buffer, envelope pointer, position, state) |
| Grain content buffers | 24 × 48000 samples | Max 1000 ms grain at 48 kHz per slot |
| Input capture buffer | 4096 samples × 2 ch | Circular buffer for Mode B (input-seeded grains) |
| Analysis buffer | 2048 samples | YIN pitch detection input |
| Filterbank state | 8 bands × 2 ch × SVF state | Per-band filter state variables |
| Delay lines | 8 bands × 2 ch × 64 samples | Per-band fractional delay for ITD processing |
| Hann window LUT | Multiple sizes or 1 large + interpolation | Pre-computed grain envelopes |
| Overlap-add output | Block size + max grain duration | Accumulation buffer for grain output |

5.2 Control Rate Processing

Parameter updates and grain scheduling run at control rate, not sample rate. Recommended control block size: 64–256 samples (1.3–5.3 ms at 48 kHz). The following operations happen at control rate:

Parameter smoothing (one-pole LP, 10–20 ms time constant), formant drift update (vowel-space walk + LFO), grain onset scheduling and slot allocation, analysis result interpolation (pitch, energy, flux values), spectral asymmetry parameter distribution to filterbank bands.

5.3 Parameter Smoothing

All five (or seven) parameters must be smoothed to prevent discontinuities. Use a one-pole low-pass filter per parameter:

> // One-pole smoother
>
> smoothed += (target - smoothed) \* coefficient
>
> // where coefficient = 1.0 - exp(-1.0 / (time_constant_sec \* control_rate_hz))
>
> // Recommended time constants:
>
> // Grain Density: 10 ms (fast response for rhythmic changes)
>
> // Formant Center: 20 ms (avoid bandpass click on fast sweeps)
>
> // Formant Drift: 15 ms
>
> // Spectral Asymmetry: 15 ms
>
> // Dry/Wet Mix: 10 ms

5.4 Filter Implementation

Use state-variable filters (SVF) for all bandpass operations. SVF topology is preferred over Butterworth for this application because it allows simultaneous low/band/high outputs from a single computation, the center frequency and Q can be modulated at control rate without instability, and it has lower computational cost per band than equivalent Butterworth cascades.

Each filterbank band requires one SVF per channel (16 total for 8 bands stereo). The grain synthesis bandpass (for noise and resonator modes) requires an additional 1–2 SVFs. Total filter count: approximately 18–20 SVFs, which is well within budget.

5.5 Mono Compatibility

The spectral asymmetry stage is inherently mono-hostile. At high asymmetry values, L+R summation will produce comb filtering and cancellation in the formant bands. This is by design — the effect requires headphone or stereo speaker monitoring. However, the dry/wet mix and Spectral Asymmetry parameter at lower values should maintain reasonable mono fold-down. Document this as a user-facing note: “Pareidolia is designed for stereo monitoring. Mono summation at high Spectral Asymmetry values will produce artifacts.”

5.6 Edge Cases & Robustness

The following edge cases must be handled explicitly to ensure stability under all operating conditions.

**Parameter Changes Mid-Grain:**

When a continuous parameter (Formant Center, Grain Density, Drift) changes while grains are active, already-sounding grains must NOT be retroactively modified. Grains are “fire and forget” — once a grain is launched, its bandpass center frequency, duration, and envelope are fixed for its lifetime. Only newly scheduled grains pick up updated parameter values. This prevents discontinuities and filter instability. Exception: the Spectral Asymmetry and Dry/Wet parameters operate post-grain-summation and are applied continuously to the mixed output, so they update in real-time at control rate with smoothing.

**Grain Source Mode Switching:**

When switching between Modes A/B/C, do NOT kill active grains. Allow all currently sounding grains to complete their envelopes naturally. New grains after the switch use the new mode. This produces a smooth crossfade between modes over the span of one grain cycle (~10–1000 ms). Additionally, when switching TO Mode C (resonator), reset the SVF filter states for newly allocated grains to prevent state carry-over from previous grain content.

**Silent Input:**

When input_energy falls below the noise floor (suggested threshold: -60 dBFS), the analysis stage should hold its last valid pitch estimate and set pitch_confidence to 0.0, forcing the grain engine into free-run mode. If Input Tracking is enabled, the energy gate (Section 3.7) will fade the wet signal toward zero, preventing phantom voice generation from silence. If Input Tracking is disabled (or set to 0.0), the effect continues generating at full intensity regardless of input level — this is the “autonomous texture generator” mode.

**Extreme Parameter Combinations:**

Grain Density at maximum (1.0) with Formant Drift at maximum (1.0) produces rapid grain scheduling with wildly moving filter centers. This is CPU-intensive and potentially unstable. The grain pool cap at 24 slots provides natural protection — excess grain onsets simply recycle the oldest slot. However, verify that SVF filters remain stable when their center frequency is modulated at the maximum drift rate. If instability occurs (output exceeds ±1.0), apply a soft clipper to the grain output stage.

**DC Offset:**

The overlap-add grain accumulation and filterbank recombination stages can introduce DC offset, particularly with asymmetric grain envelopes or filter ringing. Apply a DC-blocking filter (single-pole highpass at 10–20 Hz) to the final stereo output before the dry/wet mix stage.

**Startup Initialization:**

On plugin initialization: all grain slots set to inactive, all filter states zeroed, analysis buffers filled with silence, all parameter smoothers initialized to their target values (not zero — this prevents an audible sweep from zero to the initial parameter value on first audio callback).

6\. Testing & Validation

6.1 Unit Tests

|  |  |  |
|----|----|----|
| **Test** | **Method** | **Pass Criteria** |
| Filterbank energy preservation | Sweep sine 20 Hz–20 kHz through filterbank at zero asymmetry, measure output level | Output ±0.5 dB of input across full range |
| Grain pool exhaustion | Set density to max, verify no crashes or memory issues | Oldest grains force-released; no allocation, no crash |
| Pitch detection accuracy | Feed known-pitch sine waves (100, 200, 440 Hz) | YIN estimate within ±2 Hz; confidence \> 0.9 |
| Pitch detection noise rejection | Feed white noise | Confidence \< 0.3; grain engine in free-run mode |
| Parameter smoothing | Step parameter from 0 to 1 instantly | No audible click or zipper noise |
| Dry/wet bypass | Set mix to 0.0 | Output is bit-identical to input |
| CPU budget | Profile full chain at max density, max asymmetry | Completes within audio callback deadline |

6.2 Perceptual Validation

These cannot be automated but should be verified during development by ear:

|  |  |  |
|----|----|----|
| **Test** | **Method** | **Expected Result** |
| Voice hallucination threshold | Play pink noise through effect at preset “Natural Choir” settings | Listener perceives phantom vocal-like textures |
| Externalization | A/B test: asymmetry 0.0 vs 0.6 on headphones | 0.6 perceived as notably more “outside the head” |
| Formant sweep character | Slowly sweep Formant Center 0→1 with density at 0.5 | Perceived voice character shifts from deep/male to bright/present |
| Vowel-space drift | Set Drift to 0.7, listen for 30+ seconds | Perception of “almost-speech” or “someone talking in another room” |
| Mode comparison | Switch A/B/C grain source modes with identical parameters | Mode A: choir. Mode B: uncanny/familiar. Mode C: most voice-like |

7\. Recommended Development Sequence

Build and test incrementally. Each phase should be independently audible and verifiable before proceeding.

Phase 1: Granular Engine

Implement the grain pool, Hann window LUT, grain scheduling (free-running mode only), and overlap-add output. Use Mode A (noise grains) with a fixed bandpass frequency. Verify: grains produce smooth, click-free overlapping noise bursts at variable density. Test parameter: Grain Density only.

Phase 2: Formant Control

Add the Formant Center parameter (sweepable bandpass). Add static formant region targeting. Verify: sweeping Formant Center produces audible tonal shift in grain content. Add parameters: Formant Center.

Phase 3: Formant Drift

Implement the vowel-space attractor walk and LFO modulation system. Verify: at mid drift values, output has organic, voice-like movement. At high values, formants slide dramatically. Add parameter: Formant Drift.

Phase 4: Pitch Detection & Sync

Implement YIN pitch detection with confidence output. Switch grain scheduling to pitch-synchronous mode when confidence is high. Add input analysis (spectral centroid, flux, energy). Verify: grain timing locks to input pitch when playing tonal material; falls back to free-run on noise.

Phase 5: Spectral Distortion

Implement the 8-band filterbank. Add per-band ILD and ITD processing. Add phase offset at high asymmetry values. Add pinna simulation shelving. Verify: externalization effect is clearly audible on headphones at asymmetry 0.5+. Add parameter: Spectral Asymmetry.

Phase 6: Integration & Polish

Add dry/wet mix, parameter smoothing on all controls, grain source mode switching (Modes A/B/C), optional Coherence and Input Tracking parameters. Verify all presets from Section 4.2. CPU profile and optimize. Create user documentation.

8\. References & Background

The following references provide background on the psychoacoustic and DSP concepts underlying this design:

**Auditory Pareidolia:** Voss, P. et al. (2020). Auditory hallucination-like experiences in the general population. The brain’s voice-detection circuitry in the superior temporal sulcus has an extremely low activation threshold for formant-structured stimuli.

**YIN Pitch Detection:** de Cheveigné, A. & Kawahara, H. (2002). YIN, a fundamental frequency estimator for speech and music. Journal of the Acoustical Society of America, 111(4), 1917–1930.

**Granular Synthesis:** Roads, C. (2001). Microsound. MIT Press. Comprehensive reference on grain-based audio synthesis techniques.

**Formant Frequencies:** Peterson, G. & Barney, H. (1952). Control methods used in a study of the vowels. Journal of the Acoustical Society of America, 24(2), 175–184. Canonical vowel formant measurements.

**Externalization & HRTF:** Hartmann, W.M. & Wittenberg, A. (1996). On the externalization of sound images. Journal of the Acoustical Society of America, 99(6), 3678–3688. Key reference on ILD/ITD/pinna cues for sound externalization.

**State-Variable Filters:** Chamberlin, H. (1985). Musical Applications of Microprocessors. Hayden Books. SVF topology reference for real-time audio.

**Overlap-Add Synthesis:** Smith, J.O. (2011). Spectral Audio Signal Processing. W3K Publishing. COLA conditions and grain window design.
