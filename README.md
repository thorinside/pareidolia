# Pareidolia

Pareidolia is a stereo texture effect for *disting NT* that turns incoming audio into ghostly, voice-like layers.

It is made for musical use first:
- expressive movement
- playable rhythm interaction
- controllable chaos without needing deep DSP knowledge

## What You Will Hear

Depending on settings, Pareidolia can sound like:
- whispered choirs behind your source
- shimmering resonant vowels
- broken radio voices
- soft, haunted ambience
- rhythmic grain clouds locked to clock

It is not a clean delay or standard reverb. It is an illusion generator.

## Quick Start (Musician Version)

1. Feed a mono or stereo source into Pareidolia.
2. Start with these settings:
- `Grain Source`: `Resonator`
- `Grain Density`: `25-40%`
- `Grain Length`: `50-70%`
- `Formant Center`: `40-60%`
- `Formant Drift`: `10-25%`
- `Input Gain`: `0 dB`
- `Dry/Wet Mix`: `35-55%`
- `Coherence`: `50-70%`
- `Input Tracking`: `20-40%`
- `Pitch Track`: `Off`
- `Freeze`: `Off`
3. Bring up `Dry/Wet` slowly while listening on headphones or stereo monitors.
4. Sweep `Formant Center` for vowel color.
5. Add `Formant Drift` for motion.

## Controls (What They Mean Musically)

- `Grain Source`: choose your texture engine. `Noise` is airy, `Input` is uncanny and source-related, `Resonator` is most voice-like.
- `Grain Density`: low is sparse, high is thick. In clock mode this also feels like rhythmic rate/division.
- `Grain Length`: short is chattery/percussive, long is smoother and more sustained.
- `Formant Center`: low is darker/chest-like, high is brighter/whispery.
- `Formant Drift`: low is stable, high is animated and talking-like.
- `Input Gain` (`-70 dB` to `+24 dB`, default `0 dB`): your main gain staging control.
- `Dry/Wet Mix`: `0%` dry, `100%` effect.
- `Coherence`: stereo link amount. Low is wider/more independent L-R, high is tighter/more centered.
- `Input Tracking`: how much the effect follows your source behavior.
- `Pitch Track`: timing follows detected pitch. Best on monophonic material.
- `Freeze`: captures and holds the current texture for live sculpting.

## Clock Use

With a clock patched:
- texture rhythm follows clock timing
- `Grain Density` changes rhythmic rate feel (including slower-than-clock options)
- `Grain Length` controls how long each grain rings out, independent of rhythm

This gives a rhythmic grain instrument feel, not a standard echo.

## Three Fast Performance Recipes

### 1) Haunted Pad
- `Resonator`
- `Density 20-30%`
- `Length 70-90%`
- `Drift 10-20%`
- `Mix 35-50%`

### 2) EVP Radio Voices
- `Input`
- `Density 10-20%`
- `Length 50-75%`
- `Center 60-85%`
- `Drift 40-70%`
- `Mix 60-100%`

### 3) Clocked Pulse Choir
- Patch clock in
- `Resonator` or `Noise`
- `Density` to taste for division/rate
- `Length 40-80%`
- `Coherence 40-65%`
- `Mix 35-60%`

## Troubleshooting (Musician-Oriented)

- Too distorted: lower `Input Gain` first, then lower `Dry/Wet` or `Density` if needed.
- Too quiet: raise `Input Gain` toward `0 dB` or above, then raise `Dry/Wet`.
- Too harsh or buzzy: lower `Formant Center`, lower `Density`, or try `Resonator`.
- Too static: raise `Formant Drift`, raise `Input Tracking`, or lower `Coherence`.
- Guitar feels unstable with `Pitch Track`: turn `Pitch Track` off and keep `Input Tracking` moderate.

## Tips

- Pareidolia rewards movement. Slow knob gestures are very musical.
- Stereo monitoring gives the best illusion.
- Use `Freeze` as a live performance move: grab a moment, then reshape it.
- If gain staging feels off, treat `Input Gain` as your first control.

## Hardware Controls

Pareidolia maps the most important parameters to physical controls for hands-on performance without menu diving.

| Control | Parameter | Behavior |
|---------|-----------|----------|
| Pot L | Grain Density | 0-100%, soft takeover |
| Pot C | Grain Length | 0-100%, soft takeover |
| Pot R | Formant Center | 0-100%, soft takeover |
| Encoder L turn | Dry/Wet Mix | ±2% per click |
| Encoder L press | Freeze | Toggle on/off |
| Encoder R turn | Grain Source | Steps through Noise / Input / Resonator (clamped at ends) |
| Buttons | — | Not used (default behavior retained) |

**Soft takeover**: When switching presets or after power-on, pot positions may not match parameter values. Move a pot until it "catches" the current value (within ±2% or at an endpoint), then it tracks normally. This prevents parameter jumps.
