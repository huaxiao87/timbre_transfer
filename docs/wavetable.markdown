---
layout: default
title: >-
    Distilling DDSP: Exploring Real-Time Audio Generation on Embedded Systems
permalink: /wavetable/
---

# Wavetable

Wavetable synthesis creates sound by cycling through pre-defined waveforms stored in memory. These waveforms are stored in tables, called wavetables, which capture the spectral characteristics of a sound.

An interpolation function $\mathbb{f}(w;\phi[n])$ can be used to retrieve an arbitrary value of the stored waveform. For example, linear interpolation is defined as:

$
\mathbb{f}(w; \phi[n]) = w[i] + \alpha \cdot \left(w[i+1] - w[i]\right)
% \] 
$

Where the index $i$ is calculated from the phase $\phi[n]$ as the integer part, $w[i]$ and $w[i+1]$ are consecutive values stored in the wavetable, and $\alpha$ is the interpolation factor that blends between these waveform values, typically equal to the fractional part of the phase.
First introduced in the 1980s by PPG and later popularized by synthesizers like Waldorf's Wave, wavetable synthesis offers computational efficiency and rich timbral variety. 

By interpolating between multiple wavetables, this synthesis technique enables dynamic timbre transitions and richer generation. Based on this approach, a signal $x[n]$ can be expressed as a sum of multiple wavetable oscillators:

$
x[n] = \sum_{k=1}^M A_k[n]\mathbb{f}(w_k; \phi_k[n])
$

Where $A_k[n]$ is the amplitude of the $k$-th wavetable oscillator, and $M$ is the number of wavetables.
Wavetable synthesis is a staple in digital synthesizers thanks to its dynamic sound design capabilities, versatility, and ease of use.

## DDSP Implementation


## Audio Examples

<style>
  /* Hide the seek bar but retain the timeline */
  audio::-webkit-media-controls-timeline-container {
    display: none; /* Hides the seek bar container */
  }
  audio::-webkit-media-controls-timeline {
    display: none; /* Hides the seek bar */
  }
  /* Adjust the audio player to keep the timer visible */
  audio {
    width: 150px; /* Adjust the player width */
    height: 30px; /* Adjust the player height */
  }
</style>

<table>
  <tr>
    <th></th>
    <th style="text-align: center;">Reference</th>
    <th style="text-align: center;">Anchor (LPC)</th>
  </tr>
  <tr>
    <td>🪈 Flute</td>
    <td>
      <audio controls>
        <source src="{{ site.baseurl }}/misc/audio/ref_anchor/flute_1_anchor_p10_w800_srcimpulse.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
    <td>
      <audio controls style="width: 150px; height: 30px;">
         <source src="{{ site.baseurl }}/misc/audio/ref_anchor/flute_1_anchor_p10_w800_srcimpulse.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
  </tr>
  <tr>
    <td>🎺 Trumpet</td>
    <td>
      <audio controls style="width: 150px; height: 30px;">
         <source src="{{ site.baseurl }}/misc/audio/ref_anchor/flute_1_anchor_p10_w800_srcimpulse.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
    <td>
      <audio controls style="width: 150px; height: 30px;">
         <source src="{{ site.baseurl }}/misc/audio/ref_anchor/flute_1_anchor_p10_w800_srcimpulse.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
  </tr>
  <tr>
    <td>🎻 Violin</td>
    <td>
      <audio controls style="width: 150px; height: 30px;">
         <source src="{{ site.baseurl }}/misc/audio/ref_anchor/flute_1_anchor_p10_w800_srcimpulse.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
    <td>
      <audio controls style="width: 150px; height: 30px;">
         <source src="{{ site.baseurl }}/misc/audio/ref_anchor/flute_1_anchor_p10_w800_srcimpulse.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
  </tr>
  <tr>
    <td>🎹 Piano</td>
    <td>
      <audio controls style="width: 150px; height: 30px;">
         <source src="{{ site.baseurl }}/misc/audio/ref_anchor/flute_1_anchor_p10_w800_srcimpulse.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
    <td>
      <audio controls style="width: 150px; height: 30px;">
         <source src="{{ site.baseurl }}/misc/audio/ref_anchor/flute_1_anchor_p10_w800_srcimpulse.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
  </tr>
</table>

### Wavetable
