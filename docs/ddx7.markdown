---
layout: default
title: >-
    Distilling DDSP: Exploring Real-Time Audio Generation on Embedded Systems
permalink: /ddx7/
---

<script type="text/x-mathjax-config"> MathJax.Hub.Config({ TeX: { equationNumbers: { autoNumber: "all" } } }); </script>
<script type="text/x-mathjax-config">
	MathJax.Hub.Config({
		tex2jax: {
			inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      processEscapes: true
  }
});
</script>
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# DDX7
Frequency Modulation (FM) synthesis generates complex timbres by modulating the frequency of a carrier oscillator using another modulator oscillator.

A signal $x[n]$ is expressed as:

$
x[n] = A_c \sin\left(2\pi f_c n T + I \sin\left(2\pi f_m n T\right)\right)
$

where $A_c$ and $f_c$ are respectively the amplitude and frequency of the carrier, while $f_m$ is the modulator frequency, and $I$ is the modulation index that determines the spectral complexity.
FM synthesis gained prominence in the 1980s with Yamaha's DX7 synthesizer. It became a standard for generating bright, dynamic timbres in electric piano, bass, brass, and bell sounds.

## DDSP Implementation
![DDSP Implementation Diagram](misc/images/ddx7_architecture.png)

The DDX7 architecture employs a TCN decoder conditioned on a sequence of pitch and loudness frames to drive the envelopes of a few-oscillator differentiable FM synthesizer that features a fixed FM configuration with fixed frequency ratios, effectively mapping continuous controls of pitched musical instruments to a well-known synthesis architecture.

## Audio Examples

<style>
  /* Hide the seek bar but retain the timeline */
  audio::-webkit-media-controls-timeline-container {
    display: none; /* Hides the seek bar container */
  }
  /* Hide the three dots menu (more options) */
  audio::-webkit-media-controls-panel {
    display: flex;
    justify-content: space-between;
  }
  audio::-webkit-media-controls-menu-button {
    display: none; /* Hides the three dots button */
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
        <source src="{{ site.baseurl }}/misc/audio/ref_anchor/flute_1_reference.wav" type="audio/mpeg">
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
         <source src="{{ site.baseurl }}/misc/audio/ref_anchor/trumpet_1_reference.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
    <td>
      <audio controls style="width: 150px; height: 30px;">
         <source src="{{ site.baseurl }}/misc/audio/ref_anchor/trumpet_1_anchor_p10_w800_srcimpulse.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
  </tr>
  <tr>
    <td>🎻 Violin</td>
    <td>
      <audio controls style="width: 150px; height: 30px;">
         <source src="{{ site.baseurl }}/misc/audio/ref_anchor/violin_1_reference.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
    <td>
      <audio controls style="width: 150px; height: 30px;">
         <source src="{{ site.baseurl }}/misc/audio/ref_anchor/violin_1_anchor_p10_w800_srcimpulse.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
  </tr>
  <tr>
    <td>🎹 Piano</td>
    <td>
      <audio controls style="width: 150px; height: 30px;">
         <source src="{{ site.baseurl }}/misc/audio/ref_anchor/piano_1_reference.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
    <td>
      <audio controls style="width: 150px; height: 30px;">
         <source src="{{ site.baseurl }}/misc/audio/ref_anchor/piano_1_anchor_p10_w800_srcimpulse.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
  </tr>
</table>

### DDX7

<table>
  <tr>
    <th></th>
    <th style="text-align: center;">Full</th>
    <th style="text-align: center;">Reduced</th>
    <th style="text-align: center;">Reduced+AD</th>
    <th style="text-align: center;">Reduced+CD</th>
  </tr>
  <tr>
    <td>🪈 Flute</td>
    <td>
      <audio controls>
        <source src="{{ site.baseurl }}/misc/audio/ddx7/flute_1_ddx7_teacher.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
    <td>
      <audio controls>
        <source src="{{ site.baseurl }}/misc/audio/ddx7/flute_1_ddx7_student.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
    <td>
      <audio controls>
        <source src="{{ site.baseurl }}/misc/audio/ddx7/flute_1_ddx7_student_KD_audio_ddx7.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
    <td>
      <audio controls>
        <source src="{{ site.baseurl }}/misc/audio/ddx7/flute_1_ddx7_student_KD_params_ddx7.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
  </tr>
  <tr>
    <td>🎺 Trumpet</td>
    <td>
      <audio controls>
        <source src="{{ site.baseurl }}/misc/audio/ddx7/trumpet_1_ddx7_teacher.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
    <td>
      <audio controls>
        <source src="{{ site.baseurl }}/misc/audio/ddx7/trumpet_1_ddx7_student.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
    <td>
      <audio controls>
        <source src="{{ site.baseurl }}/misc/audio/ddx7/trumpet_1_ddx7_student_KD_audio_ddx7.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
    <td>
      <audio controls>
        <source src="{{ site.baseurl }}/misc/audio/ddx7/trumpet_1_ddx7_student_KD_params_ddx7.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
  </tr>
  <tr>
    <td>🎻 Violin</td>
    <td>
      <audio controls>
        <source src="{{ site.baseurl }}/misc/audio/ddx7/violin_1_ddx7_teacher.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
    <td>
      <audio controls>
        <source src="{{ site.baseurl }}/misc/audio/ddx7/violin_1_ddx7_student.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
    <td>
      <audio controls>
        <source src="{{ site.baseurl }}/misc/audio/ddx7/violin_1_ddx7_student_KD_audio_ddx7.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
    <td>
      <audio controls>
        <source src="{{ site.baseurl }}/misc/audio/ddx7/violin_1_ddx7_student_KD_params_ddx7.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
  </tr>
  <tr>
    <td>🎹 Piano</td>
    <td>
      <audio controls>
        <source src="{{ site.baseurl }}/misc/audio/ddx7/piano_1_ddx7_teacher.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
    <td>
      <audio controls>
        <source src="{{ site.baseurl }}/misc/audio/ddx7/piano_1_ddx7_student.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
    <td>
      <audio controls>
        <source src="{{ site.baseurl }}/misc/audio/ddx7/piano_1_ddx7_student_KD_audio_ddx7.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
    <td>
      <audio controls>
        <source src="{{ site.baseurl }}/misc/audio/ddx7/piano_1_ddx7_student_KD_params_ddx7.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
  </tr>
</table>