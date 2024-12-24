---
layout: default
title: >-
    Distilling DDSP: Exploring Real-Time Audio Generation on Embedded Systems
permalink: /ddx7/
---

# DDX7

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

### DDX7
