---
layout: default
title: >-
    Distilling DDSP: Exploring Real-Time Audio Generation on Embedded Systems
permalink: /hpn/
---

<script src="{{ site.baseurl }}/js/trackswitch.min.js"></script>
<link rel="stylesheet" href="{{ site.baseurl }}/css/trackswitch.min.css">


# Harmonic-plus-Noise

## Audio Examples

#### Reference and Anchor examples

<table>
  <tr>
    <th></th>
    <th style="text-align: center;">Reference</th>
    <th style="text-align: center;">Anchor (LPC)</th>
  </tr>
  <tr>
    <td>🪈 Flute</td>
    <td>
      <div class="jquery-trackswitch" id="flute-reference">
        <ts-track src="{{ site.baseurl }}/examples/sounds/flute_reference.wav"></ts-track>
      </div>
    </td>
    <td>
      <div class="jquery-trackswitch" id="flute-anchor">
        <ts-track src="{{ site.baseurl }}/examples/sounds/flute_anchor.wav"></ts-track>
      </div>
    </td>
  </tr>
  <tr>
    <td>🎺 Trumpet</td>
    <td>
      <div class="jquery-trackswitch" id="trumpet-reference">
        <ts-track src="{{ site.baseurl }}/examples/sounds/trumpet_reference.wav"></ts-track>
      </div>
    </td>
    <td>
      <div class="jquery-trackswitch" id="trumpet-anchor">
        <ts-track src="{{ site.baseurl }}/examples/sounds/trumpet_anchor.wav"></ts-track>
      </div>
    </td>
  </tr>
  <tr>
    <td>🎻 Violin</td>
    <td>
      <div class="jquery-trackswitch" id="violin-reference">
        <ts-track src="{{ site.baseurl }}/examples/sounds/violin_reference.wav"></ts-track>
      </div>
    </td>
    <td>
      <div class="jquery-trackswitch" id="violin-anchor">
        <ts-track src="{{ site.baseurl }}/examples/sounds/violin_anchor.wav"></ts-track>
      </div>
    </td>
  </tr>
  <tr>
    <td>🎹 Piano</td>
    <td>
      <div class="jquery-trackswitch" id="piano-reference">
        <ts-track src="{{ site.baseurl }}/examples/sounds/piano_reference.wav"></ts-track>
      </div>
    </td>
    <td>
      <div class="jquery-trackswitch" id="piano-anchor">
        <ts-track src="{{ site.baseurl }}/examples/sounds/piano_anchor.wav"></ts-track>
      </div>
    </td>
  </tr>
</table>

### Harmonic-plus-Noise: Full vs Reduced

<table>
  <tr>
    <th></th>
    <th style="text-align: center;">Full</th>
    <th style="text-align: center;">Reduced</th>
  </tr>
  <tr>
    <td>🪈 Flute</td>
    <td>
      <div class="jquery-trackswitch" id="flute-full">
        <ts-track src="{{ site.baseurl }}/examples/sounds/flute_full.wav"></ts-track>
      </div>
    </td>
    <td>
      <div class="jquery-trackswitch" id="flute-reduced">
        <ts-track src="{{ site.baseurl }}/examples/sounds/flute_reduced.wav"></ts-track>
      </div>
    </td>
  </tr>
  <tr>
    <td>🎺 Trumpet</td>
    <td>
      <div class="jquery-trackswitch" id="trumpet-full">
        <ts-track src="{{ site.baseurl }}/examples/sounds/trumpet_full.wav"></ts-track>
      </div>
    </td>
    <td>
      <div class="jquery-trackswitch" id="trumpet-reduced">
        <ts-track src="{{ site.baseurl }}/examples/sounds/trumpet_reduced.wav"></ts-track>
      </div>
    </td>
  </tr>
  <tr>
    <td>🎻 Violin</td>
    <td>
      <div class="jquery-trackswitch" id="violin-full">
        <ts-track src="{{ site.baseurl }}/examples/sounds/violin_full.wav"></ts-track>
      </div>
    </td>
    <td>
      <div class="jquery-trackswitch" id="violin-reduced">
        <ts-track src="{{ site.baseurl }}/examples/sounds/violin_reduced.wav"></ts-track>
      </div>
    </td>
  </tr>
  <tr>
    <td>🎹 Piano</td>
    <td>
      <div class="jquery-trackswitch" id="piano-full">
        <ts-track src="{{ site.baseurl }}/examples/sounds/piano_full.wav"></ts-track>
      </div>
    </td>
    <td>
      <div class="jquery-trackswitch" id="piano-reduced">
        <ts-track src="{{ site.baseurl }}/examples/sounds/piano_reduced.wav"></ts-track>
      </div>
    </td>
  </tr>
</table>

<script>
  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.jquery-trackswitch').forEach(function (element) {
      $(element).Plugin();
    });
  });
</script>
