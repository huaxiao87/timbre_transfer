---
layout: page
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

<!-- ... -->

<link href="https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css" rel="stylesheet" integrity="sha384-wvfXpqpZZVQGK6TAh5PVlGOfQNHSoD2xbE+QkPxCAFlNEevoEH3Sl0sibVcOQVnN" crossorigin="anonymous" />
<link rel="stylesheet" href="{{ site.baseurl}}/css/trackswitch.min.css" />

# DDX7



## Audio Examples

<table>
  <tr>
    <th></th>
    <th style="text-align: center;">Reference</th>
    <th style="text-align: center;">Anchor (LPC)</th>
  </tr>
  <tr>
    <td>🪈 Flute</td>
    <td>
      <audio controls style="width: 150px; height: 30px;">
        <source src="{{ site.baseurl}}/examples/sounds/ism.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
    <td>
      <audio controls style="width: 150px; height: 30px;">
        <source src="{{ site.baseurl}}/examples/sounds/ism.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
  </tr>
  <tr>
    <td>🎺 Trumpet</td>
    <td>
      <audio controls style="width: 150px; height: 30px;">
        <source src="{{ site.baseurl}}/examples/sounds/ism.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
    <td>
      <audio controls style="width: 150px; height: 30px;">
        <source src="{{ site.baseurl}}/examples/sounds/ism.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
  </tr>
  <tr>
    <td>🎻 Violin</td>
    <td>
      <audio controls style="width: 150px; height: 30px;">
        <source src="{{ site.baseurl}}/examples/sounds/ism.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
    <td>
      <audio controls style="width: 150px; height: 30px;">
        <source src="{{ site.baseurl}}/examples/sounds/ism.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
  </tr>
  <tr>
    <td>🎹 Piano</td>
    <td>
      <audio controls style="width: 150px; height: 30px;">
        <source src="{{ site.baseurl}}/examples/sounds/ism.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
    <td>
      <audio controls style="width: 150px; height: 30px;">
        <source src="{{ site.baseurl}}/examples/sounds/ism.wav" type="audio/mpeg">
        Your browser does not support the audio tag.
      </audio>
    </td>
  </tr>
</table> 

### Harmonic-plus-Noise
<table>
  <tr>
    <th></th>
    <th style="text-align: center;">🎵 Flute</th>
    <th style="text-align: center;">🎺 Trumpet</th>
    <th style="text-align: center;">🎻 Violin</th>
    <th style="text-align: center;">🎹 Piano</th>
  </tr>
  <tr>
    <td style="text-align: center;">Full</td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/flute_full_hp.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/trumpet_full_hp.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/violin_full_hp.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/piano_full_hp.wav" type="audio/mpeg"></audio></td>
  </tr>
  <tr>
    <td style="text-align: center;">Reduced</td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/flute_reduced_hp.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/trumpet_reduced_hp.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/violin_reduced_hp.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/piano_reduced_hp.wav" type="audio/mpeg"></audio></td>
  </tr>
  <tr>
    <td style="text-align: center;">Reduced + AD</td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/flute_reduced_ad_hp.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/trumpet_reduced_ad_hp.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/violin_reduced_ad_hp.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/piano_reduced_ad_hp.wav" type="audio/mpeg"></audio></td>
  </tr>
  <tr>
    <td style="text-align: center;">Reduced + CD</td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/flute_reduced_cd_hp.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/trumpet_reduced_cd_hp.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/violin_reduced_cd_hp.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/piano_reduced_cd_hp.wav" type="audio/mpeg"></audio></td>
  </tr>
</table>

### DDX7
<table>
  <tr>
    <th></th>
    <th style="text-align: center;">🎵 Flute</th>
    <th style="text-align: center;">🎺 Trumpet</th>
    <th style="text-align: center;">🎻 Violin</th>
    <th style="text-align: center;">🎹 Piano</th>
  </tr>
  <tr>
    <td style="text-align: center;">Full</td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/flute_full_ddx7.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/trumpet_full_ddx7.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/violin_full_ddx7.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/piano_full_ddx7.wav" type="audio/mpeg"></audio></td>
  </tr>
  <tr>
    <td style="text-align: center;">Reduced</td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/flute_reduced_ddx7.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/trumpet_reduced_ddx7.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/violin_reduced_ddx7.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/piano_reduced_ddx7.wav" type="audio/mpeg"></audio></td>
  </tr>
  <tr>
    <td style="text-align: center;">Reduced + AD</td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/flute_reduced_ad_ddx7.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/trumpet_reduced_ad_ddx7.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/violin_reduced_ad_ddx7.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/piano_reduced_ad_ddx7.wav" type="audio/mpeg"></audio></td>
  </tr>
  <tr>
    <td style="text-align: center;">Reduced + CD</td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/flute_reduced_cd_ddx7.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/trumpet_reduced_cd_ddx7.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/violin_reduced_cd_ddx7.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/piano_reduced_cd_ddx7.wav" type="audio/mpeg"></audio></td>
  </tr>
</table>

### Wavetable
<table>
  <tr>
    <th></th>
    <th style="text-align: center;">🎵 Flute</th>
    <th style="text-align: center;">🎺 Trumpet</th>
    <th style="text-align: center;">🎻 Violin</th>
    <th style="text-align: center;">🎹 Piano</th>
  </tr>
  <tr>
    <td style="text-align: center;">Full</td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/flute_full_wt.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/trumpet_full_wt.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/violin_full_wt.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/piano_full_wt.wav" type="audio/mpeg"></audio></td>
  </tr>
  <tr>
    <td style="text-align: center;">Reduced</td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/flute_reduced_wt.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/trumpet_reduced_wt.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/violin_reduced_wt.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/piano_reduced_wt.wav" type="audio/mpeg"></audio></td>
  </tr>
  <tr>
    <td style="text-align: center;">Reduced + AD</td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/flute_reduced_ad_wt.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/trumpet_reduced_ad_wt.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/violin_reduced_ad_wt.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/piano_reduced_ad_wt.wav" type="audio/mpeg"></audio></td>
  </tr>
  <tr>
    <td style="text-align: center;">Reduced + CD</td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/flute_reduced_cd_wt.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/trumpet_reduced_cd_wt.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/violin_reduced_cd_wt.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/piano_reduced_cd_wt.wav" type="audio/mpeg"></audio></td>
  </tr>
  <tr>
    <td style="text-align: center;">Reduced w/ Prt + CD</td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/flute_reduced_prt_cd_wt.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/trumpet_reduced_prt_cd_wt.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/violin_reduced_prt_cd_wt.wav" type="audio/mpeg"></audio></td>
    <td style="text-align: center;"><audio controls style="width: 150px; height: 30px;"><source src="{{ site.baseurl}}/examples/sounds/piano_reduced_prt_cd_wt.wav" type="audio/mpeg"></audio></td>
  </tr>
</table>


