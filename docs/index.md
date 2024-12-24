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

#  Sound of DAFx Special Issue - 2025

Accompanying page with supplemental materials


## Abstract
This paper investigates the feasibility of running neural audio generative models on embedded systems, by comparing the performance of various models and evaluating their trade-offs in audio quality, inference speed, and memory usage. We focus on Differentiable Digital Signal Processing (DDSP) models, due to their hybrid architecture, which combines the efficiency and interoperability of traditional DSP with the flexibility of neural networks. In addition, we explore the application of Knowledge Distillation (KD) to improve the performance of smaller models. We implemented and evaluated two types of distillation strategies: audio distillation and control distillation. We applied these methods to three foundation DDSP generative models which integrate Harmonic-plus-Noise, FM, and Wavetable synthesis. Our results demonstrate the overall effectiveness of KD: we were able to train student models that are up to a hundred times smaller than their teacher counterparts, while maintaining comparable performance and significantly improving inference speed and memory efficiency. However, we have also observed cases where KD failed to improve or even degrade student performance. We provide a critical reflection on the advantages and limitations of KD, exploring its application in diverse use cases, and emphasizing the need for carefully tailored strategies to maximize its potential.


## Supplemental Material
- [Harmonic-plus-Noise]()