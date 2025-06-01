# PAHQ - Quantization Experiment
This repository is used to show the result of our quantization experiment.
  
This two-stage curve confirms that (1) a global 8-bit quantization (which would quantize critical and non-critical edges together) suffers a large accuracy loss, and (2) PAHQ’s targeted preservation of FP32 on critical edges is the only way to maintain high accuracy under an 8-bit budget.
