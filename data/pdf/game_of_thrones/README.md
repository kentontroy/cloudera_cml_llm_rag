## Quantization
Quantization is the process of converting model weights from higher precision to lower precision floating point values.

## GGML and GGUF
GGML stands for Georgi Gerganov’s Machine Learning. 
The GGML format provided a way to encapsulate an entire model into a single file. Given the use of quantization, model inferencing
could be done on CPUs with or without the use of GPUs. When GPUs are used, one could limit the amount of VRAM used dependening
upon model size.

Facebook released the Georgi Gerganov’s Unified Format (GGUF) in August of 2023 to overcome some of the limitations of the GGML format.
For example,new features can be added to the format without breaking previous, deployed models.Tokenization can now better support
special characters. Reportedly,there is also enhanced performance.

New versions of tokenizers > 0.13.3 should be used to support the GGUF format.


## Cloudera Machine Learning Documentation
For detailed instructions on how to run these scripts, see the [documentation](https://docs.cloudera.com/machine-learning/cloud/models/topics/ml-creating-and-deploying-a-model.html).
