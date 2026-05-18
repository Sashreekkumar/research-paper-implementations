# Encoder-Decoder Transformer

## Note
> The complete architecture and pipeline are fully implemented and functional. However, the current training loop is not yet optimized, with each epoch taking approximately 10 hours to complete. As a result, the model has not been fully trained, and final validation metrics are not yet available.

## Config 
|**Parameter**|**Value**|**Description**|
|---|---|---|
|`d_model`|`256`|Dimensionality of the structural hidden representations|
|`d_ff`|`2048`|Dimensionality of the internal feed-forward network layers|
|`heads (h)`|`8`|Number of parallel attention heads|
|`N (layers)`|`6`|Total number of stacked blocks in both encoder and decoder|
|`seq_len`|`128`|Maximum token sequence length window allowed per sequence|
|`batch_size`|`4`|Number of concurrent training sequences processed per step|
|`learning_rate`|`1e-4`|Learning rate assigned to the Adam optimization setup|
|`epochs`|`10`|Maximum complete dataset training runs scheduled|
|`loss_function`|`CrossEntropy`|Cross Entropy Loss adjusted with `label_smoothing=0.1`|

## My Improvements:
1. Used `Einsum`
2. Used `type casting`


## References and Credits:

This was possible because of:

1. [Umar Jamil - Encoder-Decoder Transformer](https://youtu.be/ISNdQcPhsts?si=IoRtT4mhSR4lpZW9)