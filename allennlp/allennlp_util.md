# Allennlp Util

`allennlp.nn.util` 中写好了非常多有用的工具，例如 masked_softmax、batched_index_select、viterbi 等等，使用原生 pytorch 编写代码也建议可以把这份代码复制到项目中，可以简化我们的开发，提升开发效率

### batched_index_select

```python
def batched_index_select(
    target: torch.Tensor,
    indices: torch.LongTensor,
    flattened_indices: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
```

>  The given `indices` of size `(batch_size, d_1, ..., d_n)` indexes into the sequence 
>
>  dimension (dimension 2) of the target, which has size `(batch_size, sequence_length,`
>
>  `embedding_size)`. Return a tensor with shape `[indices.size(), target.size(-1)]`

*简言之，它在 sequence_length 维度上，把你指定的 indices 所对应的那些位置的向量取出来。*

例如给定一个 (2, 10, 32) 的 target，给定 [[3, 4], [7, 8]] 的 indices，该函数就会将第一个 batch 的第 3、4 个 token 的表征和第二个 batch 的第 7、8 个 token 的表征取出来，其维度为 (2, 2, 32)

具体实现上，函数是将 target 和 indices 都**展平**后用 tensor.index_select 来取特征的。其中：

* 把 target 展平成 (batch_size * sequence_length, embedding_size)

* 把 indices 展平成 (batch_size * d1 * ... * dn)。依然用上面的例子，target -> (2 * 10, 32)，indices 展平后变成了 (3, 4, 17, 18) -> size 为 (batch_size = 2 * d1 = 2)。注意 17、18 这两个索引，展平之后，第二个 batch 的起始位置是 (2 - 1) * sequence_length = 10，所以要取第二个 batch 的第 7、8 个 token 实际上是取展平后的第 17、18 个 token。indices 的展平是通过 util 包中的 `flatten_and_batch_shift_indices` 函数来实现的。

  值得注意的是，`batched_index_select` 函数提供了一个可选的 `flattened_indices` 参数，若要多次调用同样的 `indices`，可以事先用 `flatten_and_batch_shift_indices` 处理好 `indices`，这样就避免了 indices 的多次展平。

### batched_span_select

```python
def batched_span_select(
	target: torch.Tensor, 
	spans: torch.LongTensor
) -> torch.Tensor:
```

>The given `spans` of size `(batch_size, num_spans, 2)` indexes into the sequence
>
> dimension (dimension 2) of the target, which has size `(batch_size, sequence_length,`
>
>`embedding_size)`. Return a span_tensor with shape `(batch_size, num_spans, `
>
>`max_batch_span_width, embedding_size)` and a span_mask with shape 
>
>`(batch_size, num_spans, max_batch_span_width)`, which are combined to a turple

与 `batched_index_select` 类似的，前者是指定 indices 把对应位置的向量取出来，`batched_span_select` 顾名思义，是将指定的 spans 所对应的向量取出来。

然后下面还有两个值得注意的点：

1. 输入的 spans 的最后一个维度是 2，用于表示一个 span 的区间 [start, end]，***此区间为闭区间***
2. 输出的特征张量第三个维度是 `max_batch_span_width`，表示整个 batch 中最长的 span 长度。小于这个长度的 span 会用所属句子的第一个 token 的特征向量补齐，然后在 span_mask 中补齐的位置为 False

所以其实事实上它还是返回了所有 spans 中每个 token 的向量(虽然被 max_batch_span_width 补齐了，但是依然可以从 mask 中找回来)，这么做的原因是后续可以很方便地自定义 span 表征，例如，取 span 的第一个 token 作为整个 span 的表征、取 span 的所有 token 的平均作为 span 表征或者其他更加复杂的操作。

该函数一个很好的使用例子是 bert：原句子的 token 被处理成 wordpiece 送入 bert，并用 offsets 标识原来的某个 token 对应的 wordpiece 是哪些，bert 输出 wordpiece 特征向量后，就可以把 wordpiece_embed 作为 target，把 offsets 作为 spans 使用 batched_span_select 得到 span_tensor，对 span_tensor 取 “first” 或 “avg” 即可得到 token 级别的特征向量。
