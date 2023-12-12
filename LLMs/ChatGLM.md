## ChatGLM 微调

### Input-Output Format

* 数据集: AdvertiseGen

  ```json
  {
    "content": "类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞", 
    "summary": "简约而不简单的牛仔外套，白色的衣身十分百搭。衣身多处有做旧破洞设计，打破单调乏味，增加一丝造型看点。衣身后背处有趣味刺绣装饰，丰富层次感，彰显别样时尚。"
  }
  
  ```

* InputOutoutDataset `__getitem__()`:

  ```python
  def __getitem__(self, i) -> dict:
      data_item = self.data[i]
  		# 把提示(输入部分)编码
      a_ids = self.tokenizer.encode(text=data_item['prompt'], add_special_tokens=True, truncation=True,
                                       max_length=self.max_source_length)
  		# 把回答(输出部分)编码
      b_ids = self.tokenizer.encode(text=data_item['response'], add_special_tokens=False, truncation=True,
                                  max_length=self.max_target_length)
  
      context_length = len(a_ids)
      # 输入是 "输入 + 输出 + <eos>"
      input_ids = a_ids + b_ids + [self.tokenizer.eos_token_id]
    	# 标签是 "<PAD输入> + 输出 + <eos>"  
      labels = [self.tokenizer.pad_token_id] * context_length + b_ids + [self.tokenizer.eos_token_id]
  
      pad_len = self.max_seq_length - len(input_ids)
      input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
      # 把 labels id 为 PAD 的换成 -100
      labels = labels + [self.tokenizer.pad_token_id] * pad_len
      labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]
  
      assert len(input_ids) == len(labels), f"length mismatch: {len(input_ids)} vs {len(labels)}"
  		
      # 这里没有得到 attention_mask
      return {
          "input_ids": input_ids,
          "labels": labels
      }
  
  ```

* tokenization_chatglm `_pad()`: => 如果输入没有 attention_mask 会自动生成全 1 的 attention mask

  ```python
  def _pad(
          self,
          encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
          max_length: Optional[int] = None,
          padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
          pad_to_multiple_of: Optional[int] = None,
          return_attention_mask: Optional[bool] = None,
  ) -> dict:
      """
      Pad encoded inputs (on left/right and up to predefined length or max length in the batch)
  
      Args:
          encoded_inputs:
              Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
          max_length: maximum length of the returned list and optionally padding length (see below).
              Will truncate by taking into account the special tokens.
          padding_strategy: PaddingStrategy to use for padding.
  
              - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
              - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
              - PaddingStrategy.DO_NOT_PAD: Do not pad
              The tokenizer padding sides are defined in self.padding_side:
  
                  - 'left': pads on the left of the sequences
                  - 'right': pads on the right of the sequences
          pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
              This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
              `>= 7.5` (Volta).
          return_attention_mask:
              (optional) Set to False to avoid returning attention mask (default: set to model specifics)
      """
      # Load from model defaults
      assert self.padding_side == "left"
  
      required_input = encoded_inputs[self.model_input_names[0]]
      seq_length = len(required_input)
  
      if padding_strategy == PaddingStrategy.LONGEST:
          max_length = len(required_input)
  
      if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
          max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
  
      needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length
  		
    	# 这里如果 attention_mask 为空直接生成一个全 1 的 mask  
      # Initialize attention mask if not present.
      if "attention_mask" not in encoded_inputs:
          encoded_inputs["attention_mask"] = [1] * seq_length
  
      if "position_ids" not in encoded_inputs:
          encoded_inputs["position_ids"] = list(range(seq_length))
  
      if needs_to_be_padded:
          difference = max_length - len(required_input)
  				
          # PAD attention_mask 和 position_ids
          if "attention_mask" in encoded_inputs:
              encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
          if "position_ids" in encoded_inputs:
              encoded_inputs["position_ids"] = [0] * difference + encoded_inputs["position_ids"]
          encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input
  
      return encoded_inputs
  ```

  