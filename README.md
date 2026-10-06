# LLM Post-Training & Alignment

学习并实践 LLM Post-Training，重点实现和理解 GRPO，并对比 DPO、SFT等方法。

## Methods

- **GRPO**：基于 group-relative rewards 进行强化学习优化
- **DPO**：基于 chosen/rejected preference data 进行偏好对齐
- **SFT**：基于 instruction-response 数据进行监督微调


## Results

### GRPO — OLMo-2-1B

| Model | GSM8K |
|---|---:|
| Before GRPO | 38.59 |
| After GRPO | **52.92** |

