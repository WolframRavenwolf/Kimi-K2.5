<div align="center">
  <picture>
      <img src="figures/kimi-logo.png" width="30%" alt="Kimi K2.5">
  </picture>
</div>
<hr>
<div align="center" style="line-height:1">
  <a href="https://www.kimi.com" target="_blank"><img alt="Chat" src="https://img.shields.io/badge/🤖%20Chat-Kimi%20K2.5-ff6b6b?color=1783ff&logoColor=white"/></a>
  <a href="https://www.moonshot.ai" target="_blank"><img alt="Homepage" src="https://img.shields.io/badge/Homepage-Moonshot%20AI-white?logo=Kimi&logoColor=white"/></a>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/moonshotai" target="_blank"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Moonshot%20AI-ffc107?color=ffc107&logoColor=white"/></a>
  <a href="https://twitter.com/kimi_moonshot" target="_blank"><img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-Kimi.ai-white?logo=x&logoColor=white"/></a>
    <a href="https://discord.gg/TYU2fdJykW" target="_blank"><img alt="Discord" src="https://img.shields.io/badge/Discord-Kimi.ai-white?logo=discord&logoColor=white"/></a>
</div>
<div align="center" style="line-height: 1;">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Modified_MIT-f5de53?&color=f5de53"/></a>
</div>
<p align="center">
<b>📰&nbsp;&nbsp;<a href="https://www.kimi.com/blog/kimi-k2-5.html">Tech Blog</a></b> | &nbsp;&nbsp;&nbsp; <b>📄&nbsp;&nbsp;<a href="tech_report.pdf">Full Report</a></b>
</p>

## 3. Evaluation Results



<div align="center">
<table>
<thead>
<tr>
<th align="center">Benchmark</th>
<th align="center"><sup>Kimi K2.5<br><sup>(Thinking)</sup></sup></th>
<th align="center"><sup>GPT-5.2 <br><sup>(xhigh)</sup></sup></th>
<th align="center"><sup>Claude 4.5 Opus <br><sup>(Extended Thinking)</sup></sup></th>
<th align="center"><sup>Gemini 3 Pro <br><sup>(High Thinking Level)</sup></sup></th>
<th align="center"><sup>DeepSeek V3.2 <br><sup>(Thinking)</sup></sup></th>
<th align="center"><sup>Qwen3-VL-<br>235B-A22B-<br>Thinking</sup></th>
</tr>
</thead>
<tbody>
<tr>
<td align="center" colspan=8><strong>Coding</strong></td>
</tr>
<tr>
<td align="center" style="vertical-align: middle">Terminal Bench 2.0</td>
<td align="center" style="vertical-align: middle">50.8</td>
<td align="center" style="vertical-align: middle">54.0</td>
<td align="center" style="vertical-align: middle">59.3</td>
<td align="center" style="vertical-align: middle">54.2</td>
<td align="center" style="vertical-align: middle">46.4</td>
<td align="center" style="vertical-align: middle">-</td>
</tr>
</tbody>
</table>
</div>

<details>
<summary><b>Footnotes</b></summary>

1. General Testing Details
   - We report results for Kimi K2.5 and DeepSeek-V3.2 with thinking mode enabled, Claude Opus 4.5 with extended thinking mode, GPT-5.2 with xhigh reasoning effort, and Gemini 3 Pro with a high thinking level. For vision benchmarks, we additionally report results for Qwen3-VL-235B-A22B-Thinking.
   - Unless otherwise specified, all Kimi K2.5 experiments were conducted with temperature = 1.0, top-p = 0.95, and a context length of 256k tokens.
   - Benchmarks without publicly available scores were re-evaluated under the same conditions used for Kimi K2.5 and are marked with an asterisk (*).
   - We could not evaluate GPT-5.2 xhigh on all benchmarks due to service stability issues. For benchmarks that were not tested, we mark them as "-".
5. Coding Tasks
   - Terminal-Bench 2.0 scores were obtained with the default agent framework (Terminus-2) and the provided JSON parser. In our implementation, we evaluated Terminal-Bench 2.0 under non-thinking mode. This choice was made because our current context management strategy for the thinking mode is incompatible with Terminus-2.
   - All reported scores of coding tasks are averaged over 5 independent runs.

</details>

## 4. Native INT4 Quantization
Kimi-K2.5 adopts the same native int4 quantization method as [Kimi-K2-Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking#4-native-int4-quantization).

## 5. Deployment
> [!Note]
> You can access Kimi-K2.5's API on https://platform.moonshot.ai and we provide OpenAI/Anthropic-compatible API for you. To verify the deployment is correct, we also provide the  [Kimi Vendor Verifier](https://kimi.com/blog/kimi-vendor-verifier.html).
Currently, Kimi-K2.5 is recommended to run on the following inference engines:
* vLLM
* SGLang
* KTransformers

The minimum version requirement for `transformers` is `4.57.1`.

Deployment examples can be found in the [Model Deployment Guide](docs/deploy_guidance.md).


---
## 6. Model Usage

The usage demos below demonstrate how to call our official API.

For third-party APIs deployed with vLLM or SGLang, please note that:
> [!Note]
> - Chat with video content is an experimental feature and is only supported in our official API for now.
>
> - The recommended `temperature` will be `1.0` for Thinking mode and `0.6` for Instant mode.
>
> - The recommended `top_p` is `0.95`.
>
> - To use instant mode, you need to pass `{'chat_template_kwargs': {"thinking": False}}` in `extra_body`.

### Chat Completion

This is a simple chat completion script which shows how to call K2.5 API in Thinking and Instant modes.

```python
import openai
import base64
import requests
def simple_chat(client: openai.OpenAI, model_name: str):
    messages = [
        {'role': 'system', 'content': 'You are Kimi, an AI assistant created by Moonshot AI.'},
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'which one is bigger, 9.11 or 9.9? think carefully.'}
            ],
        },
    ]
    response = client.chat.completions.create(
        model=model_name, messages=messages, stream=False, max_tokens=4096
    )
    print('====== Below is reasoning_content in Thinking Mode ======')
    print(f'reasoning content: {response.choices[0].message.reasoning_content}')
    print('====== Below is response in Thinking Mode ======')
    print(f'response: {response.choices[0].message.content}')

    # To use instant mode, pass {"thinking" = {"type":"disabled"}}
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=False,
        max_tokens=4096,
        extra_body={'thinking': {'type': 'disabled'}},  # this is for official API
        # extra_body= {'chat_template_kwargs': {"thinking": False}}  # this is for vLLM/SGLang
    )
    print('====== Below is response in Instant Mode ======')
    print(f'response: {response.choices[0].message.content}')
```


### Chat Completion with visual content

K2.5 supports Image and Video input.

The following example demonstrates how to call K2.5 API with image input:

```python
import openai
import base64
import requests

def chat_with_image(client: openai.OpenAI, model_name: str):
    url = 'https://huggingface.co/moonshotai/Kimi-K2.5/resolve/main/figures/kimi-logo.png'
    image_base64 = base64.b64encode(requests.get(url).content).decode()
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'Describe this image in detail.'},
                {
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/png;base64, {image_base64}'},
                },
            ],
        }
    ]

    response = client.chat.completions.create(
        model=model_name, messages=messages, stream=False, max_tokens=8192
    )
    print('====== Below is reasoning_content in Thinking Mode ======')
    print(f'reasoning content: {response.choices[0].message.reasoning_content}')
    print('====== Below is response in Thinking Mode ======')
    print(f'response: {response.choices[0].message.content}')

    # Also support instant mode if you pass {"thinking" = {"type":"disabled"}}
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=False,
        max_tokens=4096,
        extra_body={'thinking': {'type': 'disabled'}},  # this is for official API
        # extra_body= {'chat_template_kwargs': {"thinking": False}}  # this is for vLLM/SGLang
    )
    print('====== Below is response in Instant Mode ======')
    print(f'response: {response.choices[0].message.content}')

    return response.choices[0].message.content
```

The following example demonstrates how to call K2.5 API with video input:

```python
import openai
import base64
import requests

def chat_with_video(client: openai.OpenAI, model_name:str):
    url = 'https://huggingface.co/moonshotai/Kimi-K2.5/resolve/main/figures/demo_video.mp4'
    video_base64 = base64.b64encode(requests.get(url).content).decode()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text","text": "Describe the video in detail."},
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{video_base64}"},
                },
            ],
        }
    ]

    response = client.chat.completions.create(model=model_name, messages=messages)
    print('====== Below is reasoning_content in Thinking Mode ======')
    print(f'reasoning content: {response.choices[0].message.reasoning_content}')
    print('====== Below is response in Thinking Mode ======')
    print(f'response: {response.choices[0].message.content}')

    # Also support instant mode if pass {"thinking" = {"type":"disabled"}}
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=False,
        max_tokens=4096,
        extra_body={'thinking': {'type': 'disabled'}},  # this is for official API
        # extra_body= {'chat_template_kwargs': {"thinking": False}}  # this is for vLLM/SGLang
    )
    print('====== Below is response in Instant Mode ======')
    print(f'response: {response.choices[0].message.content}')
    return response.choices[0].message.content
```

### Interleaved Thinking and Multi-Step Tool Call

K2.5 shares the same design of Interleaved Thinking and Multi-Step Tool Call as K2 Thinking. For usage example, please refer to the [K2 Thinking documentation](https://platform.moonshot.ai/docs/guide/use-kimi-k2-thinking-model#complete-example).


### Coding Agent Framework

Kimi K2.5 works best with Kimi Code CLI as its agent framework — give it a try at https://www.kimi.com/code.


---

## 7. License

Both the code repository and the model weights are released under the [Modified MIT License](LICENSE).


---

## 9. Contact Us

If you have any questions, please reach out at [support@moonshot.cn](mailto:support@moonshot.cn).
