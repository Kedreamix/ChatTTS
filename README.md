# ChatTTS

首先非常感谢的`ChatTTS`的开源，真如他infer.ipynb里面的一句话“我觉得像我们这些写程序的人，他，我觉得多多少少可能会对开源有一种情怀在吧我觉得开源是一个很好的形式。现在其实最先进的技术掌握在一些公司的手里的话，就他们并不会轻易的开放给所有的人用。”

我自己也深有感触，首先我给予一定的respect，我本身也在DataWahle开源组织里面进行贡献，也带有这样的情怀，非常respect这样的精神



在今天早上看到有关于ChatTTS的介绍，十分的不错，因此也是通过视频和代码学习了以下ChatTTS的架构和使用方法，也给分享出来学习，并且整理了一个colab代码，可以直接运行和学习。

> colab在线体验分享：[https://colab.research.google.com/github/Kedreamix/ChatTTS/blob/main/ChatTTS_infer.ipynb](https://colab.research.google.com/github/Kedreamix/ChatTTS/blob/main/ChatTTS_infer.ipynb)

现在ChatTTS功能总结如下

- ChatTTS 文本转语音
- ChatTTS + LLM 对话
- 高斯采样说话人向量 +调整韵律等

## ChatTTS 文本转语音

如果最简单的使用方法就是直接用官方给的那段代码，实际上也就是ChatTTS文本转语音

```PYTHON
import torch
import ChatTTS
from IPython.display import Audio
# 定义ChatTTS 导入模型
chat = ChatTTS.Chat()
chat.load_models()
# 输入对应的文本
texts = ["<YOUR TEXT HERE>",]
# 语音合成
wavs = chat.infer(texts, use_decoder=True)

Audio(wavs[0], rate=24_000, autoplay=True)
```

当然这是最基础的用法，也是最简单的用法，在我看了ChatTTS的源码之后，首先感叹一下代码写的还是很不错的，还是非常简单易懂的

我发现infer里面有多个参数，首先最简单的就是`skip_refine_text`参数和`use_decoder`参数

- skip_refine_text：这个参数实际上是预处理文本，对于输入的文本会加入一些[uvbreak]的韵律标记。
- use_decoder：这个参数可能是用了不同的deoder的向量的结构，在实际测试中没有发现什么区别，但是作者默认是True

```python
def infer(self, text, skip_refine_text=False, params_refine_text={}, params_infer_code={}, use_decoder=False):
    assert self.check_model(use_decoder=use_decoder)
    # 重写文本，实际上是预处理文本，加入了uvbreak的标记
    if not skip_refine_text:
        text_tokens = refine_text(self.pretrain_models, text, **params_refine_text)['ids']
        text_tokens = [i[i < self.pretrain_models['tokenizer'].convert_tokens_to_ids('[break_0]')] for i in text_tokens]
        text = self.pretrain_models['tokenizer'].batch_decode(text_tokens)
        # 打印处理后的文本
        print('Refine text:')
        print("\n".join(text))
    result = infer_code(self.pretrain_models, text, **params_infer_code, return_hidden=use_decoder)
    if use_decoder:
        mel_spec = [self.pretrain_models['decoder'](i[None].permute(0,2,1)) for i in result['hiddens']]
    else:
        mel_spec = [self.pretrain_models['dvae'](i[None].permute(0,2,1)) for i in result['ids']]
    wav = [self.pretrain_models['vocos'].decode(i).cpu().numpy() for i in mel_spec]
    return wav
```

### skip_refine_text参数

通过调节skip_refine_text参数来进行测试，如果走默认参数，也就是使用进行预处理文本，我们可以看到预处理后的文本的格式

```python
# 6条文本
texts = ["So we found being competitive and collaborative was a huge way of staying motivated towards our goals, so one person to call when you fall off, one person who gets you back on then one person to actually do the activity with.",]*3 \
        + ["我觉得像我们这些写程序的人，他，我觉得多多少少可能会对开源有一种情怀在吧我觉得开源是一个很好的形式。现在其实最先进的技术掌握在一些公司的手里的话，就他们并不会轻易的开放给所有的人用。"]*3     
    
wavs = chat.infer(texts, skip_refine_text = False, use_decoder=True)
```

```bash
 29%|██▊       | 110/384 [00:04<00:11, 24.44it/s]
Refine text:
so we found being competitive and collaborative was a huge way of staying [uv_break] motivated towards our goals, [uv_break] so [uv_break] one person to call when you fall off, [uv_break] one person who gets you back on [uv_break] then one person to [uv_break] actually do the activity with.
so we found [uv_break] being competitive and collaborative [uv_break] was a huge way of staying motivated towards our goals, [uv_break] so [uv_break] one person to call when you fall off, [uv_break] one person who gets you back on then one person [uv_break] to actually [uv_break] do the activity [uv_break] with.
so we found [uv_break] being competitive [uv_break] and collaborative was [uv_break] a huge way of [uv_break] staying [uv_break] motivated towards our goals, [uv_break] so [uv_break] one person to call [uv_break] when you fall off, [uv_break] one person who gets you back on then one person [uv_break] to [uv_break] actually do the activity with.
我 觉 得 像 我 们 这 些 [uv_break] 写 程 序 的 人 ， 他 [uv_break] ， 我 觉 得 多 多 少 少 可 能 会 对 开 源 [uv_break] 有 一 种 [uv_break] 情 怀 在 吧 [uv_break] 啊 我 觉 得 开 源 是 一 个 很 好 的 形 式 [uv_break] 。 那 现 在 其 实 最 先 进 的 技 术 掌 握 在 一 些 公 司 的 手 里 的 话 ， 就 [uv_break] 他 们 并 不 会 轻 易 的 开 放 给 所 有 的 人 用 。
我 觉 得 像 我 们 这 些 写 程 序 的 人 ， 他 ， [uv_break] 我 觉 得 多 多 少 少 可 能 会 对 开 源 有 一 种 [uv_break] 情 怀 在 吧 [uv_break] 啊 我 觉 得 开 源 是 一 个 很 好 的 形 式 [uv_break] 。 然 后 现 在 其 实 最 先 进 的 技 术 [uv_break] 掌 握 在 [uv_break] 一 些 公 司 的 手 里 的 话 [uv_break] ， 就 他 们 并 不 会 轻 易 的 就 开 放 给 [uv_break] 呃 [uv_break] 就 是 所 有 的 人 用 。
我 觉 得 像 [uv_break] 我 们 这 些 写 程 序 的 人 ， 他 [uv_break] 呃 ， [uv_break] 我 觉 得 多 多 少 少 可 能 会 对 开 源 有 一 种 [uv_break] 情 怀 在 吧 [uv_break] 呃 [uv_break] 我 觉 得 开 源 是 一 个 很 好 的 形 式 。 现 在 其 实 最 先 进 的 技 术 [uv_break] 掌 握 在 一 些 [uv_break] 公 司 的 手 里 的 话 [uv_break] ， 就 他 们 并 不 会 轻 易 的 开 放 给 所 有 的 人 用 。
 48%|████▊     | 991/2048 [00:28<00:30, 34.68it/s]
```

从中是可以看到，对文本实际上都进行韵律的处理，在一些地方加上了[uv_break]标记，实际上就是一个停顿或者呼吸，能让生成的语音听起来更加的自然和谐，具体声音可以在colab中体验到。

### use_decoder参数

use_decoder参数体验下来没有什么感觉，但是感觉会觉得use_decoder为True的效果会更加好，整体的质感也会更好。大家也可以多对比一些，decoder为False似乎语音的自然性还不够，后续也会挖掘一些，可能是模型的问题



## ChatTTS + LLM对话

ChatTTS中在内置了一部分LLM对话的code，也就是他可以直接在里面使用大语言模型，也就是外接了一个LLM，这些LLM是通过API直接访问的

如果正常访问的话，大家体验可能需要APIKEY，我懒了一下，仿照ChatTTS的代码，加入了gpt4free的代码，现在就可以免费用了哈哈

```python
# 可使用内置deepspeek，kimi等api
from ChatTTS.experimental.llm import llm_api

API_KEY = ''
client = llm_api(api_key=API_KEY,
        base_url="https://api.deepseek.com",
        model="deepseek-chat")

#---------
# 直接上面忽略，使用gpt4free的API
text = client.call(user_question, prompt_version = 'gpt4free')
text
```

```bash
四川有很多好吃的美食，比如麻辣火锅、水煮鱼、夫妻肺片、回锅肉等。这些都是非常有名的四川美食，口味麻辣鲜香，非常值得一试。
```



## 说话人向量生成

在ChatTTS的介绍中也有说到，说话人生成的主要方法是，首先从高斯噪声中采样， 然后得到一个固定长度的说话人向量，最后作为额外的信息,输入到网络

在提供的infer.ipynb读入了spk_stat.pt文件，仔细看了以后，我猜测是有关于speaker声音的均值和方差，然后进行一个采样，但是文件中没有提供，所以直接运行会报错

```bash
# 暂无提供pt文件，应该是有关于speaker声音的均值和方差，会报错
spk_stat = torch.load('ChatTTS/asset/spk_stat.pt')
# 生成一个随机的说话者向量
rand_spk = torch.randn(768) * spk_stat.chunk(2)[0] + spk_stat.chunk(2)[1]
```

但是我们依然可以随机生成一个说话人向量，这样我们就固定了说话人

```bash
rand_spk = torch.randn(768)

# 替代方案，可以随机生成一个向量作为说话人向量
params_infer_code = {'spk_emb' : rand_spk, 'temperature':.3}
params_refine_text = {'prompt':'[oral_2][laugh_0][break_6]'}
# wav = chat.infer('四川美食可多了，有麻辣火锅、宫保鸡丁、麻婆豆腐、担担面、回锅肉、夫妻肺片等，每样都让人垂涎三尺。', params_refine_text=params_refine_text, params_infer_code=params_infer_code)
wav = chat.infer('四川美食确实以辣闻名，但也有不辣的选择。比如甜水面、赖汤圆、蛋烘糕、叶儿粑等，这些小吃口味温和，甜而不腻，也很受欢迎。', use_decoder=True, params_refine_text=params_refine_text, params_infer_code=params_infer_code)
```



### 文本韵律调整

同时，我们其实还可以调整文本的韵律，前面我们介绍过skip_refine_text参数，实际上他是自动的对文本预处理，但是可能这种预处理是不够精细的，也有可能是文本不想关，我们拿b站视频的例子，我们跳过skip_refine_text参数，自己给出带有韵律的文本，我们不仅可以可控生成停顿的韵律，也可以听到笑声。

在参数里面我还看到了oral等参数，我猜有比较多的情绪控制，就看接下来ChatTTS放出什么了哈哈，期待！

```bash
# 替代方案，可以随机生成一个向量作为说话人向量
params_infer_code = {'spk_emb' : rand_spk, 'temperature':.3}
params_refine_text = {'prompt':'[oral_2][laugh_0][break_6]'}
# wav = chat.infer('四川美食可多了，有麻辣火锅、宫保鸡丁、麻婆豆腐、担担面、回锅肉、夫妻肺片等，每样都让人垂涎三尺。', params_refine_text=params_refine_text, params_infer_code=params_infer_code)
wav = chat.infer('ChatTTS 不仅能够生成自然流畅的语音[uv_break],还能控制[laugh]笑声[laugh],[uv_break]停顿啊和语气词啊等副语言现象[uv_break]。其这个韵律呢超越了许多开源模型。',
                 skip_refine_text = True, use_decoder=True,
                 params_refine_text=params_refine_text, params_infer_code=params_infer_code)
```



## To be Finished

```python
import torch
import ChatTTS
from IPython.display import Audio

chat = ChatTTS.Chat()
chat.load_models()

texts = ["<YOUR TEXT HERE>",]

wavs = chat.infer(texts, use_decoder=True)

Audio(wavs[0], rate=24_000, autoplay=True)
```

## Disclaimer: For Academic Purposes Only

The information provided in this document is for academic purposes only. It is intended for educational and research use, and should not be used for any commercial or legal purposes. The authors do not guarantee the accuracy, completeness, or reliability of the information. The information and data used in this document, are for academic and research purposes only. The data have been obtained from publicly available sources, and the authors do not claim any ownership or copyright over the data.

免责声明：仅供学术交流

本文件中的信息仅供学术交流使用。其目的是用于教育和研究，不得用于任何商业或法律目的。作者不保证信息的准确性、完整性或可靠性。本文件中使用的信息和数据，仅用于学术研究目的。这些数据来自公开可用的来源，作者不对数据的所有权或版权提出任何主张。
