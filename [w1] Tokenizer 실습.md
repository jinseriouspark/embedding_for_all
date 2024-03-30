
[colab 링크](https://colab.research.google.com/drive/1i5IDWz74jFuoXBH2HD_UXC9Z3q4KFDI5)

# Tokenizers

1. SentencePiece
2. BPE from scratch (Sentencepiece paper : link)


## 1. SentencePiece

참고자료 : https://devocean.sk.com/blog/techBoardDetail.do?ID=164570&boardType=techBloghttps://process-mining.tistory.com/191

프로세스

- corpus를 subword단위로 쪼갠 다음, 빈도수를 계산해 높은 빈도로 함께 등장한 subword를 병합해 학습할거에요.
- 적절한 subword 를 파악하기 위해서는 사전에 정리된 corpus도 필요해요.
- 링크1 : https://github.com/e9t/nsmc
- 링크2 : [https://huggingface.co/datasets/nsmc](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fhuggingface.co%2Fdatasets%2Fnsmc)



BBPE를 Sentencepiece 라이브러리를 활용해 학습해보겠습니다. 간단해요!
학습 코드는 다음과 같습니다.
```

import sentencepiece as spm
from pathlib import Path

prefix = 'sp-nsmc-test'
vocab_size = 31900 - 7
# vocab_size 는 10_000 ~ 52_000 사이이며 모델에 따라 다르다고 합니다.
# subwords token 31_900개
# special token 7개 제외
# additional_special_tokens (T5용 <extra_id_XX> 토큰) 100개
# -> 총 32_000개

spm.SentencePieceTrainer.train(
     f' --input={corpus} --model_prefix={prefix}' +
     f' --vocab_size={vocab_size + 7}' + #  최종 데이터셋 갯수
      ' --model_type=bpe' + # 어떤 토크나이저를 사요할 것인지 설정 : unigram, bpe, char, word
      ' --max_sentence_length=999_999' +
      ' --pad_id=0 --pad_piece=<pad>' +
      ' --unk_id=1 --unk_piece=<unk>' +
      ' --bos_id=2 --bos_piece=<s>' +
      ' --eos_id=3 --eos_piece=</s>' +
      ' --user_defined_symbols=<sep>,<cls>,<mask>' +
      ' --byte_fallback=True'
)

```

학습 완료 시, .model 과 .vocab 파일을 얻을 수 있습니다. 
학습된 모델을 활용하여 토크나이징하면 다음과 같은 결과를 얻을 수 있어요.

![](https://velog.velcdn.com/images/magnussapiens/post/b2707f9b-7216-4d1b-9539-ad3a98ccb646/image.png)


## 2. BPE from Scratch
구현은 byte 가 아니라 char 기준으로 진행했어요.

프로세스

- 빈도 계산
- 모든 단어를 글자 단위로 분리
- 병합



### 1. 빈도 계산

: 문장 내 아래 단어 발생 빈도를 집계합니다.

문장 : 나는 **계란밥**을 먹는다. **계란밥**! 얼마나 맛있던가. 나는 **계란밥**이 세상에서 제일좋다. 물론 **간장밥** 러버도 있겠지. 예를 들어 내 동생은 **간장밥** 하나만 있으면 1주일을 버틸 수 있다. 어떻게 하면 **간장밥** 그 단일 메뉴가 아침을 해결할 수 있을까. 말이 안된다. 중학생 때 먹은 **간장밥**, 그게 내 마지막 **간장밥**이다.

나는 내동생과 서로를 이해하지 못했다. 어느날 부모님이 계란과 간장을 섞어 **간장계란밥**을 만들었다. 노란색의 밥에 검정 소스가 있는, 이상한 그 메뉴의 이름을 묻자 부모님은 웃으며 내게 말했다. '**간장계란밥**'

**볶음밥**이야? 라고 물어봤지만 부모님은 고개를 저었다. 아니 **볶음밥**이라니? 이건 기름에 볶지 않았어. **볶음밥**의 기본은 볶음인걸? 이건 간장과 계란이 합쳐진 비빔밥이야.

위의 문장을 아래처럼 만들고자 합니다.

```
< dictionary>
: 훈련데이터에 있는 단어와 등장 빈도수
- '계란밥': 3
- '간장밥' : 5
- '간장계란밥' : 2
- '볶음밥' : 3

  <vocabulary>
- 계란밥
- 간장밥
- 간장계란밥
- 볶음밥
```

### 2. 분리: 모든 단어를 글자 단위로 나눔(char)

```
< dictionary>
-  계 란 밥 : 3
    - 간 장 밥 : 5
    - 간 장 계 란 밥 : 2
    - 볶 음 밥 : 3

  < vocab>
- 계, 란, 밥, 간, 장, 볶, 음
```

### 3. 병합

: 몇 번 merge 해서 하나의 유니그램으로 통합할까요?

- num_merges = 10 -> 총 10회 반복할거에요

아래는 그 merge 횟수에 따라 통합되는 생김새를 예시로 보여드립니다.

1회 : 빈도수가 5 + 2= 7 인 (간, 장) 의 쌍을 간장 으로 통합

```
- 계 란 밥 : 3
- 간장 밥: 5
- 간장 계 란 밥 : 2
- 볶 음 밥 : 3
```

2회 : 빈도수가 2 + 3 = 5로 가장 높은 (계, 란) 을 계란 으로 통합, (간장, 밥) 을 간장밥 으로 통합

```
- 계란 밥: 3
- 간장밥: 5
- 간장 계란 밥 : 2
- 볶 음 밥 : 3
```

3회 : 빈도수가 3 + 2 = 5로 가장 높은 (계란, 밥) 을 계란밥으로 통합

```
- 계란밥 : 3
- 간장밥 : 5
- 간장 계란밥 : 2
- 볶 음 밥 : 3
```

... 반복

3번 반복하였을 때 결과는 다음과 같아요.

```
< dictionary>
- 계란밥 : 3
- 간장밥 : 5
- 간장 계란밥 : 2
- 볶 음 밥 : 3

< vocab>
- 계, 란, 밥, 간, 장, 볶, 음, 계란밥, 간장밥, 간장
```

만약, 계란간장볶음밥이 등장한다면?

- 계란 간장 볶 음 밥 으로 분리 되며, 이 모든 것은 단어 집합에 있는 단어이기 때문에 OOV가 아니게 됩니다!


먼저, 필요한 함수를 아래와 같이 정의합니다.
```
import re, collections

# Character pair encoding
def get_word_freq(corpus):
  word_freq = collections.defaultdict(int)
  for sentence in corpus:
    for word in sentence.split():
      word_space = ' '.join([w for w in word if len(word)>1])
      word_freq[word_space] += 1
  return word_freq

def get_stats(vocab):
  pairs = collections.defaultdict(int)
  for word, freq in vocab.items():
    symbols = word.split()
    for i in range(len(symbols)-1):
      pairs[symbols[i], symbols[i+1]] += freq
  return pairs

def merge_vocab(pair, v_in):
  v_out = {}
  bigram = re.escape(' '.join(pair))
  p = re.compile(f'(?<!\S)' + bigram + r'(?!\S)')
  for word in v_in:
    w_out = p.sub(''.join(pair), word)
    v_out[w_out] = v_in[word]
  return v_out
 ```
 
 그리고 나서, corpus 를 가져옵니다
 ```
 # 위의 예시에 따른 결과값을 확인하기 위해 띄어쓰기를 부자연스럽게 적용하였습니다.
corpus = [
    '나는 계란밥 을 먹는다',
    '계란밥 ! 얼마나 맛있던가.',
    '나는 계란밥 이 세상에서 제일좋다.',
    '물론 간장밥 러버도 있겠지.',
    '예를 들어 내 동생은 간장밥 하나만 있으면 1주일을 버틸 수 있다.',
    '어떻게 하면 간장밥 그 단일 메뉴가 아침을 해결할 수 있을까.',
    '말이 안된다. 중학생 때 먹은 간장밥 , 그게 내 마지막 간장밥 이다.',
    '나는 내동생과 서로를 이해하지 못했다.',
    '어느날 부모님이 계란과 간장을 섞어 간장계란밥 을 만들었다.',
    '노란색의 밥에 검정 소스가 있는, 이상한 그 메뉴의 이름을 묻자 부모님은 웃으며 내게 말했다.',
    '간장계란밥',
    '볶음밥 이야? 라고 물어봤지만 부모님은 고개를 저었다.',
    '아니 볶음밥 이라니? 이건 기름에 볶지 않았어.',
    '볶음밥 의 기본은 볶음인걸? 이건 간장과 계란이 합쳐진 비빔밥이야.'
      ]

 ```
 
 앞서 설명한 BPE 방식대로 단어를 만들게 되면 아래와 같습니다. 
 ```
 bpe_vocab_history = {}
vocab = get_word_freq(corpus)
num_merges = 10
for i in range(num_merges):
  pairs = get_stats(vocab)
  best = max(pairs, key=pairs.get)
  vocab = merge_vocab(best, vocab)
  bpe_vocab_history[best] = i
 ```
 
 ![](https://velog.velcdn.com/images/magnussapiens/post/68c089c4-dc2d-47dc-9ce7-27c45ac18460/image.png)
 
 
그 다음, BPE merge 방식에 따라 단어를 만듦니다.

```
class BPE_tokenizer:
  def __init__(self, num_merges):
    self.num_merges = num_merges
    self.vocab = None
    self.byte_pairs = None
    self.token_vocab = None
    self.bpe_vocab_history = {}

  # BPE알고리즘 학습
  def train(self, corpus):
    self.vocab = get_word_freq(corpus)
    for i in range(self.num_merges):
      self.byte_pairs = get_stats(self.vocab)
      best = max(self.byte_pairs, key=self.byte_pairs.get)
      self.vocab = merge_vocab(best, self.vocab)
      self.bpe_vocab_history[best] = i

    self.token_vocab = set() # 완료 시 만든 vocab 을 추가해줌
    for word in self.vocab.keys():
      self.token_vocab.update(word.split())

  def tokenize(self, text):
    tokens = text.split()
    tokenized_text = []
    for token in tokens:
      if token in self.token_vocab:
        tokenized_text.append(token)
      else:
        tokenized_text.extend(self.byte_tokenize(token))
    return tokenized_text

  # 텍스트 토큰화
  def byte_tokenize(self, token):
    tokens = []
    while len(token) > 0:
      found = False
      for i in range(len(token), 0, -1):
        subword = token[:i]
        #print(subword)
        if subword in self.token_vocab:
          tokens.append(subword)
          token = token[i:]
          found = True
          break
      if not found :
        tokens.append(token[0])
        token = token[1:]
    return tokens
   ```
   
num_merges 는 목표 vocab_size 와 token 품질에 따라 adaptive 하게 적용할 수 있습니다. 병합이 완료된 모델을 활용해 토크나이징이 얼마나 잘 되었는지는 아래처럼 확인할 수 있어요.

![](https://velog.velcdn.com/images/magnussapiens/post/7a4b6ed6-f4f5-4d9c-bb64-336b955c522c/image.png)

