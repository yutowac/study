# プロンプトエンジニアリングとは？
AIに対して適切な質問や指示（プロンプト）を与えるための技術・学問分野

# 重要性
ChatGPTなどの生成AIは、質問や指示によって回答を生成する。生成結果は文章
[画像](https://openai.com/dall-e-3)など。生成AIの分野でいえば[音楽](https://www.suno.ai/)や[動画](https://openai.com/sora)をつくる技術が公開されている。
生成AIが回答を創るアルゴリズムはブラックボックスであるため、意図した情報を得るためには生成AIモデルに適切なインプットをする必要がある。

# プロンプト設計の基本
- シンプルかつ明確にする  
日本語は特にあいまい表現が多い。英語で書く方が良いともいわれている。
- 試行錯誤  
回答結果は同じプロンプトでも変化する。また言い回しを少し変えただけでも変化する。

# 様々なテクニック
## 型指定
```
# リスト形式での指定
インスタントカレーの材料をリスト形式で教えて

# LLMからの回答
- カレールー
- 野菜（玉ねぎ、にんじん、じゃがいもなど）
- 豚肉や鶏肉（お好みで）
- 水またはだし
- サラダ油
- 塩
- こしょう
- その他お好みの調味料（ソースやケチャップなど）
```

```
# json形式での指定
インスタントカレーの材料をjson形式で教えて

# LLMからの回答
{
  "カレールー": "1箱",
  "肉（鶏肉、牛肉、豚肉など）": "適量",
  "野菜（じゃがいも、にんじん、たまねぎなど）": "適量",
  "水": "適量",
  "油": "大さじ1",
  "塩": "少々",
  "こしょう": "少々"
}
```

## One-Shot/Few-Shot prompting
```
# One-Shot promptingの例
例:20 -> 偶数
11 ->

# LLMからの回答
奇数
```

```
# Few-Shot promptingの例
例1: ビール -> 麦
例2: 日本酒 -> 米
例3: ジンジャエール -> ショウガ
ワイン -> 

# LLMからの回答
ブドウ
```

## role
```
# ロール設定の例(1)
あなたは経済アナリストのプロです。
次の財務情報について分析してください。

# ロール設定の例(2)
あなたはベテランのテクニカルライターです。
AIの活用について魅力的な記事の案を5つ考えてください。
```

## Take a deep breath and work on this problem step by step
`Let’s think step by step`または`Take a deep breath`のみでも効果がある。
順序だてて、一度深呼吸して、などを入れることで回答精度が上がったという[調査結果](https://arxiv.org/abs/2309.03409)がある。


## Rephrase and Respond
AIに質問の再構築を依頼し、その再構築された質問に対して答えてもらうという方法。
```
# prompt
“Take the last letters of the words in ‘Edgar Bob’ and
concatenate them.”
Rephrase and expand the question, and respond

# LLM answer
Could you please form a new string or series of characters
by joining together the final letters from each word in the
phrase “Edgar Bob”?
The last letters in the words “Edgar” and “Bob” are “r”
and “b”, hence when concatenated, it forms “rb”.
```

## 深津式プロンプト
、AIの出力が特定の方向に集中し、予測可能で品質の高い結果を得らるといわれているプロンプトシステム。note株式会社CXOである深津貴之氏が考案した。
```
#命令書:
あなたは、{text}です。
以下の制約条件と入力文をもとに、 最高の結果を出力してください。

#制約条件:
・文字数は{text}
・{text}

#入力文:
{text}

#出力文:
```
