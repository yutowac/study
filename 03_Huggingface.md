## Hugging Faceとは？

[Hugging Face](https://huggingface.co/)は機械学主モデルやデータセット、アプリを共有するプラットフォームです。
GitHubのようにバージョン管理や実行環境も備えていることからAI版GitHubなどとも言われています。
![image](https://github.com/yutowac/study/assets/44987057/1aa2e1e7-aab1-4ffc-8183-5f6425cc5ebc)

🤗←Hugging Face

## コンテンツ
様々なサービスを提供しています。
1. **Hugging Face Hub**  
  概要: モデル、データセット、トレーニングスクリプトを共有するためのプラットフォーム  
  特徴: 数千の事前訓練済みモデルやデータセットが利用可能。コミュニティによる貢献が活発で、最新の研究成果にアクセスできる

3. **Transformersライブラリ**  
  概要: オープンソースのライブラリで、さまざまなトランスフォーマーモデル（BERT, GPT-2, RoBERTaなど）を提供  
  特徴: モデルのトレーニング、評価、推論が容易に行える。幅広いタスク（テキスト分類、生成、翻訳など）に対応

5. **Datasetsライブラリ**  
  概要: 機械学習モデルのトレーニングに必要なデータセットを簡単に取得・操作できるライブラリ  
  特徴: 数百のデータセットが利用可能で、データの前処理や操作が簡単に行える

7. **Tokenizersライブラリ**  
  概要: 高速で柔軟なトークナイザを提供するライブラリ  
  特徴: NLPタスクに必要なトークン化を効率的に実行。PythonおよびRustで実装され、高速処理が可能

9. **Inference API**  
  概要: Hugging Face Hubにホストされているモデルを簡単に利用できるAPIサービス  
  特徴: REST APIを通じてモデルにアクセスし、推論結果を得ることができる。開発者向けの簡単な統合が可能

11. [**AutoNLP**](https://huggingface.co/autotrain)  
  概要: ノーコードで機械学習モデルを自動的にトレーニングするプラットフォーム  
  特徴: データをアップロードするだけで、最適なモデルが自動的に選定・トレーニングされる。特に非技術者向け

13. [**Gradio**](https://www.gradio.app/)  
  概要: 機械学習モデルのインターフェースを簡単に作成できるツール  
  特徴: Webブラウザ上で動作するインターフェースを簡単に作成し、モデルのデモや評価に使用できる

15. **Spaces**
  概要: Hugging Face Hub上でホスティングされるアプリケーションプラットフォーム
  特徴: ユーザーが自分のアプリケーションをホストし、共有することができる。GradioやStreamlitなどのフレームワークをサポート
16. **Courses and Tutorials**  
  概要: 自然言語処理や機械学習のオンラインコースやチュートリアルを提供  
  特徴: 初心者から上級者まで対応。実践的なハンズオンチュートリアルが豊富

18. **Enterprise Solutions**  
  概要: 企業向けのカスタマイズされたサポートとソリューション  
  特徴: セキュリティ、プライバシー、スケーラビリティに対応したサービス。専用サポートやトレーニングも提供

20. **Community and Research**  
  概要: オープンソースコミュニティと共同研究を推進  
  特徴: 研究者や開発者が協力し合うことで、最新の技術や研究成果を迅速に取り入れることができる

とにかくAIアルゴリズム関係のものは全てあると考えて良いです。例えばこんなものもあります。
- [sample](https://huggingface.co/learn/cookbook/index)

## Inference APIを使ってみよう
モデルを探してみてください。
[sample](https://huggingface.co/sd-community/sdxl-flash)

## スペースを使ってみよう

- [sample1](https://huggingface.co/spaces/jbilcke-hf/ai-comic-factory)
- [sample2](https://huggingface.co/spaces/ehristoforu/dalle-3-xl-lora-v2)

## 企業としての利用
[Organization企業の一覧](https://huggingface.co/organizations)

[事例](https://huggingface.co/ByteDance)

[sample](https://x.com/RisingSayak/status/1782597654901330393)

1. **NLPモデルの利用**  
Hugging Faceは、トランスフォーマーモデル（BERT, GPT, RoBERTaなど）を提供しており、これらのモデルを企業の特定のニーズに合わせてカスタマイズすることができます。
[事例](https://dev.classmethod.jp/articles/huggingface-jp-text-classification/)

2. **トレーニングとファインチューニング**  
企業はHugging Faceのライブラリを使用して、独自のデータセットを用いてモデルをトレーニングし、既存のモデルをファインチューニングすることができます。これにより、特定の業界やドメインに最適化されたモデルを作成することができます。

3. **APIの利用**  
Hugging Faceは、モデルのAPIアクセスを提供しており、これを利用することで簡単にNLP機能をアプリケーションに統合できます。これにより、エンジニアリングリソースを節約しながら高品質なNLP機能を迅速に導入することができます。
→Inference API


## 参考
[https://ledge.ai/articles/The-Al-community](https://ledge.ai/articles/The-Al-community)
[https://note.com/noa813/n/n5a75675a9331](https://note.com/noa813/n/n5a75675a9331)
