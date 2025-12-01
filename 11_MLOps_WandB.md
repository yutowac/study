![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/986141/50bb2511-fb7d-40de-a416-d9f8ff79991a.png)

## 機械学習モデルの育て方

機械学習パラメータの調整と管理には **Weights & Biases (WandB)** というサービスが役に立ちます。

https://wandb.ai/site/ja/

このような形でパラメータを管理しモデルを比較することができます。

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/986141/614e852d-2e44-41d5-ad45-b4bb0316a5fb.png)


### パラメータの調整とは
機械学習モデルを作る際に目指すことは、誤差の最小化です。つまり、Y軸に誤差を表現した関数を用意した場合に、その極小点を探索することになります。

ですが、誤差関数が複雑な場合、極小点は1つでなく、意図しない極小値で学習を終了する可能性があるため、適切な極小値を学習するためのパラメータを調整する必要があります。

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/986141/4a87407b-619f-4a77-97c2-f29b51db36f8.png)


## 準備するもの

- [WandB](https://wandb.ai/site/ja/) のアカウント

- モデルをつくるデータセット

データは Kaggle のコンペデータを使います。タスクはお店の売上予測です。

https://www.kaggle.com/competitions/store-sales-time-series-forecasting

## 1. WandB のアカウント作成

こちらからサインアップできます。詳細は割愛しますが、迷う部分は特にないと思います。

https://wandb.ai/signup

ここでは Kaggle の Notebook 環境で進めます。もちろんローカル環境でも使えます。

まずは、WandB の API キーを使うので、取得していきましょう。
WandB のマイページのユーザーアイコンから `User settings` を選択し、下の方へスクロールすると `API keys` の項目があるので、`Reveal` ボタンを押し表示させましょう。このキーをコピーしておきます。

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/986141/7a869d57-b970-4e18-bf0d-69a229326be2.png)

## 2. Kaggle Notebook で WandB を接続してみる

こちらにサンプルコードを用意しました。このコードに書き加えていきます。
そのままでも割としっかり前処理してますので、よろしければご活用ください。

https://www.kaggle.com/code/yutodennou/practice-wandb-lightgbm

サイドバーの`Session options` の`Internet on` のトグルをオンにしておきます。

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/986141/c7ebc5a1-7b8e-4610-b168-01011219b6de.png)


`Add-ons`から`Secrets`を選択し、WandB の API キーを環境変数に入れます。

![スクリーンショット 2025-11-08 202357.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/986141/eeb5b8b6-0479-4079-8100-07801dfc460d.png)


キーとバリューは以下のようにします。

`Label`: wandb_api_key
`VALUE`: 1.で取得した API キー



### 2-1. ライブラリのインポートと WandB へのログイン

他のライブラリと一緒にこちらをインポートしておきましょう。

```python
import wandb
from wandb.integration.lightgbm import log_summary, wandb_callback
from dataclasses import dataclass, field, asdict
from typing import List
```

ログインを試してみます。
以下のコードでキーを取得できます。冒頭に実行して、`secret_value_wb` にキーを格納しておきましょう。

```python
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_wb = user_secrets.get_secret("wandb_api_key")
wandb.login(key=secret_value_wb)
```

実行して `True` が返却されればログイン成功です。

### 2-2. Config の設定と WandBの初期化 

では、まずは管理していくパラメータを決めていきます。パラメータを`CFG`としてクラスで定義し、WandB に記録していくようなイメージを持っておきましょう。

データの加工が概ね終えた段階のところから始めましょう。
サンプルコードの場合は以下の「ここから書いていきます」を編集していきます。

```python
# -----------------------------
# 4) Split train/valid/test / 学習・検証・テスト分割
# -----------------------------
train_df = data[data['is_train']==1].copy()
test_df  = data[data['is_train']==0].copy()

# EN: Use last 28 days of train as validation (simple, fast).
# JA: 学習データの最後28日を検証に使用（簡易・高速）。
last_date = train_df['date'].max()
val_from = last_date - pd.Timedelta(days=27)
train_part = train_df[train_df['date'] < val_from]
valid_part = train_df[train_df['date'] >= val_from]

# EN: Feature columns / JA: 特徴量カラム
drop_cols = ['id', 'date', 'sales', 'is_train']
feature_cols = [c for c in train_df.columns if c not in drop_cols]

# -----------------------------
ここから書いていきます
# -----------------------------

X_train = train_part[feature_cols]
y_train = train_part['sales']
X_valid = valid_part[feature_cols]
y_valid = valid_part['sales']
```

パラメータは CFG クラスで定義していきます。わかりやすく初期値を入れています。

```python
@dataclass
class CFG:
    exp_name: str = 'exp001'
    learning_rate: float = 0.1
    num_leaves: int = 31
    n_estimators: int = 10000
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    min_data_in_leaf: int = 50
    stopping_rounds: int = 200
    log_evaluation: int = 200
    random_state: int = 42
    objective: str = 'regression'
    metric: str = 'rmse'
    feature: List[str] = field(default_factory=lambda:feature_cols)

config = CFG()

```

WandB には辞書型で渡すので`asdict`を使って`config_dict`をつくっておきます。

```python
config_dict = asdict(config)
print(config_dict)

>>>{'exp_name': 'exp001', 'learning_rate': 0.1, 'num_leaves': 31, 'n_estimators': 10000, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'min_data_in_leaf': 50, 'stopping_rounds': 200, 'log_evaluation': 200, 'random_state': 42, 'objective': 'regression', 'metric': 'rmse', 'feature': ['store_nbr', 'family', 'onpromotion', 'city', 'state', 'type', 'cluster', 'is_holiday', 'oil_price', 'oil_ma7', 'oil_ma30', 'transactions', 'trans_ma7', 'trans_ma30', 'doy', 'woy', 'dom', 'dow', 'month', 'year', 'is_weekend', 'lag_7', 'lag_14', 'lag_28', 'roll_mean_7', 'roll_median_7', 'roll_max_7', 'roll_min_7', 'roll_std_7', 'roll_mean_14', 'roll_median_14', 'roll_max_14', 'roll_min_14', 'roll_std_14', 'roll_mean_28', 'roll_median_28', 'roll_max_28', 'roll_min_28', 'roll_std_28', 'onpromo_lag_7', 'trans_lag_7', 'onpromo_lag_14', 'trans_lag_14']}
```

WandBを初期化してみましょう。

```python
wandb.init(
    project="store_sales_time_series_forecasting",
    config=config_dict,
    name=config.exp_name
)
```

このような出力があれば OK です。ちなみに `Display W&B run`を押すと Notebook 上で WandB のプロジェクトページのダッシュボードを見ることができます。

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/986141/bc1e7568-cf97-4652-a366-539c6abb33f5.png)

ちなみにWandBを開いてみると、新たに "**store_sales_time_series_forecasting**" というプロジェクトができています。

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/986141/14384788-c196-459a-961b-f4f53c901414.png)


## 3.解析結果を管理してみよう

### 3-1. config を使ってモデルを作成

あとは、実はそれほど難しいことはありません。作成した`config`を使ってモデルを定義していきます。
サンプルコードの場合は次のように書き換えていきましょう。

```python
X_train = train_part[feature_cols]
y_train = train_part['sales']
X_valid = valid_part[feature_cols]
y_valid = valid_part['sales']
```
↓
```python
X_train = train_part[config.feature]
y_train = train_part['sales']
X_valid = valid_part[config.feature]
y_valid = valid_part['sales']
```



LightGBM のパラメータも同様です。コールバックには WandB と連携するコールバック関数を使うと、モデルの訓練プロセスに関する様々な情報をリアルタイムで記録することができます。

https://docs.wandb.ai/models/integrations/lightgbm

```python
# -----------------------------
# 5) Train LightGBM / LightGBM学習
# -----------------------------
lgb_params = {
    'objective': config.objective,
    'metric': config.metric,
    'learning_rate': config.learning_rate,
    'num_leaves': config.num_leaves,
    'feature_fraction': config.feature_fraction,
    'bagging_fraction': config.bagging_fraction,
    'bagging_freq': config.bagging_freq,
    'min_data_in_leaf': config.min_data_in_leaf,
    'verbose': -1,
    'seed': config.random_state,
}

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_valid = lgb.Dataset(X_valid, label=y_valid, reference=lgb_train)

evals_result = {}

callbacks = [
    lgb.early_stopping(stopping_rounds=config.stopping_rounds, verbose=True),
    lgb.log_evaluation(period=config.log_evaluation),
    lgb.record_evaluation(evals_result),
    wandb_callback()    
]

model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=config.n_estimators,
    valid_sets=[lgb_train, lgb_valid],
    valid_names=['train','valid'],
    callbacks=callbacks
)

# EN: Report RMSLE for reference (competition uses RMSE, but RMSLE helps sanity-check).
# JA: 参考としてRMSLEを表示（本番指標はRMSEだが検算に有用）。
valid_pred = model.predict(X_valid, num_iteration=model.best_iteration)
val_rmsle = rmsle(y_valid.values, np.maximum(valid_pred, 0))
print(f"[Validation] RMSLE: {val_rmsle:.6f}")

```

### 3-2. 実行 & WandB で結果をチェック

ここまでで準備は整いました。順番に実行してみましょう。
編集後の Notebook も用意しましたので、必要な方はこちらを使ってください。

https://www.kaggle.com/code/yutodennou/practice-wandb-lightgbm-whole-codes

例えば、Notebook 上では以下のコードを実行するとこのような学習過程の出力があります。

```python
train_rmse = evals_result['train']['rmse']
valid_rmse = evals_result['valid']['rmse']

plt.figure(figsize=(8, 5))
plt.plot(train_rmse, label='Train RMSE')
plt.plot(valid_rmse, label='Valid RMSE')
plt.xlabel('Iteration (boosting round)') 
plt.ylabel('RMSE')   
plt.title('LightGBM Learning Curves')  
plt.legend()
plt.grid(True)
plt.tight_layout()
```

![download.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/986141/a3e4edd4-d46b-4421-8457-1717d87441ed.png)

もちろんこのような出力も WandB に記録されます。

`log_summary()`を実行します。

```python
log_summary(model, save_model_checkpoint=True)
```

これで、WandB でどのように記録されているか確認しましょう。

ワークスペース上に `exp001` という実行結果ができています。

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/986141/597d9c35-7059-4f8d-b93d-4a84d66769ea.png)

`exp001`を開くと、`Charts` タブで先ほどの学習過程のグラフを見ることができます。

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/986141/b347a3e6-cfd4-4ef9-a4e5-c7767b6afe1e.png)

また`Overview` タブでパラメータやサマリーも確認できます。

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/986141/2b5e345d-81f3-4eba-9dfb-14e3cb8867e6.png)


最後に`finish()`関数で終了しましょう。

```python
wandb.finish()
```

実行サマリーが出力されます。
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/986141/ce0f035c-2067-4cd9-aa63-38d118a2e859.png)

## 補足

### ディレクトリ構成
今回の記事をベースに各パラメータでの実行結果を記録できますが、実験管理のディレクトリ構成はもっと複雑になる可能性があります。こちらは末尾に紹介している参考書籍を読むとわかりやすいです。

参考までに、いくつかアイデアを紹介します。

### 基本：１実験１ Notebook

```
experiments/
├─exp001.ipynb
├─exp002.ipynb
├─exp003.ipynb
└─exp004.ipynb
```

### 関数など機能を切り離す：１実験１ディレクトリ

```
experiments/
├─exp001
│  ├─config.yaml
│  ├─train.py
│  └─utils.py
├─exp002
│  ├─config.yaml
│  ├─train.py
│  └─utils.py
└─exp003
     ├─config.yaml
     ├─train.py
     └─utils.py
```

出力結果を管理する場合。expで揃える。

```
project/
├─experiments/
│ ├─exp001
│  │  ├─config.yaml
│  │  ├─train.py
│  │  └─utils.py
│ ├─exp002
│  │  ├─config.yaml
│  │  ├─train.py
│  │  └─utils.py
│  └─exp003
│       ├─config.yaml
│       ├─train.py
│       └─utils.py
├─results/
      ├─exp001
       │  ├─result.csv
       │  └─model.pkl
      ├─exp002
       │  ├─result.csv
       │  └─model.pkl
       └─exp003
            ├─result.csv
            └─model.pkl
```

### WandB のインテグレーション
この記事では、LightGBM を使いましたが、PyTorch や TensorFlow などでの人気なMLフレームワークでのトレーニングプロセスの追跡ももちろん可能です。

#### 1. テーブルデータのコンペ

```python
import xgboost as xgb
from wandb.integration.xgboost import WandbCallback
model = xgb.train(
  params,
  dtrain,
  num_boost_round=config.n_estimators,
  evals=[(dtrain, 'train'), (dtest, 'eval')],
  early_stopping_rounds=50,
  verbose_eval=100,
  callbacks=[
    WandbCallback()
  ]
)
```

#### 2. NLPのコンペ
Hugging Face のインテグレーションが有効。事前学習済みのトランスフォーマーモデルを適用できる。

```python
from transformers import TrainingArguments, Trainer
```

#### 3. 画像のコンペ
`precision` や `recall` などの指標を残したい時に`Ultralystics`が有効。

```python
from wandb.integration.ultralystics import add_wandb_callback
```


## 参考

https://www.shoeisha.co.jp/book/detail/9784798187457
