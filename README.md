# AutoEncoder
一次元輝度分布センサを用いた転倒検知をAutoEncoderを用いて直立データのみの学習で行う

基本的にコードを上から実行していけば解析できます
NormalAutoEncoder.ipynb, DenoisingAutoEncoder.ipynb, VAE .ipynb
基本的にこの三つのファイルを実行すればいいです。

関係してくるのは以下のファイルたち、そのほかのファイルは特に気にする必要なし
### dataset.py
データセットを作ってる、データ拡張、エポック数などの指定、ノイズ付加、配列のシャッフルなど

### LossFunction.py
自作Loss関数を用いたいなら、このファイルの雛形をもとに変更していけば使えます。

### mainmodel.py modelVAE.py
モデルのクラスが入っている、

### dataloader.py
OCSVM用データセット作成用

### calcTime.ipynb
解析時間計測用