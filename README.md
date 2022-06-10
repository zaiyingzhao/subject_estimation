# subject_estimation

pandas, MeCab, gensim, tensorflow.kerasを用い, 教師あり学習により大学入試問題文からその科目を推定するコードを作成した。

## 1. pandasによるデータ読み込みおよび整形
入試問題文とその科目はそれぞれexamination_test.csv, examination_answer.csvとcsvファイルで与えられており, これらをまずpandasを用いて読み込み, かつMeCabが利用できる形に整形した。

## 2. MeCabによる文章の分かち書き
MeCabを用いてgensimのdoc2vecが利用できるように文章を分かち書きした。

## 3. gensimによるdoc2vec
はじめにexamination_answer.csvで与えられている答えの科目を対応する数値ラベルに変換する関数labelize()を作成し, trainlabelにその数値ラベル, trainarrayにgensimのdoc2vecによって得られた文章の特性を表すベクトルを格納した。

## 4. tensorflow.kerasによる教師あり学習
今回は教師あり学習にkeras.optimizers.RMSpropを利用した。試行回数が少ないと有意な結果が得られないため, 試行回数は10000回とした。学習が修了するまでにおよそ30分かかる。

## 5. 学習後のテスト
学習後に同じexamination_test.csvから科目を推定させた。その結果得られた正答率はおよそ89％であり, 推定の結果はpred.csvに格納した。