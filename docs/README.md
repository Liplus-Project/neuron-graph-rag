# NGR 文書索引

仕様・設計・実験の用途別入口です。設計案と実装済み機能は各文書の状態を確認してください。

- [製品仕様](specifications/README.md)
- [利用・運用](guides/README.md)
- [設計](architecture/README.md)
- [実験・観測](experiments/README.md)
- [意思決定・Wiki](decisions/README.md)

## 整理と凍結契約

既存83ファイルのうち7ファイルを用途別フォルダへ移動し、76ファイルは元のパスとbytesを維持していました。#309 で凍結済みの実験・観測 module、runner、テスト、fixture、証拠と、それらを対象とする文書を削除し、台帳に残る文書は18ファイル（移動2、元パス維持16）になりました。削除した文書と証拠は git 履歴に残ります。

文書を元パスに固定していた manifest 内の hash registry は、同じ変更で削除されたため残っていません。各カテゴリの「元パス維持」は、移動しなかった文書を示します。[分類台帳](https://github.com/Liplus-Project/neuron-graph-rag/blob/main/docs/documentation-layout.json)で各文書の移動先と維持文書からの参照を確認できます。

残した fixture・manifest・観測証拠の内部パスは当時の識別子として維持します。古い実験の再実行や共有データベースへのアクセスは行いません。

## Wiki

Home.md、_Footer.md、Decision-Structure.mdは元パスを維持しています。既存のHome.mdからのmain上のリンク先も維持しています。この索引はGitHubリポジトリ向けで、Wiki-onlyのページや同期設定は変更していません。
