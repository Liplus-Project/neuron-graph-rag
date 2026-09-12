# NGR 文書索引

仕様・設計・実験の用途別入口です。設計案と実装済み機能は各文書の状態を確認してください。

- [製品仕様](specifications/README.md)
- [利用・運用](guides/README.md)
- [設計](architecture/README.md)
- [実験・観測](experiments/README.md)
- [意思決定・Wiki](decisions/README.md)

## 整理と凍結契約

既存83ファイルのうち8ファイルを用途別フォルダへ移動し、75ファイルは元のパスとbytesを維持しています。

manifest内の各hash registryに67文書が直接登録されています。残る8文書はmanifestの文書参照と維持文書からの参照を保護するために元パスを維持しています。凍結本文の書換えや互換本文の複製を避けています。各カテゴリの「元パス維持」が該当します。[分類台帳](documentation-layout.json)で各文書の移動先、登録元、維持文書からの参照を確認できます。

fixture・manifest・観測証拠の内部パスは当時の識別子として維持します。古い実験の再実行や共有データベースへのアクセスは行いません。

リポジトリ直下のREADME.mdも現行の凍結hash検証対象のため、案内リンクを追記していません。GitHubでdocsフォルダを開くと、このREADMEが文書入口になります。

## Wiki

Home.md、_Footer.md、Decision-Structure.mdは元パスを維持しています。既存のHome.mdからのmain上のリンク先も維持しています。この索引はGitHubリポジトリ向けで、Wiki-onlyのページや同期設定は変更していません。
