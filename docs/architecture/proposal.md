# 新アーキテクチャ案

[設計索引](README.md) / [文書索引](../README.md)

[アーキテクチャHTML](ngr-architecture.html)は、Python の共有ローカル MCP と Windows 用 onedir / per-user インストーラーの実装範囲、未実装の統合初期設定とクラウド連携を区別する設計のひな型です。配布物の契約は [Windows package 仕様](../specifications/windows-package.md)を参照します。

GitHubのファイル画面ではHTMLをそのままWebページとして閲覧できません。リンク先のファイルをダウンロードして、ローカルのブラウザで開いてください。この追加ではGitHub Pages等への公開は行っていません。

HTMLの「活性化の計算規則は未決定」は、新アーキテクチャで採用する規則の話です。既存NGRには活性化・伝播の実装があります。

ローカルメモリの意味検索など、HTMLで未決定とされている事項は未決定のままです。共有ローカル MCP の導入手順と実装範囲は[ガイド](../guides/shared-local-mcp.md)と[仕様](../specifications/shared-local-mcp.md)を参照してください。
