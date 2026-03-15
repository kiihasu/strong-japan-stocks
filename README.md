# 強い日本株スクリーナー

## 条件

### 成長トレンド条件
- 1年騰落率 +30%以上
- 200日移動平均線より上

### ファンダ条件
- 売上成長率 +10%以上
- 営業利益率 上昇トレンド

## データソース
- JPX 上場銘柄一覧
- Yahoo Finance (`yfinance`)

## 使い方

```bash
pip install -r requirements.txt
python screen_japan_stocks.py
