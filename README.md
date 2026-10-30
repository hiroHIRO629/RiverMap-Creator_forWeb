# RiverMap-Creator for Web

国内の対象河川を検索し、Streamlit 上で軽量に描画・ダウンロードできるアプリです。  
水系/河川の範囲を folium で可視化し、GeoJSON と PNG 画像に出力できます。

## 主な機能

- 河川名の一部、水系名、都道府県名からの検索と、水系・都道府県による絞り込み
- 同名河川を、水系・所在地・河川コードを見ながら選択
- 水系全体／河川単体／重ね合わせの表示、背景地図の切り替え、複数地点の目印
- 地図に収録されているライン数・概算ライン長の確認
- PNG と GeoJSON のダウンロード

> 検索結果は最大200件です。候補が多い場合は、水系または都道府県で絞り込んでください。

## セットアップ

```bash
python -m venv .venv        # 既存環境があれば任意
source .venv/bin/activate
pip install -r requirements.txt
```

## データの用意

`data/all_river_light.parquet` はリポジトリに含まれている軽量版の河川ラインデータです。  
元の Shapefile から再生成したい場合は `scripts/prepare_data.py` を実行してください。

```bash
python scripts/prepare_data.py
```

> スクリプトは `RiverMap-Creator/riverline/all_river/all_river.shp` を読み込み、ジオメトリを簡略化して Parquet 形式に変換します。  
> 詳細解析が必要な場合は `simplify` のパラメータを調整してください。

## ローカル実行

```bash
streamlit run streamlit_app.py
```

## 公開について

- ベースマップに OpenStreetMap / CARTO / 地理院タイルを利用しています。表示や配布時はクレジット表記を残してください。
- Streamlit Community Cloud へデプロイする場合は、GitHub リポジトリを連携して `streamlit_app.py` を指定するだけで動作します。
