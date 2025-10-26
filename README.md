# RiverMap-Creator for Web

国内の対象河川を検索し、Streamlit 上で軽量に描画・ダウンロードできるアプリです。  
水系/河川の範囲を folium で可視化し、GeoJSON と PNG 画像に出力できます。

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

