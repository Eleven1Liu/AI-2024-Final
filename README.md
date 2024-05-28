# AI Term Project

## Collect Data

```bash
python download_data.py -t 300 -d data 
```

- `-d`: Path to the data directory.
- `-t`: Time interval for fetching server.


## Processed Data
|                         |  Time Interval  |        Area        |  Size (per site) |  Download Link                                                                        |
|:------------------------|----------------:|-------------------:|-----------------:|--------------------------------------------------------------------------------------:|
| 2024-05-06 ~ 2024-05-12 |      2 mins     |  大安區 (180 sites) |       4,362 (with data missing in 5/8 caused by socket error)  | https://drive.google.com/file/d/1dRaOBnsf8pavtQtFlDFrM--o09W6CLiZ/view?usp=drive_link |  
|                         |                 |  臺大公館校區 (53 sites) |         | https://drive.google.com/file/d/1lrGFXNFG1jmY7vwoJPdiETVF7AVcC9WJ/view?usp=drive_link | 
| 2024-05-13 ~ 2024-05-19 |      2 mins     |  大安區 (180 sites)     |       5,004      | https://drive.google.com/file/d/10ZNX_Fs73Dod9WoSnm6XonltMY5dtFV-/view?usp=drive_link |
|                         |                 |  臺大公館校區 (53 sites) |         | https://drive.google.com/file/d/1ylBo5_aS9EmUx_yV5hs1UjLUyWRyxfus/view?usp=drive_link |
| 2024-05-20 ~ 2024-05-26 |      2 mins     |  大安區 (180 sites)     |       5,018      | https://drive.google.com/file/d/18I001qKLku85ilSZ6R9gjLZht5crGepQ/view?usp=drive_link |
|                         |                 |  臺大公館校區 (53 sites) |                  | https://drive.google.com/file/d/1qxRbZD17JWFOUw-UOalGovETPAWH45wV/view?usp=drive_link |

> error files: 
>   20240510232222.csv,
>   20240515140202.csv, 
>   20240520155837.csv, 20240525032928.csv, 20240525033128.csv, 20240525042337.csv, 20240525044540.csv, 20240525054752.csv

## Meta Data

| File Name                            | Primary Key    | Description                         |
|:-------------------------------------|---------------:|------------------------------------:|
| metadata/taipei_mrt_info_utf8.csv    | [臺北捷運車站出入口座標](https://data.taipei/dataset/detail?id=cfa4778c-62c1-497b-b704-756231de348b) UTF8 版本 |
| metadata/mrt_ubike_shortest_dist.csv | mrt_shortname  | 大安區,臺大公館校區 ubike (180+53=233) 到最近 MRT 站點距離 (公尺) | 
| metadata/april_mrt_traffic.csv       | mrt_shortname  | 2024/04 MRT 分時資料 (12am-1am, 5am-11pm)，可用 `mrt_shortname` join `metadata/mrt_ubike_shortest_dist.csv`, 包含近大安區和臺大公館校區的ubike 站點的捷運站 (古亭, 東門, 台電大樓, 公館, 忠孝新生, 大安森林公園, 忠孝復興, 科技大樓, 大安, 萬隆, 忠孝敦化, 六張犁, 信義安和, 國父紀念館, 麟光) | 


### Columns
|                        |   NA count |    column names  |
|:-----------------------|-----------:|-----------------:|
| sno                    |         0  | 站點代號           |
| sna                    |         0  | 場站中文名稱       |
| sarea                  |         0  | 場站區域 (大同區,信義區,**大安區**,萬華區,中正區,中山區,文山區,北投區,松山區,南港區,**臺大公館校區**,士林區,內湖區) |
| mday                   |         0  | 資料更新時間       |
| ar                     |         0  | 地點 (地址)       |
| sareaen                |         0  | 場站區域英文       |
| snaen                  |         0  | 場站名稱英文       |
| aren                   |         0  | 地址英文          |
| act                    |         0  | 全站禁用狀態       |
| srcUpdateTime          |         0  | YouBike2.0系統發布資料更新的時間 |
| updateTime             |         0  | 大數據平台經過處理後將資料存入DB的時間 |
| infoTime               |         0  | 各場站來源資料更新時間 |
| infoDate               |         0  | 各場站來源資料更新時間 |
| total                  |    423900  | 場站總停車格       |
| available_rent_bikes   |    423900  | 場站目前車輛數量    |
| latitude               |    423900  | 緯度              |
| longitude              |    423900  | 經度              |
| available_return_bikes |    423900  | 空位數量          |

## Features v1
- Time: 2024-05-06 ~ 2024-05-27;共21天
- Interval: 2 mins 
- code: features.ipynb https://colab.research.google.com/drive/1grd-f810UtMokbN3jeeWvhyV0BrnLXXJ?usp=drive_link
- google drive: https://drive.google.com/drive/folders/1QVDKGHayxGpnZkQ-VlLFS8HfGR9TO34U?usp=sharing

| Files                          | Description                         |
|:-------------------------------------|------------------------------------:|
| feature_500101001_0.4        | 站點500101001的特徵資料，方便查看| 
| feature_0.2_2mins.csv  | sample rate為0.2的資料|
| feature_0.4_2mins.csv  | sample rate為0.4的資料|
| feature_1_nontomperal.csv        | sample rate為1，不含時序特徵的資料| 

- [0528] 更新公館大學區資料 + 0520~0527資料


## Features v2
- Time: 5/3-5/24; 共16天
- Interval: 5 mins
- Data source: https://drive.google.com/file/d/1vDmGQkM6EJU3T5yLD8CKOcmx0WyjF1kZ/view?usp=sharing
- Code: features_v2.ipynb
- Features: https://drive.google.com/drive/folders/1QVDKGHayxGpnZkQ-VlLFS8HfGR9TO34U?usp=drive_link 

| Files                          | Description                         |
|:-------------------------------------|------------------------------------:|
| feature_500101001_1_5mins.csv       | 站點500101001的特徵資料，方便查看| 
| feature_1_5mins.csv  | sample rate為1的資料|

## Features Description
| columes                          | Description                         |
|:-------------------------------------|------------------------------------:|
| date_value  | 日期相關number. eg.506|
| time        | 時間相關number. eg.1101 | 
| week        | week=1 週中； week=0 週末 | 
| popularity        | 十大熱門站位10-1，其餘0| 
| rainfall        | 當日雨量| 
| see_rate_value        | 見車率1,2,3；見車率未覆蓋站點-1| 
| mrt_distance        | 到最近捷運距離| 
| sbi_onehour        | 歷史前一小時數據，2mins為間隔，共30 points，缺失值標記為-1| 
| sbi_history        | 歷史5天前後兩小時數據，2mins為間隔，(0-5)*60，(0-5取決於是否有歷史數據)| 
| sbi_onehour        | 歷史前一小時數據，2mins為間隔，共30 points，缺失值標記為-1| 
| sbi_prediction        | 對當前時間之後一小時的預測（不含當前時間），2mins為間隔，共30 points| 
| sbi_history_mask        | 標記是否為缺失值的mask(transformer會用到)，True為缺失值(-1)False為正常值| 
| sbi_prediction_mask        | 標記是否為缺失值的mask(transformer會用到)，True為缺失值(-1)False為正常值| 

## Model
- Temporal CNN:
   - 架構：t-cnn + fc
   - 輸入1h序列即其餘features輸出後1h預測；
  




