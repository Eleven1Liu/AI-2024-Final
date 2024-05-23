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
| 2024-05-13 ~ 2024-05-19 |      2 mins     |  大安區 (180 sites) |       5,004      | https://drive.google.com/file/d/10ZNX_Fs73Dod9WoSnm6XonltMY5dtFV-/view?usp=drive_link |

### Columns
|                        |   NA count |    column names  |
|:-----------------------|-----------:|-----------------:|
| sno                    |         0  | 站點代號           |
| sna                    |         0  | 場站中文名稱       |
| sarea                  |         0  | 場站區域 (大同區,信義區,**大安區**,萬華區,中正區,中山區,文山區,北投區,松山區,南港區,臺大公館校區,士林區,內湖區) |
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
