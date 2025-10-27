# Sample++

1. create table (use Flights as example) and import data

```sql
CREATE TABLE flights (
    quarter              INTEGER,
    month                INTEGER,
    dayofmonth           INTEGER,
    dayofweek            INTEGER,
    reporting_airline    TEXT,
    origin               TEXT,
    originstatename      TEXT,
    dest                 TEXT,
    deststatename        TEXT,
    depdelay             REAL,
    taxiout              REAL,
    arrdelay             REAL,
    taxiin               REAL,
    airtime              REAL,
    distance             REAL,
    id                   BIGINT NOT NULL PRIMARY KEY
);
```

2. Configure Python environment

```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Configure database connection parameters in `application.yml`

```yml
database:
  user: postgres
  host: localhost
  port: 5432
```

4. Run the main program

```shell
python main.py --dataset flights --agg avg --workload_names avg --workload_paths sqls/flights/avg.sql --roaring_bitmap --sample_rate 0.1 --sampling_size 100 --max_iter 30
```