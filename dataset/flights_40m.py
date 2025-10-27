from dataset.dataset import Dataset, fact
from dataset.table import Table


@fact(['flights'])
class Flights40M(Dataset):
    db_name = 'flights_40m'
    tables = {
        "flights": Table(
            schema={
                "quarter": "int",
                "month": "int",
                "dayofmonth": "int",
                "dayofweek": "int",
                "reporting_airline": "str",
                "origin": "str",
                "originstatename": "str",
                "dest": "str",
                "deststatename": "str",
                "depdelay": "num",
                "taxiout": "num",
                "arrdelay": "num",
                "taxiin": "num",
                "airtime": "num",
                "distance": "num"
            },
            rows=30931969,
            sample_rate=0.5,
        )
    }
    best_state_paths = {
        "flights": "gen_model/vaeac/best_states/flights_40m/flights.pkl"
    }