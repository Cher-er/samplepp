from dataset.dataset import Dataset, fact
from dataset.table import Table


@fact(['flights'])
class Flights20M(Dataset):
    db_name = 'flights_20m'
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
            rows=15465984,
            sample_rate=1.0,
        )
    }
    best_state_paths = {
        "flights": "gen_model/vaeac/best_states/flights_20m/flights.pkl"
    }