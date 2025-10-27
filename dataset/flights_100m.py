from dataset.dataset import Dataset, fact
from dataset.table import Table


@fact(['flights'])
class Flights100M(Dataset):
    db_name = 'flights_100m'
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
            rows=77329922,
            sample_rate=0.2,
        )
    }
    best_state_paths = {
        "flights": "gen_model/vaeac/best_states/flights_100m/flights.pkl"
    }
