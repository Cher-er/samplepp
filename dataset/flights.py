from dataset.dataset import Dataset, fact
from dataset.table import Table


@fact(['flights'])
class Flights(Dataset):
    db_name = 'flights'
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
            rows=6532011,
            sample_rate=1.0,
        )
    }
    best_state_paths = {
        "flights": "gen_model/vaeac/best_states/flights/flights.pkl"
    }