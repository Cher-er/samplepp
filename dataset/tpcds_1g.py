from dataset.dataset import Dataset, fact
from dataset.table import Table


@fact(["store_sales"])
class Tpcds1g(Dataset):
    db_name = 'tpcds_1g'
    tables = {
        "store": Table(
            schema={
                "s_store_sk": "int",
                "s_store_name": "str",
                "s_number_employees": "int",
                "s_floor_space": "int",
                "s_gmt_offset": "num",
                "s_tax_percentage": "num",
            },
            rows=12,
        ),
        "store_sales": Table(
            schema={
                "ss_store_sk": "int",
                "ss_quantity": "int",
                "ss_wholesale_cost": "num",
                "ss_list_price": "num",
                "ss_sales_price": "num"
            },
            rows=2628627,
            sample_rate=1.0
        )
    }
    best_state_paths = {
        "store_sales": "gen_model/vaeac/best_states/tpcds_1g/store_sales.pkl"
    }