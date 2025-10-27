import sqlglot
import sqlglot.expressions as exp
from typing import Any

class Query:
    expressions : list[exp.Expression]
    from_ : exp.From
    where : exp.Where
    joins : list[exp.Join]
    group : exp.Group

    aggregate_func : str
    aggregate_attr : str
    from_table : str
    predicates : list[tuple[str, str, str]]
    groupby_attrs : list[str]

    def __init__(self, sql):
        self.sql = sql
        self.parsed = sqlglot.parse_one(sql)
        assert isinstance(self.parsed, exp.Select), f"Only support Select SQL: {sql}"
        self.expressions = self.parsed.expressions  # list
        self.from_ = self.parsed.args.get('from')   # exp.From
        self.where = self.parsed.args.get('where')  # exp.Where
        self.joins = self.parsed.args.get('joins')  # list
        self.group = self.parsed.args.get('group')  # exp.Group

        self.aggregate_func = None
        self.aggregate_attr = None
        self.from_table = None
        self.join_table = None
        self.join_attrs = None
        self.predicates = None
        self.groupby_attrs = None

    def __str__(self):
        return self.sql
    
    def get_aggregate_func(self):
        if self.aggregate_func is None:
            self.aggregate_func = self.expressions[-1].__class__.__name__.upper()
        return self.aggregate_func
    
    def get_aggregate_attr(self):
        if self.aggregate_attr is None:
            self.aggregate_attr = self.expressions[-1].this.name.lower()
        return self.aggregate_attr
    
    def get_from_table(self):
        if self.from_table is None:
            if self.from_ is not None:
                self.from_table = self.from_.this.name.lower()
        return self.from_table
    
    def get_predicates(self) -> list[tuple[Any]]:
        if self.predicates is not None:
            return self.predicates
        
        self.predicates = []

        def add_predicate(predicate : exp.Predicate):
            self.predicates.append((predicate.this.name.lower(), predicate.__class__.__name__.upper(), predicate.expression.name))
        
        def process_and(_and : exp.And):
            if isinstance(_and.this, exp.And):
                process_and(_and.this)
            elif isinstance(_and.this, exp.Predicate):
                add_predicate(_and.this)
            add_predicate(_and.expression)
        
        if self.where is not None:
            if isinstance(self.where.this, exp.And):
                process_and(self.where.this)
            elif isinstance(self.where.this, exp.Predicate):
                add_predicate(self.where.this)
        
        return self.predicates
    
    def get_groupby_attrs(self):
        if self.groupby_attrs is None:
            self.groupby_attrs = []
            if self.group is not None:
                for expression in self.group.expressions:
                    self.groupby_attrs.append(expression.name.lower())
        return self.groupby_attrs
    
    def get_join_table(self):
        if self.join_table is not None:
            return self.join_table
        if self.joins is None:
            self.join_table = ""
        else:
            self.join_table = self.joins[0].this.name
        return self.join_table
    
    def get_join_attrs(self):
        if self.join_attrs is not None:
            return self.join_attrs
        
        if self.joins is None:
            self.join_attrs = ("", "")
        else:
            self.join_attrs = (self.joins[0].args['on'].expression.name, self.joins[0].args['on'].left.name)
        return self.join_attrs