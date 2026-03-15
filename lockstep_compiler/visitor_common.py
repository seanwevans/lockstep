from dataclasses import dataclass
from typing import Any

from .models import SemanticSymbol


SEMANTIC_DIAGNOSTIC_CODES = {
    "undefined_identifier": "LCK301",
    "invalid_field_access_non_struct": "LCK302",
    "invalid_field_access_unknown_field": "LCK302",
    "bind_argument_count_mismatch": "LCK303",
    "bind_unknown_target": "LCK304",
    "bind_type_mismatch": "LCK305",
    "duplicate_declaration": "LCK306",
    "duplicate_kernel_declaration": "LCK307",
    "bind_modifier_mismatch": "LCK308",
    "bind_output_target_kind_mismatch": "LCK309",
    "unknown_declared_type": "LCK310",
    "duplicate_struct_field": "LCK311",
    "bind_output_symbol_mismatch": "LCK312",
    "unknown_fold_operator": "LCK401",
    "fold_type_mismatch": "LCK402",
    "fold_unknown_source": "LCK403",
    "fold_unknown_target": "LCK404",
    "pure_unknown_function": "LCK410",
    "pure_argument_count_mismatch": "LCK411",
    "pure_argument_type_mismatch": "LCK412",
    "pure_missing_return": "LCK413",
    "pure_multiple_returns": "LCK414",
    "pure_unreachable_after_return": "LCK415",
    "var_initializer_type_mismatch": "LCK416",
    "assignment_type_mismatch": "LCK417",
    "pure_return_type_mismatch": "LCK418",
    "uniform_initializer_type_mismatch": "LCK419",
    "invalid_operand_types": "LCK420",
    "unused_symbol": "LCK421",
    "unbound_pipeline_resource": "LCK422",
    "cannot_infer_type": "LCK423",
    "implicit_numeric_widening": "LCK424",
    "use_before_definition": "LCK425",
}


@dataclass
class ScopedSymbolData:
    symbol: SemanticSymbol
    usage_count: int
    declaration_ctx: Any
    is_assigned: bool


class Scope:
    def __init__(self):
        self.symbols: dict[str, ScopedSymbolData] = {}

    def __contains__(self, name: str) -> bool:
        return name in self.symbols

    def __getitem__(self, name: str) -> SemanticSymbol:
        return self.symbols[name].symbol
