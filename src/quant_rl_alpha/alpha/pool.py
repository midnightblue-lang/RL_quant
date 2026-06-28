from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_rl_alpha.alpha.metrics import (
    AlphaCalculator,
    make_weighted_pool_values,
    mutual_ic,
)
from quant_rl_alpha.utils.config import load_config


@dataclass(frozen=True)
class AlphaEntry:
    name: str
    formula: str
    values: pd.DataFrame
    ic: float
    rank_ic: float
    tokens: tuple[str, ...] = ()


@dataclass(frozen=True)
class PoolUpdateResult:
    added: bool
    removed: str | None
    pool_ic: float
    pool_loss: float
    weights: dict[str, float]
    reason: str | None = None


@dataclass
class AlphaPool:
    labels: pd.DataFrame
    capacity: int = 30
    learning_rate: float = 5e-4
    max_steps: int = 10000
    tolerance: int = 500
    l1_alpha: float = 5e-3
    gradient_steps: int | None = None
    min_valid_dates: int = 1
    min_valid_ratio: float = 0.0
    min_stocks_per_date: int = 2
    entries: list[AlphaEntry] = field(default_factory=list)
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    mutual_cache: dict[tuple[str, str], float] = field(default_factory=dict)
    failure_cache: set[str] = field(default_factory=set)
    calculator: AlphaCalculator = field(init=False)

    def __post_init__(self) -> None:
        if self.gradient_steps is not None:
            self.max_steps = self.gradient_steps
        self.calculator = AlphaCalculator(self.labels)

    def add(
        self,
        name: str,
        formula: str,
        values: pd.DataFrame,
        *,
        tokens: tuple[str, ...] = (),
    ) -> PoolUpdateResult:
        if any(entry.name == name for entry in self.entries):
            raise ValueError(f"Alpha already exists in pool: {name}")
        if any(
            entry.formula == formula or (tokens and entry.tokens == tokens)
            for entry in self.entries
        ):
            return self._result(False, None, "duplicate_formula")
        cache_key = _failure_key(formula, tokens)
        if cache_key in self.failure_cache:
            return self._result(False, None, "failure_cache")

        aligned_values = align_values_to_labels(values, self.labels)
        if not self._has_enough_coverage(aligned_values):
            return self._result(False, None, "insufficient_coverage")

        ic, rank_ic = self.calculator.calc_single_all_ret(aligned_values)
        if not np.isfinite(ic):
            return self._result(False, None, "nonfinite_ic")
        entry = AlphaEntry(
            name=name,
            formula=formula,
            values=aligned_values,
            ic=ic,
            rank_ic=rank_ic,
            tokens=tokens,
        )
        mutual_values = self._candidate_mutual_ics(entry)
        if mutual_values is None:
            self.failure_cache.add(cache_key)
            return self._result(False, None, "nonfinite_mutual_ic")

        candidate_entries = [*self.entries, entry]
        candidate_weights = np.append(self.weights, _initial_weight(ic, self.weights))
        candidate_mutual = self._extended_mutual_matrix(mutual_values)
        new_weights = self._optimize(candidate_entries, candidate_weights, candidate_mutual)
        candidate_loss = pool_loss(
            _clean_weights(new_weights),
            _ic_vector(candidate_entries),
            candidate_mutual,
        )
        if not np.isfinite(candidate_loss):
            self.failure_cache.add(cache_key)
            return self._result(False, None, "nonfinite_weights")

        removed = None
        final_entries = candidate_entries
        final_weights = new_weights
        final_mutual = candidate_mutual
        if len(candidate_entries) > self.capacity:
            remove_index = int(np.argmin(np.abs(new_weights)))
            if remove_index == len(candidate_entries) - 1:
                self.failure_cache.add(cache_key)
                return self._result(False, None, "new_alpha_pruned")
            removed = candidate_entries[remove_index].name
            final_entries = [
                item for index, item in enumerate(candidate_entries) if index != remove_index
            ]
            final_weights = np.delete(new_weights, remove_index)
            final_mutual = np.delete(
                np.delete(candidate_mutual, remove_index, axis=0),
                remove_index,
                axis=1,
            )

        self.entries = final_entries
        self.weights = _clean_weights(final_weights)
        self._refresh_mutual_cache(final_mutual)
        self.failure_cache.clear()
        return self._result(True, removed)

    def optimize_weights(self) -> np.ndarray:
        if not self.entries:
            self.weights = np.zeros(0, dtype=float)
            return self.weights
        self.weights = self._optimize(self.entries, self.weights, self.mutual_ic_matrix())
        return self.weights

    def ic_vector(self) -> np.ndarray:
        return _ic_vector(self.entries)

    def mutual_ic_matrix(self) -> np.ndarray:
        size = len(self.entries)
        matrix = np.eye(size, dtype=float)
        for row in range(size):
            for col in range(row + 1, size):
                key = _cache_key(self.entries[row].name, self.entries[col].name)
                if key not in self.mutual_cache:
                    self.mutual_cache[key] = mutual_ic(
                        self.entries[row].values,
                        self.entries[col].values,
                    )
                value = self.mutual_cache[key]
                matrix[row, col] = value
                matrix[col, row] = value
        return matrix

    def pool_loss(self) -> float:
        if not self.entries:
            return float("nan")
        ic_vector = self.ic_vector()
        mutual = self.mutual_ic_matrix()
        return pool_loss(self.weights, ic_vector, mutual)

    def pool_values(self) -> pd.DataFrame:
        if not self.entries:
            return pd.DataFrame(columns=["date", "symbol", "value"])
        return make_weighted_pool_values([entry.values for entry in self.entries], self.weights)

    def pool_ic(self) -> float:
        if not self.entries:
            return float("nan")
        return self.calculator.calc_pool_IC_ret(
            [entry.values for entry in self.entries],
            self.weights,
        )

    def weight_map(self) -> dict[str, float]:
        return {
            entry.name: float(weight)
            for entry, weight in zip(self.entries, self.weights, strict=True)
        }

    def summary_frame(self) -> pd.DataFrame:
        columns = ["name", "formula", "tokens", "ic", "rank_ic", "weight"]
        rows = [
            {
                "name": entry.name,
                "formula": entry.formula,
                "tokens": " ".join(entry.tokens),
                "ic": entry.ic,
                "rank_ic": entry.rank_ic,
                "weight": float(weight),
            }
            for entry, weight in zip(self.entries, self.weights, strict=True)
        ]
        return pd.DataFrame(rows, columns=columns)

    def _has_enough_coverage(self, values: pd.DataFrame) -> bool:
        valid = values["value"].notna()
        if len(values) == 0 or valid.mean() < self.min_valid_ratio:
            return False
        daily_counts = values.loc[valid].groupby("date")["symbol"].count()
        return bool((daily_counts >= self.min_stocks_per_date).sum() >= self.min_valid_dates)

    def _candidate_mutual_ics(self, candidate: AlphaEntry) -> list[float] | None:
        values = []
        for entry in self.entries:
            value = self.calculator.calc_mutual_IC(candidate.values, entry.values)
            if not np.isfinite(value):
                return None
            values.append(float(value))
        return values

    def _extended_mutual_matrix(self, candidate_mutual: list[float]) -> np.ndarray:
        size = len(self.entries) + 1
        matrix = np.eye(size, dtype=float)
        if self.entries:
            matrix[:-1, :-1] = self.mutual_ic_matrix()
            matrix[:-1, -1] = candidate_mutual
            matrix[-1, :-1] = candidate_mutual
        return matrix

    def _optimize(
        self,
        entries: list[AlphaEntry],
        initial_weights: np.ndarray,
        mutual_matrix: np.ndarray,
    ) -> np.ndarray:
        return optimize_weights(
            _ic_vector(entries),
            mutual_matrix,
            initial_weights=initial_weights,
            learning_rate=self.learning_rate,
            max_steps=self.max_steps,
            tolerance=self.tolerance,
            l1_alpha=self.l1_alpha,
        )

    def _refresh_mutual_cache(self, mutual_matrix: np.ndarray) -> None:
        self.mutual_cache = {}
        for row in range(len(self.entries)):
            for col in range(row + 1, len(self.entries)):
                key = _cache_key(self.entries[row].name, self.entries[col].name)
                self.mutual_cache[key] = float(mutual_matrix[row, col])

    def _result(
        self,
        added: bool,
        removed: str | None,
        reason: str | None = None,
    ) -> PoolUpdateResult:
        return PoolUpdateResult(
            added=added,
            removed=removed,
            pool_ic=self.pool_ic(),
            pool_loss=self.pool_loss(),
            weights=self.weight_map(),
            reason=reason,
        )


def alpha_pool_from_config(labels: pd.DataFrame) -> AlphaPool:
    config = load_config("alpha")
    return AlphaPool(
        labels=labels,
        capacity=int(config["pool_size"]),
        learning_rate=float(config["learning_rate"]),
        max_steps=int(config["max_steps"]),
        tolerance=int(config["tolerance"]),
        l1_alpha=float(config["l1_alpha"]),
        min_valid_dates=int(config["min_valid_dates"]),
        min_valid_ratio=float(config["min_valid_ratio"]),
        min_stocks_per_date=int(config["min_stocks_per_date"]),
    )


def pool_loss(weights: np.ndarray, ic_vector: np.ndarray, mutual_ic_matrix: np.ndarray) -> float:
    if len(weights) == 0:
        return float("nan")
    if (
        len(weights) != len(ic_vector)
        or mutual_ic_matrix.shape != (len(weights), len(weights))
        or not np.isfinite(weights).all()
        or not np.isfinite(ic_vector).all()
        or not np.isfinite(mutual_ic_matrix).all()
    ):
        return float("nan")
    return float(1 - 2 * weights @ ic_vector + weights @ mutual_ic_matrix @ weights)


def _cache_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def align_values_to_labels(values: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    keys = _sample_keys(labels)
    frame = values.loc[:, ["date", "symbol", "value"]].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["value"] = frame["value"].replace([np.inf, -np.inf], np.nan)
    if frame[["date", "symbol"]].duplicated().any():
        raise ValueError("Alpha values contain duplicate (date, symbol) rows")
    return keys.merge(frame, on=["date", "symbol"], how="left")


def _sample_keys(labels: pd.DataFrame) -> pd.DataFrame:
    keys = labels.loc[:, ["date", "symbol"]].copy()
    keys["date"] = pd.to_datetime(keys["date"]).dt.normalize()
    keys["symbol"] = keys["symbol"].astype(str).str.zfill(6)
    keys = keys.sort_values(["date", "symbol"]).reset_index(drop=True)
    if keys.duplicated().any():
        raise ValueError("Labels contain duplicate (date, symbol) rows")
    return keys


def optimize_weights(
    ic_vector: np.ndarray,
    mutual_ic_matrix: np.ndarray,
    *,
    initial_weights: np.ndarray | None = None,
    learning_rate: float = 5e-4,
    max_steps: int = 10000,
    tolerance: int = 500,
    l1_alpha: float = 5e-3,
    gradient_steps: int | None = None,
) -> np.ndarray:
    if gradient_steps is not None:
        max_steps = gradient_steps
    weights = (
        np.zeros(len(ic_vector), dtype=float)
        if initial_weights is None
        else initial_weights.astype(float).copy()
    )
    if len(weights) == 0:
        return weights
    if len(weights) != len(ic_vector) or mutual_ic_matrix.shape != (len(weights), len(weights)):
        raise ValueError("Invalid alpha pool optimization shapes")
    if (
        not np.isfinite(weights).all()
        or not np.isfinite(ic_vector).all()
        or not np.isfinite(mutual_ic_matrix).all()
    ):
        return np.full(len(weights), np.nan)
    if max_steps <= 0:
        return weights
    if np.isclose(l1_alpha, 0.0):
        try:
            return np.linalg.lstsq(mutual_ic_matrix, ic_vector, rcond=None)[0]
        except (np.linalg.LinAlgError, ValueError):
            return weights

    best = weights.copy()
    best_loss = pool_loss(best, ic_vector, mutual_ic_matrix)
    if not np.isfinite(best_loss):
        return np.full(len(weights), np.nan)
    first_moment = np.zeros_like(weights)
    second_moment = np.zeros_like(weights)
    beta1, beta2 = 0.9, 0.999
    tolerance_count = 0

    for step in range(1, max_steps + 1):
        gradient = -2 * ic_vector + 2 * mutual_ic_matrix @ weights + l1_alpha * np.sign(weights)
        if not np.isfinite(gradient).all():
            break
        first_moment = beta1 * first_moment + (1 - beta1) * gradient
        second_moment = beta2 * second_moment + (1 - beta2) * gradient * gradient
        update = (first_moment / (1 - beta1**step)) / (
            np.sqrt(second_moment / (1 - beta2**step)) + 1e-8
        )
        weights = weights - learning_rate * update

        loss_ic = pool_loss(weights, ic_vector, mutual_ic_matrix)
        if not np.isfinite(loss_ic):
            break
        if best_loss - loss_ic > 1e-6:
            tolerance_count = 0
        else:
            tolerance_count += 1
        if loss_ic < best_loss:
            best = weights.copy()
            best_loss = loss_ic
        if tolerance_count >= tolerance:
            break
    return best


def _initial_weight(ic: float, weights: np.ndarray) -> float:
    return float(max(ic, 0.01)) if len(weights) == 0 else float(weights.mean())


def _ic_vector(entries: list[AlphaEntry]) -> np.ndarray:
    return np.array([entry.ic for entry in entries], dtype=float)


def _clean_weights(weights: np.ndarray) -> np.ndarray:
    if np.isfinite(weights).all():
        return weights.astype(float).copy()
    return np.full(len(weights), np.nan)


def _failure_key(formula: str, tokens: tuple[str, ...]) -> str:
    return " ".join(tokens) if tokens else formula
