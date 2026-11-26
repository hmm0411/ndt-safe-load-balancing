from __future__ import annotations

import csv
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Literal

from src.experiments.workloads.scale import scale_for


RangePolicy = Literal["reject", "clamp"]


class CapacityWorkloadMapper:
	def __init__(
		self,
		capacity_runs_path: str | Path,
		controller_capacity_path: str | Path,
		range_policy: RangePolicy = "reject",
	) -> None:
		if range_policy not in ("reject", "clamp"):
			raise ValueError(
				"range_policy must be 'reject' or 'clamp'"
			)

		self._range_policy = range_policy
		self._safe_capacity = self._load_safe_capacity(
			Path(controller_capacity_path)
		)
		self._benchmarks = self._load_benchmarks(
			Path(capacity_runs_path)
		)

	@staticmethod
	def _load_safe_capacity(path: Path) -> dict[str, float]:
		try:
			payload = json.loads(path.read_text(encoding="utf-8"))
		except FileNotFoundError as exc:
			raise ValueError(f"capacity file does not exist: {path}") from exc

		controllers = payload.get("controllers")
		if not isinstance(controllers, dict):
			raise ValueError("capacity JSON must contain a controllers object")

		result: dict[str, float] = {}
		for controller_id, config in controllers.items():
			if not isinstance(config, dict) or "c_safe_pps" not in config:
				raise ValueError(
					f"missing c_safe_pps for controller {controller_id}"
				)
			safe_capacity = float(config["c_safe_pps"])
			if not math.isfinite(safe_capacity) or safe_capacity <= 0:
				raise ValueError(
					f"invalid c_safe_pps for controller {controller_id}"
				)
			result[str(controller_id)] = safe_capacity

		return result

	@staticmethod
	def _load_benchmarks(
		path: Path,
	) -> dict[str, tuple[tuple[float, float], ...]]:
		try:
			with path.open(newline="", encoding="utf-8") as handle:
				rows = csv.DictReader(handle)

				# Store as:
				# controller -> [(offered_rate, processed_rate), ...]
				raw_points: dict[str, list[tuple[float, float]]] = defaultdict(list)

				for row in rows:
					if not CapacityWorkloadMapper._is_valid_row(row):
						continue

					controller_id = row.get("controller") or row.get(
						"controller_id"
					)
					if not controller_id:
						raise ValueError("benchmark row is missing controller")

					try:
						offered_rate = float(
							row["target_new_flow_rate"]
						)
						processed_rate = float(
							row["processed_packet_in_rate_mean"]
						)
					except (KeyError, TypeError, ValueError) as exc:
						raise ValueError(
							"benchmark rows must contain numeric "
							"target_new_flow_rate and "
							"processed_packet_in_rate_mean"
						) from exc

					if (
						not math.isfinite(offered_rate)
						or offered_rate < 0
						or not math.isfinite(processed_rate)
						or processed_rate < 0
					):
						raise ValueError(
							"benchmark rates must be finite "
							"and non-negative"
						)

					raw_points[str(controller_id)].append(
						(offered_rate, processed_rate)
					)
		except FileNotFoundError as exc:
			raise ValueError(f"benchmark file does not exist: {path}") from exc

		result: dict[
			str,
			tuple[tuple[float, float], ...],
		] = {}

		for controller_id, values in raw_points.items():
			if not values:
				continue

			# Group repeated measurements by offered rate before checking
			# monotonicity along the increasing offered-load axis.
			grouped_by_offered: dict[float, list[float]] = defaultdict(list)
			for offered_rate, processed_rate in values:
				grouped_by_offered[offered_rate].append(processed_rate)

			averaged_points = [
				(
					offered_rate,
					sum(processed_rates) / len(processed_rates),
				)
				for offered_rate, processed_rates in grouped_by_offered.items()
			]
			averaged_points.sort(key=lambda point: point[0])

			monotonic_points: list[tuple[float, float]] = []
			previous_processed: float | None = None

			for offered_rate, processed_rate in averaged_points:
				if previous_processed is not None and processed_rate <= previous_processed:
					warnings.warn(
						(
							f"non-monotonic benchmark detected for controller "
							f"{controller_id}: processed rate changed from "
							f"{previous_processed:.3f} to {processed_rate:.3f} pkt/s "
							f"at offered rate {offered_rate:.3f} flows/s; "
							"discarding this point and all higher-load observations "
							"from inverse interpolation"
						),
						RuntimeWarning,
						stacklevel=2,
					)
					break

				monotonic_points.append((offered_rate, processed_rate))
				previous_processed = processed_rate

			if len(monotonic_points) < 2:
				raise ValueError(
					f"controller {controller_id} does not have enough "
					"monotonic benchmark observations"
				)

			result[controller_id] = tuple(
				(processed_rate, offered_rate)
				for offered_rate, processed_rate in monotonic_points
			)

		if not result:
			raise ValueError("benchmark file contains no valid observations")
		return result

	@staticmethod
	def _is_valid_row(row: dict[str, str]) -> bool:
		valid = row.get("valid")
		return valid is None or valid.strip().lower() in {"true", "1", "yes"}

	def offered_rate_for_utilization(
		self,
		controller_id: str,
		target_utilization: float,
	) -> float:
		try:
			utilization = float(target_utilization)
		except (TypeError, ValueError) as exc:
			raise ValueError("target_utilization must be numeric") from exc
		# campaign load scale (NDT_WORKLOAD_SCALE, scale.py); 1.0 when unset
		utilization *= scale_for(controller_id)

        # workload burst need U = 1.10 -- U = 1.05 (based on real workload) -- U = 1.0 (based on real workload) -- U = 0.95 (based on real workload)
		if not math.isfinite(utilization) or utilization < 0.0:
			raise ValueError("target_utilization must be a finite non-negative number")

		try:
			safe_capacity = self._safe_capacity[controller_id]
		except KeyError as exc:
			raise ValueError(f"unknown controller: {controller_id}") from exc
		try:
			points = self._benchmarks[controller_id]
		except KeyError as exc:
			raise ValueError(
				f"no benchmark observations for controller: {controller_id}"
			) from exc

		target_processed_rate = utilization * safe_capacity
		minimum_processed_rate = points[0][0]
		maximum_processed_rate = points[-1][0]
		if not minimum_processed_rate <= target_processed_rate <= maximum_processed_rate:
			if self._range_policy == "reject":
				raise ValueError(
					f"target processed rate {target_processed_rate} is outside "
					f"observed benchmark range "
					f"[{minimum_processed_rate}, {maximum_processed_rate}]"
				)

			clamped_rate = min(
				maximum_processed_rate,
				max(minimum_processed_rate, target_processed_rate),
			)
			warnings.warn(
				f"target processed rate {target_processed_rate} is outside "
				f"observed benchmark range; clamping to {clamped_rate}",
				RuntimeWarning,
				stacklevel=2,
			)
			target_processed_rate = clamped_rate

		for (left_processed, left_offered), (
			right_processed,
			right_offered,
		) in zip(points, points[1:]):
			if target_processed_rate <= right_processed:
				fraction = (target_processed_rate - left_processed) / (
					right_processed - left_processed
				)
				return left_offered + fraction * (right_offered - left_offered)

		return points[-1][1]
