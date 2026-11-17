"""Lossless, fixed-order encoding of the source GGS wing as one 125-D vector.

Lengths retain the source JSON's units and angles remain in degrees. Root fields
occupy 47 coordinates; each of the three subsequent source sections occupies 26.
Shared root airfoils/chords are not duplicated. These offsets describe serialization
only: the FM network operates on the entire vector, without section tokens or poses.
No VAE, local-frame conversion, log transform, or reference-chord scaling is used.
"""

from math import cos, isfinite, radians
from typing import TypedDict, cast


SectionValues = dict[str, float | list[float]]


class WingGeometry(TypedDict):
    init: SectionValues
    mid: list[SectionValues]
    winglet: None


def require_mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{location}: expected an object with string keys.")
    return cast(dict[str, object], value)


def require_list(value: object, location: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{location}: expected a list.")
    return cast(list[object], value)


def require_number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{location}: expected a finite number, got {value!r}.")
    return float(value)


def geometry_layout() -> list[tuple[int, str, int]]:
    """Return (source section, field, coefficient); -1 denotes a scalar field."""
    fields: list[tuple[int, str, int]] = []
    for section in range(4):
        scalars = ("span", "chord_root", "chord_tip", "twist_root", "twist_tip", "sweep", "dihedral") if section == 0 else (
            "span", "chord_tip", "twist_tip", "sweep", "dihedral", "smooth_sector")
        airfoils = ("upper_cst_params_root", "lower_cst_params_root", "upper_cst_params_tip", "lower_cst_params_tip") if section == 0 else (
            "upper_cst_params_tip", "lower_cst_params_tip")
        fields.extend((section, field, -1) for field in scalars)
        fields.extend((section, field, coefficient) for field in airfoils for coefficient in range(10))
    return fields


def feature_names() -> list[str]:
    return [f"{'init' if section == 0 else f'mid[{section - 1}]'}.{field}"
            + (f"[{coefficient}]" if coefficient >= 0 else "")
            for section, field, coefficient in geometry_layout()]


def encode_wing(value: object, location: str) -> list[float]:
    wing = require_mapping(value, location)
    if wing.get("winglet") is not None:
        raise ValueError(f"{location}: this schema requires four sections and no separate winglet.")
    middle = require_list(wing["mid"], f"{location}.mid")
    if len(middle) != 3:
        raise ValueError(f"{location}: expected three mid sections, got {len(middle)}.")
    sections = [require_mapping(item, f"{location}.section[{index}]")
                for index, item in enumerate([wing["init"], *middle])]
    vector: list[float] = []
    for section, field, coefficient in geometry_layout():
        label = f"{location}.section[{section}].{field}"
        item = sections[section][field]
        if coefficient >= 0:
            values = require_list(item, label)
            if len(values) != 10:
                raise ValueError(f"{label}: expected 10 CST coefficients, got {len(values)}.")
            item = values[coefficient]
        number = require_number(item, label)
        if field in ("span", "chord_root", "chord_tip") and number <= 0:
            raise ValueError(f"{label}: physical length must be positive, got {number}.")
        vector.append(number)
    return vector


def decode_wing(vector: list[float]) -> WingGeometry:
    """Restore raw source geometry directly; generated invalid lengths fail explicitly."""
    layout = geometry_layout()
    if len(vector) != len(layout):
        raise ValueError(f"Expected {len(layout)} geometry coordinates, got {len(vector)}.")
    sections: list[SectionValues] = [{}, {}, {}, {}]
    for (section, field, coefficient), value in zip(layout, vector):
        number = require_number(value, f"vector.{section}.{field}")
        if coefficient == -1:
            sections[section][field] = number
        else:
            if coefficient == 0:
                sections[section][field] = []
            cast(list[float], sections[section][field]).append(number)
    result: WingGeometry = {"init": sections[0], "mid": sections[1:], "winglet": None}
    encode_wing(result, "decoded wing")
    return result


def projected_aspect_ratio(wing: WingGeometry) -> float:
    """Use the same projected half-span/area convention as calculate_wing_mac."""
    root_chord = cast(float, wing["init"]["chord_root"])
    area = 0.0
    span = 0.0
    for section in [wing["init"], *wing["mid"]]:
        chord_tip = cast(float, section["chord_tip"])
        projected_span = cast(float, section["span"]) * cos(radians(cast(float, section["dihedral"])))
        area += 0.5 * (root_chord + chord_tip) * projected_span
        span += projected_span
        root_chord = chord_tip
    if area <= 0 or span <= 0:
        raise ValueError(f"Projected wing dimensions must be positive: area={area}, span={span}.")
    return 2 * span ** 2 / area
