"""Read offline RADIO–TCN/VGGT-SLAM cuts against LEXI-SG's sorted RGB sequence."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
import json
import logging
import os
from pathlib import Path


logger = logging.getLogger(__name__)
_TIME_UNITS = {"s": Decimal(1), "ms": Decimal("0.001"),
               "us": Decimal("0.000001"), "ns": Decimal("0.000000001")}


def transition_source(settings):
    source = settings.get("source", "native")
    if source not in ("native", "external"):
        raise ValueError("room_segmentation.source must be 'native' or 'external'")
    return source


def sequence_paths(rgb_paths):
    return tuple(os.path.abspath(os.fspath(path)) for path in rgb_paths)


@dataclass(frozen=True)
class BoundaryPlan:
    boundary_file: str
    rgb_paths: tuple
    boundaries: tuple
    metadata: dict

    @property
    def indices(self):
        return tuple(event["frame_index"] for event in self.boundaries)


def _integer(value, field):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _timestamp(value, unit):
    if not isinstance(unit, str) or unit not in _TIME_UNITS:
        raise ValueError("Timestamp units must be explicitly set to s, ms, us, or ns")
    if isinstance(value, bool):
        raise ValueError("Timestamps must be finite numbers or numeric strings")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"Invalid timestamp: {value!r}") from exc
    if not number.is_finite():
        raise ValueError("Timestamps must be finite")
    with localcontext() as context:
        context.prec = max(28, len(number.as_tuple().digits) + 10)
        return number * _TIME_UNITS[unit]


def load_external_boundaries(settings, rgb_paths):
    """Return a validated plan, or None for native mode.

    Plain indices explicitly refer to the supplied sorted sequence. When a
    RADIO–TCN export declares a different num_frames, every cut must instead
    match an exact image_name or an explicitly configured filename timestamp.
    Relative time_seconds, fps ratios, and emitted_frame_index never move cuts.
    """
    if transition_source(settings) == "native":
        return None
    if not settings.get("boundary_file"):
        raise ValueError("External room segmentation requires room_segmentation.boundary_file")
    path = Path(settings["boundary_file"]).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle, parse_float=Decimal)
    if isinstance(payload, list):  # Also accepted by the existing VGGT-SLAM integration.
        payload = {"boundaries": payload}
    if not isinstance(payload, dict) or not isinstance(payload.get("boundaries"), list):
        raise ValueError("Boundary JSON must contain a 'boundaries' list (or be that list)")

    paths = sequence_paths(rgb_paths)
    if not paths or list(paths) != sorted(paths) or len(set(paths)) != len(paths):
        raise ValueError("Boundaries require a nonempty, unique, sorted RGB sequence")
    names = [Path(frame).name for frame in paths]
    if len(set(names)) != len(names):
        raise ValueError("RGB filenames must be unique for unambiguous boundary alignment")
    by_name = {name: index for index, name in enumerate(names)}
    count = payload.get("num_frames")
    if count is not None and _integer(count, "num_frames") < 1:
        raise ValueError("num_frames must be positive")
    different_sequence_length = count is not None and count != len(paths)
    if "fps" in payload:
        _timestamp(payload["fps"], "s")
        if Decimal(str(payload["fps"])) <= 0:
            raise ValueError("fps must be positive")

    by_timestamp = None
    aligned = []
    previous_source_index = -1
    for position, event in enumerate(payload["boundaries"]):
        if not isinstance(event, dict):
            raise ValueError(f"Boundary {position} must be an object")
        source_index = event.get("frame_index")
        if "frame_index" in event:
            source_index = _integer(source_index, f"Boundary {position} frame_index")
            if not 0 < source_index < (count if count is not None else len(paths)):
                raise ValueError(f"Boundary {position} frame_index must be between 1 and "
                                 f"{(count if count is not None else len(paths)) - 1}")
            if source_index <= previous_source_index:
                raise ValueError("Source boundary indices must be unique and strictly increasing")
            previous_source_index = source_index
        identity_index = None
        alignment = "frame_index"
        if "image_name" in event:
            name = event["image_name"]
            if not isinstance(name, str) or name not in by_name:
                raise ValueError(f"Boundary {position} image_name {name!r} is not in the RGB "
                                 "sequence; re-export boundaries for this sampling. No nearest-frame matching is used.")
            identity_index = by_name[name]
            alignment = "image_name"
        if "timestamp" in event:
            filename_unit = settings.get("frame_timestamp_unit")
            if not isinstance(filename_unit, str) or filename_unit not in _TIME_UNITS:
                raise ValueError("Timestamp alignment requires room_segmentation.frame_timestamp_unit "
                                 "(s, ms, us, ns) describing numeric RGB filename stems")
            event_unit = payload.get("timestamp_unit", filename_unit)
            if by_timestamp is None:
                timestamps = [_timestamp(Path(frame).stem, filename_unit) for frame in paths]
                if len(set(timestamps)) != len(timestamps):
                    raise ValueError("RGB filename timestamps are ambiguous")
                by_timestamp = {stamp: index for index, stamp in enumerate(timestamps)}
            stamp = _timestamp(event["timestamp"], event_unit)
            if stamp not in by_timestamp:
                raise ValueError(f"Boundary {position} timestamp does not exactly match an RGB frame")
            timestamp_index = by_timestamp[stamp]
            if identity_index is not None and identity_index != timestamp_index:
                raise ValueError(f"Boundary {position} image_name and timestamp disagree")
            identity_index = timestamp_index
            alignment = "timestamp" if "image_name" not in event else "image_name+timestamp"

        if identity_index is not None:
            if source_index is not None and not different_sequence_length and source_index != identity_index:
                raise ValueError(f"Boundary {position} frame_index and frame identity disagree; "
                                 "provide the detector's num_frames when its sampling differs")
            index = identity_index
        else:
            if source_index is None:
                raise ValueError(f"Boundary {position} needs frame_index, image_name, or an exact "
                                 "timestamp. Relative time_seconds cannot be aligned reliably.")
            if different_sequence_length:
                raise ValueError(f"Detector num_frames={count} differs from RGB count={len(paths)}; "
                                 "every boundary needs an exact image_name or timestamp. Do not scale indices by fps.")
            index = source_index
        if not 0 < index < len(paths):
            raise ValueError(f"Boundary {position} must start a new room after frame 0 and before the sequence ends")
        if aligned and index <= aligned[-1]["frame_index"]:
            raise ValueError("Boundary indices must be unique and strictly increasing after alignment")
        record = {"frame_index": index, "image_name": names[index], "alignment": alignment}
        if source_index is not None:
            record["source_frame_index"] = source_index
        if "timestamp" in event:
            record.update(timestamp=str(event["timestamp"]), timestamp_unit=event_unit)
        aligned.append(record)

    if count is None and any(event["alignment"] == "frame_index" for event in aligned):
        logger.warning("Index-only boundaries assume this exact sorted RGB sequence (%d frames). "
                       "Export num_frames and image_name to verify sampling.", len(paths))
    if different_sequence_length:
        logger.info("Aligned detector sequence (%d frames) to RGB sequence (%d frames) by exact frame identity",
                    count, len(paths))
    metadata = {key: payload[key] for key in ("sequence", "model") if isinstance(payload.get(key), str)}
    if count is not None:
        metadata["num_frames"] = count
    if "fps" in payload:
        metadata["fps"] = float(payload["fps"])
    return BoundaryPlan(str(path), paths, tuple(aligned), metadata)
