import logging
import hashlib
import json
from pathlib import Path

from lexisg.utils.data_loader import lookup_priors
from lexisg.utils.room_boundaries import load_external_boundaries, sequence_paths, transition_source
import numpy as np

logger = logging.getLogger(__name__)


class RoomSegmentation:
    def __init__(self, model, config, boundary_plan=None):

        self.corridor_confidence = 0
        self.in_doorway = False
        self.current_batch = []
        self.batch_index = 0
        self.segment_type = None

        self.model = model
        self.settings = config["room_segmentation"]
        self.source = transition_source(self.settings)
        self.boundary_plan = boundary_plan
        self._rgb_paths = None
        self._frame_lookup = {}
        self._next_frame_index = 0
        self._boundary_indices = set()
        self._batches = []
        self._assigned_frames = set()

        if self.source == "native":
            self.method = config["room_segmentation"]["method"]
            self.enter_confidence = config["room_segmentation"]["enter_confidence"]
            self.exit_confidence = config["room_segmentation"]["exit_confidence"]
            self.enter_threshold = config["room_segmentation"]["enter_threshold"]
            self.exit_threshold = config["room_segmentation"]["exit_threshold"]
        # If False, transition segments (e.g. doorways) are not emitted as their
        # own nodes: their frames are split at the midpoint between the room
        # being left and the room being entered. Keep True for transitions that
        # are real spaces in their own right (e.g. corridors).
        self.transitions_as_nodes = self.source == "native" and bool(self.settings.get("transitions_as_nodes", True))
        self._transition_start = None  # index into current_batch where the current transition began
        self.transition_frames = set()  # frames accumulated while inside a folded transition
        self.intrinsics = None
        self.depth_scale = None
        self.conf_threshold = 0.0
        self.depth_by_rgb = None
        self.pose_by_rgb = None

    def configure_sequence(self, rgb_paths):
        """Bind external indices and run diagnostics to the actual sorted input."""
        if self._rgb_paths is not None:
            raise RuntimeError("Room segmentation is already bound to a sequence")
        self._rgb_paths = list(rgb_paths)
        self._frame_lookup = {path: index for index, path in enumerate(rgb_paths)}
        if self.source == "external":
            if self.boundary_plan is None:
                self.boundary_plan = load_external_boundaries(self.settings, rgb_paths)
            if self.boundary_plan.rgb_paths != sequence_paths(rgb_paths):
                raise ValueError("RGB sequence changed since external boundaries were validated")
            self._boundary_indices = set(self.boundary_plan.indices)
        logger.info("Room transition source=%s, boundary_file=%s, external boundary indices=%s",
                    self.source, self.boundary_plan.boundary_file if self.boundary_plan else None,
                    sorted(self._boundary_indices))

    def _estimate_external_batch(self, img_path, end_of_sequence):
        if self._rgb_paths is None:
            raise RuntimeError("Call configure_sequence before using external transitions")
        if end_of_sequence:
            if self._next_frame_index != len(self._rgb_paths):
                raise ValueError("External segmentation ended before all RGB frames were consumed")
            if not self.current_batch:
                return False, None, None
            batch = self.current_batch
            self.current_batch = []
            return True, batch, ("room", self.batch_index)
        index = self._next_frame_index
        if index >= len(self._rgb_paths) or img_path != self._rgb_paths[index]:
            raise ValueError("External transitions must consume the validated RGB sequence in order")
        self._next_frame_index += 1
        if index in self._boundary_indices:
            batch = self.current_batch
            key = ("room", self.batch_index)
            self.current_batch = [img_path]  # The boundary frame belongs only to the new room.
            self.batch_index += 1
            return True, batch, key
        self.current_batch.append(img_path)
        return False, None, None

    def record_batch(self, batch, key):
        """Record batches actually submitted to the shared reconstruction backend."""
        indices = [self._frame_lookup[path] for path in batch]
        self._assigned_frames.update(indices)
        self._batches.append({
            "batch_index": int(key[1]), "label": key[0],
            "start_frame": indices[0], "end_frame_exclusive": indices[-1] + 1,
            "num_frames": len(indices),
            "first_image": Path(batch[0]).name, "last_image": Path(batch[-1]).name,
        })
        logger.info("Room batch %d (%s): frames [%d, %d), count=%d", key[1], key[0],
                    indices[0], indices[-1] + 1, len(indices))

    def save_summary(self, output_folder, complete=False):
        names = [Path(path).name for path in self._rgb_paths]
        boundaries = [{"frame_index": batch["start_frame"], "image_name": batch["first_image"]}
                      for batch in self._batches[1:]]
        summary = {
            "source": self.source, "num_frames": len(names),
            "rgb_sequence_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
            "segmentation_complete": complete, "num_room_batches": len(self._batches),
            "boundaries": boundaries, "batches": self._batches,
            "num_assigned_frames": len(self._assigned_frames),
            "unassigned_frame_count": len(names) - len(self._assigned_frames),
            "overlapping_frame_assignments": sum(batch["num_frames"] for batch in self._batches)
                                             - len(self._assigned_frames),
        }
        if self.boundary_plan is not None:
            summary.update(boundary_file=self.boundary_plan.boundary_file,
                           input_metadata=self.boundary_plan.metadata,
                           configured_boundaries=list(self.boundary_plan.boundaries))
        path = Path(output_folder) / "room_segmentation_summary.json"
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary

    def estimate_room_batch(self, img_path, logits, room_label, end_of_sequence=False):
        """
        Segments a sequence of images into rooms and doorways/corridors based on room_label predictions.
        Handles hysteresis (stability) and prevents empty batches for first frames.
        """
        if self.source == "external":
            return self._estimate_external_batch(img_path, end_of_sequence)

        # --- Determine if this frame is a corridor/doorway ---
        is_corridor = (room_label == self.method)

        # --- Update confidence counter ---
        if is_corridor:
            self.corridor_confidence = min(self.corridor_confidence + self.enter_confidence, 10)
        else:
            self.corridor_confidence = max(0.0, self.corridor_confidence - self.exit_confidence)

        # --- Initialize state on first frame ---
        if self.segment_type is None:
            self.segment_type = self.method if is_corridor else "room"
            self.current_batch = []
            self.batch_index = 0
            self.in_doorway = is_corridor
            # If we start in a corridor, pre-load confidence so we're already "in" corridor state
            if is_corridor:
                self.corridor_confidence = 6.0  # above enter_corridor threshold of 5

        if img_path is not None:
            self.current_batch.append(img_path)
            if not self.transitions_as_nodes and self.segment_type == self.method:
                self.transition_frames.add(img_path)

        # --- Stability thresholds (hysteresis) ---
        enter_corridor = (self.corridor_confidence > self.enter_threshold)
        exit_corridor  = (self.corridor_confidence < self.exit_threshold)

        # --- Folded transitions: don't emit transition nodes; split their frames between rooms ---
        if not self.transitions_as_nodes:
            if self.segment_type == "room" and enter_corridor:
                logger.info("room to %s (folded)", self.method)
                self.segment_type = self.method
                self.in_doorway = True
                self._transition_start = len(self.current_batch)
                return False, None, None
            elif self.segment_type == self.method and exit_corridor:
                logger.info("%s to room (folded)", self.method)
                self.segment_type = "room"
                self.in_doorway = False
                self.corridor_confidence = 0.0
                start = self._transition_start or 0
                self._transition_start = None
                if start == 0:
                    # Sequence began inside a transition: nothing to emit yet,
                    # these frames simply open the first room.
                    return False, None, None
                mid = start + (len(self.current_batch) - start) // 2
                batch_to_return = self.current_batch[:mid]
                key_to_return = self.batch_index
                self.batch_index += 1
                self.current_batch = self.current_batch[mid:]
                return True, batch_to_return, ("room", key_to_return)
            elif end_of_sequence:
                if len(self.current_batch) > 0:
                    batch_to_return = self.current_batch
                    key_to_return = self.batch_index
                    self.current_batch = []
                    return True, batch_to_return, ("room", key_to_return)
            return False, None, None

        # --- CASE 1: Transition from room → corridor ---
        if self.segment_type == "room" and enter_corridor:
            logger.info("room to corridor")
            # finalize the room batch, keep a few overlapping frames
            batch_to_return = self.current_batch[:-10] if len(self.current_batch) > 10 else self.current_batch
            key_to_return = self.batch_index

            # switch to corridor mode
            self.segment_type = self.method
            self.in_doorway = True
            self.batch_index += 1
            self.current_batch = self.current_batch[-10:]  # carry overlap forward

            return True, batch_to_return, ("room", key_to_return)

        # --- CASE 2: Transition from corridor → room ---
        elif self.segment_type == self.method and exit_corridor:
            logger.info("corridor to room")
            batch_to_return = self.current_batch[:-10] if len(self.current_batch) > 10 else self.current_batch
            key_to_return = self.batch_index

            # switch to room mode
            self.segment_type = "room"
            self.in_doorway = False
            self.batch_index += 1
            self.current_batch = self.current_batch[-10:]  # carry overlap forward

            # reset confidence so next corridor triggers faster
            self.corridor_confidence = 0.0

            return True, batch_to_return, (self.method, key_to_return)

        # --- CASE 3: End of the sequence ---
        elif end_of_sequence:
            if len(self.current_batch) > 0:
                batch_to_return = self.current_batch
                key_to_return = self.batch_index
                seg_type = self.segment_type
                self.current_batch = []  # clear for next sequence
                return True, batch_to_return, (seg_type, key_to_return)

        # --- Otherwise: keep accumulating ---
        return False, None, None


    def estimate_boundary_pairs(self, room_a, room_b, n_pairs=2):

        batch_a = room_a.img_paths
        batch_b = room_b.img_paths

        indices_a = list(range(max(0, len(batch_a) - n_pairs), len(batch_a)))
        indices_b = list(range(min(n_pairs, len(batch_b))))

        relative_transforms = []
        pair_data = []
        for idx_a in indices_a:
            for idx_b in indices_b:
                img_a = batch_a[idx_a]
                img_b = batch_b[idx_b]

                depth_batch, poses = lookup_priors([img_a, img_b], self.depth_by_rgb, self.pose_by_rgb)
                predictions = self.model.run(
                    batch=[img_a, img_b],
                    depth_batch=depth_batch,
                    intrinsics=self.intrinsics,
                    poses=poses,
                    conf_threshold=self.conf_threshold,
                    depth_scale=self.depth_scale,
                )

                T_b = predictions["extrinsic"][1]
                T_ab = T_b

                T_refa_a = room_a.local_poses[idx_a]
                T_refb_b = room_b.local_poses[idx_b]
                T_refa_refb = T_refa_a @ T_ab @ np.linalg.inv(T_refb_b)
                relative_transforms.append(T_refa_refb)

                pair_data.append({
                    "img_a": img_a,
                    "img_b": img_b,
                    "idx_a": idx_a,
                    "idx_b": idx_b,
                })

        return relative_transforms, pair_data
