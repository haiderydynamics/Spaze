"""Structured Floor-Plan Data Schema for SPAZE.

Defines the core data models for rooms, walls, scale information, and floor plans.
Designed to decouple the Phase 1 computer vision outputs and user corrections from
the Streamlit UI, and to serve as a clean contract for future Phase 2 (CVAE) and
Phase 3 (3D generation) engines.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class DetectionStatus(str, Enum):
    """Status lifecycle of a detected or manually added room region."""
    CANDIDATE = "Candidate"
    NEEDS_REVIEW = "Needs Review"
    USER_CONFIRMED = "User Confirmed"
    REJECTED = "Rejected"


class CoordinateUnit(str, Enum):
    """Units representing coordinates and dimensions."""
    PIXEL = "pixel"
    METER = "meter"


class MeasurementSource(str, Enum):
    """Origin of the dimension values."""
    UNCALIBRATED_PIXEL = "uncalibrated_pixel"
    USER_SCALE_CALIBRATED = "user_scale_calibrated"
    MANUAL_ENTRY = "manual_entry"


@dataclass
class ScaleInfo:
    """Scale calibration metadata linking image pixels to real-world meters."""
    is_calibrated: bool = False
    pixels_per_meter: Optional[float] = None
    reference_length_meters: Optional[float] = None
    reference_line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    calibration_method: str = "uncalibrated"
    calibration_notes: str = "No real-world scale provided. Dimensions are measured in pixels."

    def calibrate(
        self,
        pixel_distance: float,
        real_meters: float,
        reference_line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        method: str = "known_wall_reference",
    ) -> None:
        """Calibrate the scale using a known physical distance."""
        if pixel_distance <= 0:
            raise ValueError(f"Pixel distance must be positive, got {pixel_distance}")
        if real_meters <= 0:
            raise ValueError(f"Real-world meters must be positive, got {real_meters}")
        self.is_calibrated = True
        self.pixels_per_meter = pixel_distance / real_meters
        self.reference_length_meters = real_meters
        self.reference_line = reference_line
        self.calibration_method = method
        self.calibration_notes = (
            f"Calibrated via {method}: {pixel_distance:.1f} px = {real_meters:.2f} m "
            f"({self.pixels_per_meter:.2f} px/m)."
        )

    def reset(self) -> None:
        """Reset calibration back to uncalibrated pixel mode."""
        self.is_calibrated = False
        self.pixels_per_meter = None
        self.reference_length_meters = None
        self.reference_line = None
        self.calibration_method = "uncalibrated"
        self.calibration_notes = "No real-world scale provided. Dimensions are measured in pixels."

    def pixels_to_meters(self, pixels: float) -> Optional[float]:
        """Convert a pixel measurement to meters if calibrated."""
        if not self.is_calibrated or self.pixels_per_meter is None or self.pixels_per_meter <= 0:
            return None
        return pixels / self.pixels_per_meter

    def square_pixels_to_sqm(self, sq_pixels: float) -> Optional[float]:
        """Convert square pixels to square meters if calibrated."""
        if not self.is_calibrated or self.pixels_per_meter is None or self.pixels_per_meter <= 0:
            return None
        return sq_pixels / (self.pixels_per_meter ** 2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ScaleInfo:
        """Construct from dictionary."""
        ref_line = data.get("reference_line")
        if ref_line is not None:
            ref_line = (
                (float(ref_line[0][0]), float(ref_line[0][1])),
                (float(ref_line[1][0]), float(ref_line[1][1])),
            )
        return cls(
            is_calibrated=bool(data.get("is_calibrated", False)),
            pixels_per_meter=float(data["pixels_per_meter"]) if data.get("pixels_per_meter") is not None else None,
            reference_length_meters=float(data["reference_length_meters"]) if data.get("reference_length_meters") is not None else None,
            reference_line=ref_line,
            calibration_method=str(data.get("calibration_method", "uncalibrated")),
            calibration_notes=str(data.get("calibration_notes", "")),
        )


@dataclass
class WallCandidate:
    """Detected wall line candidate with geometric attributes."""
    wall_id: str
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    length_pixels: float
    angle_degrees: float
    orientation: str  # "horizontal", "vertical", or "diagonal"
    confidence_score: float
    detection_method: str = "HoughLinesP"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wall_id": self.wall_id,
            "start_point": list(self.start_point),
            "end_point": list(self.end_point),
            "length_pixels": round(self.length_pixels, 2),
            "angle_degrees": round(self.angle_degrees, 2),
            "orientation": self.orientation,
            "confidence_score": round(self.confidence_score, 3),
            "detection_method": self.detection_method,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WallCandidate:
        return cls(
            wall_id=str(data["wall_id"]),
            start_point=(float(data["start_point"][0]), float(data["start_point"][1])),
            end_point=(float(data["end_point"][0]), float(data["end_point"][1])),
            length_pixels=float(data["length_pixels"]),
            angle_degrees=float(data["angle_degrees"]),
            orientation=str(data["orientation"]),
            confidence_score=float(data["confidence_score"]),
            detection_method=str(data.get("detection_method", "HoughLinesP")),
        )


@dataclass
class Room:
    """Structured representation of a detected or manually confirmed architectural room."""
    room_id: str
    room_type: str
    label: str
    x: float
    y: float
    width: float
    height: float
    polygon: List[Tuple[float, float]] = field(default_factory=list)
    coordinate_unit: str = CoordinateUnit.PIXEL.value
    detection_status: str = DetectionStatus.CANDIDATE.value
    confidence_score: float = 0.5
    confidence_explanation: str = "Candidate region based on initial contour geometry."
    measurement_source: str = MeasurementSource.UNCALIBRATED_PIXEL.value
    is_user_confirmed: bool = False
    physical_width: Optional[float] = None
    physical_height: Optional[float] = None
    physical_area: Optional[float] = None
    adjacent_room_ids: List[str] = field(default_factory=list)
    orientation_degrees: float = 0.0

    def __post_init__(self) -> None:
        """Validate numeric coordinates and ranges."""
        if self.width <= 0:
            raise ValueError(f"Room width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"Room height must be positive, got {self.height}")
        if self.x < 0 or self.y < 0:
            raise ValueError(f"Room coordinates must be non-negative, got ({self.x}, {self.y})")
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError(f"Confidence score must be in [0.0, 1.0], got {self.confidence_score}")

        # If polygon is not provided, initialize as bounding rectangle
        if not self.polygon:
            self.polygon = [
                (self.x, self.y),
                (self.x + self.width, self.y),
                (self.x + self.width, self.y + self.height),
                (self.x, self.y + self.height),
            ]

    @property
    def id(self) -> str:
        """Alias for room_id."""
        return self.room_id

    @property
    def area_sqm(self) -> Optional[float]:
        """Alias for physical_area."""
        return self.physical_area

    @property
    def width_m(self) -> Optional[float]:
        """Alias for physical_width."""
        return self.physical_width

    @property
    def height_m(self) -> Optional[float]:
        """Alias for physical_height."""
        return self.physical_height

    @property
    def pixel_area(self) -> float:
        """Bounding box pixel area."""
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Width to height aspect ratio."""
        return self.width / self.height if self.height > 0 else 0.0

    def apply_scale(self, scale_info: ScaleInfo) -> None:
        """Update physical dimensions using scale metadata."""
        if scale_info.is_calibrated and scale_info.pixels_per_meter:
            self.physical_width = round(scale_info.pixels_to_meters(self.width) or 0.0, 2)
            self.physical_height = round(scale_info.pixels_to_meters(self.height) or 0.0, 2)
            self.physical_area = round(scale_info.square_pixels_to_sqm(self.pixel_area) or 0.0, 2)
            if self.measurement_source != MeasurementSource.MANUAL_ENTRY.value:
                self.measurement_source = MeasurementSource.USER_SCALE_CALIBRATED.value
        else:
            self.physical_width = None
            self.physical_height = None
            self.physical_area = None
            if self.measurement_source != MeasurementSource.MANUAL_ENTRY.value:
                self.measurement_source = MeasurementSource.UNCALIBRATED_PIXEL.value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "room_id": self.room_id,
            "room_type": self.room_type,
            "label": self.label,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "width": round(self.width, 2),
            "height": round(self.height, 2),
            "polygon": [[round(pt[0], 2), round(pt[1], 2)] for pt in self.polygon],
            "coordinate_unit": self.coordinate_unit,
            "detection_status": self.detection_status,
            "confidence_score": round(self.confidence_score, 3),
            "confidence_explanation": self.confidence_explanation,
            "measurement_source": self.measurement_source,
            "is_user_confirmed": self.is_user_confirmed,
            "physical_width": self.physical_width,
            "physical_height": self.physical_height,
            "physical_area": self.physical_area,
            "adjacent_room_ids": list(self.adjacent_room_ids),
            "orientation_degrees": round(self.orientation_degrees, 2),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Room:
        """Construct from dictionary."""
        raw_polygon = data.get("polygon", [])
        polygon = [(float(pt[0]), float(pt[1])) for pt in raw_polygon]
        return cls(
            room_id=str(data["room_id"]),
            room_type=str(data.get("room_type", "Unclassified")),
            label=str(data.get("label", data["room_id"])),
            x=float(data["x"]),
            y=float(data["y"]),
            width=float(data["width"]),
            height=float(data["height"]),
            polygon=polygon,
            coordinate_unit=str(data.get("coordinate_unit", CoordinateUnit.PIXEL.value)),
            detection_status=str(data.get("detection_status", DetectionStatus.CANDIDATE.value)),
            confidence_score=float(data.get("confidence_score", 0.5)),
            confidence_explanation=str(data.get("confidence_explanation", "")),
            measurement_source=str(data.get("measurement_source", MeasurementSource.UNCALIBRATED_PIXEL.value)),
            is_user_confirmed=bool(data.get("is_user_confirmed", False)),
            physical_width=float(data["physical_width"]) if data.get("physical_width") is not None else None,
            physical_height=float(data["physical_height"]) if data.get("physical_height") is not None else None,
            physical_area=float(data["physical_area"]) if data.get("physical_area") is not None else None,
            adjacent_room_ids=list(data.get("adjacent_room_ids", [])),
            orientation_degrees=float(data.get("orientation_degrees", 0.0)),
        )


@dataclass
class ProcessingMetadata:
    """Metadata regarding computer vision processing parameters and execution."""
    pipeline_version: str = "phase1-cv-v1"
    timestamp_utc: str = ""
    processing_time_ms: float = 0.0
    image_filename: str = ""
    image_sha256: str = ""
    warnings: List[str] = field(default_factory=list)
    parameters_used: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProcessingMetadata:
        return cls(
            pipeline_version=str(data.get("pipeline_version", "phase1-cv-v1")),
            timestamp_utc=str(data.get("timestamp_utc", "")),
            processing_time_ms=float(data.get("processing_time_ms", 0.0)),
            image_filename=str(data.get("image_filename", "")),
            image_sha256=str(data.get("image_sha256", "")),
            warnings=list(data.get("warnings", [])),
            parameters_used=dict(data.get("parameters_used", {})),
        )


@dataclass
class FloorPlan:
    """Master structured floor-plan schema representing an entire architectural image."""
    image_width: int
    image_height: int
    coordinate_system: str = "top_left_origin"
    scale_information: ScaleInfo = field(default_factory=ScaleInfo)
    wall_candidates: List[WallCandidate] = field(default_factory=list)
    candidate_rooms: List[Room] = field(default_factory=list)
    confirmed_rooms: List[Room] = field(default_factory=list)
    processing_metadata: ProcessingMetadata = field(default_factory=ProcessingMetadata)
    adjacency_matrix: Optional[List[List[int]]] = None
    room_order: Optional[List[str]] = None
    preference: Optional[str] = None
    diversity_score: Optional[float] = None
    is_generated: bool = False

    def __post_init__(self) -> None:
        if self.image_width <= 0:
            raise ValueError(f"Image width must be positive, got {self.image_width}")
        if self.image_height <= 0:
            raise ValueError(f"Image height must be positive, got {self.image_height}")

    @property
    def image_path(self) -> str:
        """Convenience property for image filename."""
        return self.processing_metadata.image_filename or "floorplan.png"

    @property
    def total_rooms_count(self) -> int:
        """Total active rooms (confirmed + active candidates)."""
        return len(self.confirmed_rooms) + len(self.candidate_rooms)

    def add_candidate_room(self, room: Room) -> None:
        """Add a candidate room and apply current scale if calibrated."""
        room.apply_scale(self.scale_information)
        self.candidate_rooms.append(room)

    def confirm_room(self, room_id: str) -> bool:
        """Move a room from candidate_rooms to confirmed_rooms."""
        for i, room in enumerate(self.candidate_rooms):
            if room.room_id == room_id:
                room.detection_status = DetectionStatus.USER_CONFIRMED.value
                room.is_user_confirmed = True
                self.confirmed_rooms.append(room)
                self.candidate_rooms.pop(i)
                return True
        return False

    def reject_room(self, room_id: str) -> bool:
        """Mark a room as rejected."""
        for room in self.candidate_rooms:
            if room.room_id == room_id:
                room.detection_status = DetectionStatus.REJECTED.value
                room.is_user_confirmed = False
                return True
        for room in self.confirmed_rooms:
            if room.room_id == room_id:
                room.detection_status = DetectionStatus.REJECTED.value
                room.is_user_confirmed = False
                return True
        return False

    def update_scale(
        self,
        pixel_distance: float,
        real_meters: float,
        reference_line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    ) -> None:
        """Apply a new scale calibration to the floorplan and all contained rooms."""
        self.scale_information.calibrate(pixel_distance, real_meters, reference_line)
        for room in self.candidate_rooms:
            room.apply_scale(self.scale_information)
        for room in self.confirmed_rooms:
            room.apply_scale(self.scale_information)

    def compute_adjacency(self, tolerance_px: float = 8.0) -> List[List[int]]:
        """Calculate and store pairwise room adjacency matrix using geometric proximity/contact.
        
        Returns:
            Symmetric binary adjacency matrix of size (N, N) where N is active room count.
        """
        from shapely.geometry import Polygon as ShapelyPoly
        active_rooms = self.confirmed_rooms + self.candidate_rooms
        n = len(active_rooms)
        self.room_order = [r.room_id for r in active_rooms]
        if n == 0:
            self.adjacency_matrix = []
            return []

        matrix = [[0] * n for _ in range(n)]
        polys = []
        for r in active_rooms:
            try:
                poly = ShapelyPoly(r.polygon)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                polys.append(poly)
            except Exception:
                polys.append(None)

        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = polys[i], polys[j]
                if p1 is not None and p2 is not None and not p1.is_empty and not p2.is_empty:
                    # Check distance between polygons
                    dist = p1.distance(p2)
                    if dist <= tolerance_px:
                        matrix[i][j] = 1
                        matrix[j][i] = 1
                        if active_rooms[j].room_id not in active_rooms[i].adjacent_room_ids:
                            active_rooms[i].adjacent_room_ids.append(active_rooms[j].room_id)
                        if active_rooms[i].room_id not in active_rooms[j].adjacent_room_ids:
                            active_rooms[j].adjacent_room_ids.append(active_rooms[i].room_id)

        self.adjacency_matrix = matrix
        return matrix

    def to_dict(self) -> Dict[str, Any]:
        """Convert the entire structured floor plan to a dictionary."""
        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "coordinate_system": self.coordinate_system,
            "scale_information": self.scale_information.to_dict(),
            "wall_candidates": [w.to_dict() for w in self.wall_candidates],
            "candidate_rooms": [r.to_dict() for r in self.candidate_rooms],
            "confirmed_rooms": [r.to_dict() for r in self.confirmed_rooms],
            "processing_metadata": self.processing_metadata.to_dict(),
            "adjacency_matrix": self.adjacency_matrix,
            "room_order": self.room_order,
            "preference": self.preference,
            "diversity_score": self.diversity_score,
            "is_generated": self.is_generated,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export the structured floor plan as formatted JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FloorPlan:
        """Construct a FloorPlan instance from a dictionary."""
        scale_info = ScaleInfo.from_dict(data.get("scale_information", {}))
        walls = [WallCandidate.from_dict(w) for w in data.get("wall_candidates", [])]
        candidates = [Room.from_dict(r) for r in data.get("candidate_rooms", [])]
        confirmed = [Room.from_dict(r) for r in data.get("confirmed_rooms", [])]
        metadata = ProcessingMetadata.from_dict(data.get("processing_metadata", {}))

        return cls(
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
            coordinate_system=str(data.get("coordinate_system", "top_left_origin")),
            scale_information=scale_info,
            wall_candidates=walls,
            candidate_rooms=candidates,
            confirmed_rooms=confirmed,
            processing_metadata=metadata,
            adjacency_matrix=data.get("adjacency_matrix"),
            room_order=data.get("room_order"),
            preference=data.get("preference"),
            diversity_score=float(data["diversity_score"]) if data.get("diversity_score") is not None else None,
            is_generated=bool(data.get("is_generated", False)),
        )

    @classmethod
    def from_json(cls, json_str: str) -> FloorPlan:
        """Construct from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
