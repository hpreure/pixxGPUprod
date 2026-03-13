"""
Probe Calibration — Camera Time Offset Calculator
===================================================
Calculates camera time offsets by comparing OCR-detected bib numbers
to participant finish times from the timing system.

The offset is computed as a running median of (photo_time - finish_time)
deltas across all successful OCR matches.
"""

import logging
from datetime import datetime, time as dt_time
from typing import List, Optional, Tuple, Dict
from statistics import median

from src.detection_config import settings
from src.workers.inference_engine import get_engine, InferenceResult, PROFILE_PROBE
from src.workers import identity_db as db
from src.workers.detection_common import resolve_path

logger = logging.getLogger(__name__)


class ProbeCalibration:
    """Handles probe calibration for camera time offset calculation."""
    
    def __init__(self, project_id: str):
        """
        Initialize probe calibration for a project.
        
        Args:
            project_id: The project ID to calibrate
        """
        self.project_id = str(project_id)
        self.participant_data = {}
        self._load_participant_data()
    
    def _load_participant_data(self):
        """Load all participant bib numbers and finish times for this project."""
        try:
            with db.get_cursor() as cur:
                cur.execute(
                    """
                    SELECT bib_number, finish_time
                    FROM public.api_participantinfo
                    WHERE project_id = %s AND finish_time IS NOT NULL
                    """,
                    (self.project_id,)
                )

                for bib, finish_time_str in cur.fetchall():
                    if finish_time_str:
                        self.participant_data[bib] = finish_time_str

            logger.info(
                "Loaded %d participants for project %s",
                len(self.participant_data), self.project_id,
            )
        
        except Exception as e:
            logger.error("Failed to load participant data: %s", e, exc_info=True)
            raise
    
    def calculate_offset(
        self,
        photos: List[dict],
        camera_serial: str
    ) -> Tuple[Optional[float], str]:
        """
        Calculate camera time offset from a batch of finish line photos.
        
        Args:
            photos: List of photo dicts with 'path' and 'capture_time'
            camera_serial: Camera serial number
        
        Returns:
            Tuple of (offset_seconds, status)
            - offset_seconds: Median offset in seconds, or None if failed
            - status: "completed", "insufficient_data", or "failed"
        """
        logger.info(
            f"Starting calibration for camera {camera_serial}: {len(photos)} photos"
        )
        
        # Build absolute file paths (map VPS/relative paths to local)
        photo_paths = []
        photo_times = {}

        for photo in photos:
            abs_path = resolve_path(photo['path'])
            photo_paths.append(abs_path)
            photo_times[abs_path] = photo['capture_time']
        
        # Run inference on all photos (probe profile — no ReID/Face)
        engine = get_engine(profile=PROFILE_PROBE)
        results: List[InferenceResult] = engine.process_photos(photo_paths)
        
        # Collect time deltas from successful OCR matches
        deltas = []
        
        for result in results:
            if not result.success:
                continue
            
            photo_time_str = photo_times.get(result.photo_path)
            if not photo_time_str:
                continue
            
            # Parse photo capture time — strip any timezone, use bare clock time
            # Handles: '2026-02-08 07:25:48', '2026-02-08T07:25:48-04:00', '...+00:00', '...Z'
            try:
                clean = photo_time_str.replace('Z', '+00:00')
                photo_dt = datetime.fromisoformat(clean).replace(tzinfo=None)
            except Exception as e:
                logger.warning("Failed to parse photo time %r: %s", photo_time_str, e)
                continue
            
            photo_seconds = (
                photo_dt.hour * 3600 +
                photo_dt.minute * 60 +
                photo_dt.second
            )
            
            # Check each person for OCR matches
            for person in result.persons:
                for bib_det in person.bibs:
                    bib_num = bib_det.bib_number
                    
                    if bib_num not in self.participant_data:
                        continue
                    
                    finish_time_str = self.participant_data[bib_num]
                    
                    # Convert finish time (HH:MM:SS) to seconds since midnight
                    try:
                        finish_time = self._parse_time_of_day(finish_time_str)
                        finish_seconds = (
                            finish_time.hour * 3600 +
                            finish_time.minute * 60 +
                            finish_time.second
                        )
                        
                        # Delta = camera_clock - timing_clock
                        # Positive = camera ahead, Negative = camera behind
                        # VPS corrects with: corrected = capture_time - offset
                        delta = photo_seconds - finish_seconds
                        
                        deltas.append(delta)
                        
                        logger.info(
                            f"Match: bib={bib_num}, "
                            f"photo_clock={photo_dt.strftime('%H:%M:%S')}, "
                            f"finish={finish_time_str}, "
                            f"delta={delta:.0f}s"
                        )
                    
                    except Exception as e:
                        logger.warning(
                            f"Failed to parse finish time for bib {bib_num}: "
                            f"{finish_time_str} - {e}"
                        )
        
        # Tally stats
        success_count = sum(1 for r in results if r.success)
        fail_count = sum(1 for r in results if not r.success)
        total_persons = sum(len(r.persons) for r in results if r.success)
        self.match_count = len(deltas)
        
        logger.info(
            f"Inference complete: {success_count} photos OK, {fail_count} failed, "
            f"{total_persons} persons detected, {len(deltas)} bib matches"
        )
        
        # Calculate median offset
        if len(deltas) < 3:
            logger.warning(
                f"Insufficient data for calibration: only {len(deltas)} matches "
                f"(minimum 3 required)"
            )
            return None, "insufficient_data"
        
        median_offset = median(deltas)
        
        logger.info(
            f"Calibration complete for {camera_serial}: "
            f"offset={median_offset:.2f}s from {len(deltas)} matches"
        )
        
        return median_offset, "completed"
    
    @staticmethod
    def _parse_time_of_day(time_str: str) -> dt_time:
        """
        Parse a time-of-day string in various formats.
        
        Supported formats:
        - "7:44:54"       (H:MM:SS)
        - "07:44:54"      (HH:MM:SS)
        - "7:44"          (H:MM)
        - "7:28:20.7 AM"  (12-hour with fractional seconds + AM/PM)
        - "1:30:05.6 PM"  (12-hour with fractional seconds + AM/PM)
        
        Fractional seconds are truncated to whole seconds.
        
        Args:
            time_str: Time string
        
        Returns:
            datetime.time object
        """
        time_str = time_str.strip()

        # Strip AM/PM suffix and adjust hours for 12-hour clock
        is_pm = False
        is_am = False
        upper = time_str.upper()
        if upper.endswith("PM"):
            is_pm = True
            time_str = time_str[:-2].strip()
        elif upper.endswith("AM"):
            is_am = True
            time_str = time_str[:-2].strip()

        # Try HH:MM:SS(.f) or H:MM:SS(.f)
        if ':' in time_str:
            parts = time_str.split(':')
            if len(parts) == 3:
                h, m = int(parts[0]), int(parts[1])
                s = int(float(parts[2]))          # handles "20.7" → 20
                if is_pm and h != 12:
                    h += 12
                elif is_am and h == 12:
                    h = 0
                return dt_time(h, m, s)
            elif len(parts) == 2:
                h, m = int(parts[0]), int(float(parts[1]))
                if is_pm and h != 12:
                    h += 12
                elif is_am and h == 12:
                    h = 0
                return dt_time(h, m, 0)

        raise ValueError(f"Unsupported time format: {time_str}")


def run_probe_calibration(
    project_id: str,
    photos: List[dict],
    camera_serial: str
) -> Dict:
    """
    Run probe calibration and return the result.
    
    Timezone-agnostic: compares bare camera clock time to bare finish time.
    The offset captures whatever difference exists between the two clocks,
    including any timezone mismatch, which is then consistently applied.
    
    Args:
        project_id: Project ID
        photos: List of photo dictionaries
        camera_serial: Camera serial number
    
    Returns:
        Result dictionary with offset and status
    """
    try:
        calibrator = ProbeCalibration(project_id)
        offset, status = calibrator.calculate_offset(photos, camera_serial)
        
        return {
            "task_type": "probe_calibration_result",
            "project_id": int(project_id),
            "camera_serial": camera_serial,
            "offset_seconds": round(offset, 2) if offset is not None else None,
            "status": status,
            "match_count": calibrator.match_count if hasattr(calibrator, 'match_count') else None
        }
    
    except Exception as e:
        logger.error("Probe calibration failed: %s", e, exc_info=True)
        return {
            "task_type": "probe_calibration_result",
            "project_id": int(project_id),
            "camera_serial": camera_serial,
            "offset_seconds": None,
            "status": "failed",
            "error": str(e)
        }
