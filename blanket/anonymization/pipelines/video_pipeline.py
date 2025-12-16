"""Video anonymization pipeline with frame-by-frame processing."""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import time

from blanket.anonymization.pipelines.image_pipeline import generate_synthetic_identity


class VideoPipeline:
    """Pipeline for anonymizing faces in videos."""

    def __init__(
        self,
        output_dir: str = "output",
        debug_dir: Optional[str] = None,
        face_detector_type: str = "yolo",
        landmarks_detector_type: str = "spiga",
        device: Optional[str] = None,
        identity_image_path: Optional[str] = None,
        identity_timestamp: Optional[float] = None,
        save_frames: bool = False,
        debug: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.face_detector_type = face_detector_type
        self.landmarks_detector_type = landmarks_detector_type
        self.device = device
        self.identity_image_path = identity_image_path
        self.identity_timestamp = identity_timestamp if identity_timestamp is not None else 0.0
        self.save_frames = save_frames
        self.debug = debug

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        if self.save_frames:
            self.frames_dir = self.output_dir / "frames"
            self.frames_dir.mkdir(parents=True, exist_ok=True)
        if self.debug:
            self.debug_frames_dir = self.output_dir / "debug_frames"
            self.debug_frames_dir.mkdir(parents=True, exist_ok=True)

        self._anonymizer = None

    def _get_anonymizer(self, identity_path: str):
        if self._anonymizer is not None:
            return self._anonymizer

        from blanket.anonymization.methods.facefusion import FaceFusionDirectAnonymizer
        config_path = Path(__file__).parent.parent.parent / "configs" / "module_parameters" / "facefusion_parameters.yaml"
        self._anonymizer = FaceFusionDirectAnonymizer(
            synthetic_face_path=identity_path,
            config_path=str(config_path)
        )

        return self._anonymizer

    def _extract_identity_frame(self, video_path: str) -> np.ndarray:
        """Extract identity frame from video at specified timestamp."""
        print(f"Extracting identity frame at {self.identity_timestamp}s...")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(self.identity_timestamp * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(f"Failed to extract frame at {self.identity_timestamp}s")

        return frame

    def run(self, video_path: str) -> Dict[str, Any]:
        start_time = time.time()
        video_path = Path(video_path)

        if not video_path.exists():
            return {"success": False, "error": f"Video not found: {video_path}"}

        print(f"Processing video: {video_path}")

        if self.identity_image_path:
            identity_path = self.identity_image_path
            print(f"Using custom identity: {identity_path}")
        else:
            print("Generating synthetic identity...")
            identity_frame = self._extract_identity_frame(str(video_path))

            identity_path, mask_path = generate_synthetic_identity(
                image=identity_frame,
                output_dir=str(self.output_dir),
                device=self.device,
                save_debug=self.debug_dir is not None,
            )

            print(f"Saved synthetic identity: {identity_path}")

        anonymizer = self._get_anonymizer(identity_path)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"success": False, "error": f"Failed to open video: {video_path}"}

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video info: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")

        output_video_path = self.output_dir / f"{video_path.stem}_anonymized.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

        frame_count = 0
        last_successful_frame = None
        processing_start_time = time.time()
        print("Processing frames...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % 30 == 0 or frame_count == 1:
                elapsed_processing = time.time() - processing_start_time
                current_fps = frame_count / elapsed_processing if elapsed_processing > 0 else 0
                progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                eta_seconds = (total_frames - frame_count) / current_fps if current_fps > 0 else 0
                eta_min = int(eta_seconds // 60)
                eta_sec = int(eta_seconds % 60)
                print(f"  Frame {frame_count}/{total_frames} ({progress:.1f}%) | {current_fps:.2f} FPS | ETA: {eta_min}m {eta_sec}s")

            try:
                if self.debug:
                    anonymized_frame, bounding_boxes, debug_frame = anonymizer.anonymize(
                        frame, detections=[], draw_debug_bboxes=True
                    )
                    debug_path = self.debug_frames_dir / f"debug_{frame_count:06d}.jpg"
                    cv2.imwrite(str(debug_path), debug_frame)
                else:
                    anonymized_frame, bounding_boxes = anonymizer.anonymize(frame, detections=[])

                last_successful_frame = anonymized_frame

                if self.save_frames:
                    frame_path = self.frames_dir / f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(str(frame_path), anonymized_frame)

                out.write(anonymized_frame)

            except Exception as e:
                print(f"  Warning: Failed to process frame {frame_count}: {e}")
                if self.debug and hasattr(e, 'debug_image') and e.debug_image is not None:
                    debug_path = self.debug_frames_dir / f"debug_{frame_count:06d}.jpg"
                    cv2.imwrite(str(debug_path), e.debug_image)

                if last_successful_frame is not None:
                    fallback_frame = last_successful_frame
                else:
                    # first frame detection fail fallback
                    fallback_frame = np.zeros_like(frame)
                    print(f"    Using black frame (no face detected yet)")

                out.write(fallback_frame)

                if self.save_frames:
                    frame_path = self.frames_dir / f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(str(frame_path), fallback_frame)

        cap.release()
        out.release()

        elapsed_total = time.time() - start_time
        elapsed_processing = time.time() - processing_start_time
        avg_fps = frame_count / elapsed_processing if elapsed_processing > 0 else 0

        print(f"\nProcessing complete!")
        print(f"  Frames processed: {frame_count}")
        print(f"  Processing time: {elapsed_processing:.2f}s ({avg_fps:.2f} FPS)")
        print(f"  Total time: {elapsed_total:.2f}s")
        print(f"  Output video: {output_video_path}")

        return {
            "success": True,
            "output_video": str(output_video_path),
            "identity_image": identity_path,
            "frames_processed": frame_count,
            "time_elapsed": elapsed_total,
            "processing_time": elapsed_processing,
            "avg_fps": avg_fps,
        }
