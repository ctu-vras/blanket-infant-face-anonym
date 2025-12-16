"""FaceFusion wrapper """
import cv2
import numpy as np
import yaml
from pathlib import Path

from facefusion import state_manager
from facefusion import face_analyser, face_detector, face_landmarker, face_recognizer, face_classifier
from facefusion.processors.modules.face_swapper import core as face_swapper
from facefusion.processors.modules.face_enhancer import core as face_enhancer


class IoUFilterException(RuntimeError):
    def __init__(self, message, debug_image=None):
        super().__init__(message)
        self.debug_image = debug_image


def calculate_iou(bbox1, bbox2):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = bbox1_area + bbox2_area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def match_faces_across_frames(prev_bboxes, curr_bboxes, iou_threshold=0.5):
    matches = []
    used_curr_indices = set()

    for prev_idx, prev_bbox in enumerate(prev_bboxes):
        best_iou = 0.0
        best_curr_idx = -1

        for curr_idx, curr_bbox in enumerate(curr_bboxes):
            if curr_idx in used_curr_indices:
                continue

            iou = calculate_iou(prev_bbox, curr_bbox)

            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_curr_idx = curr_idx

        if best_curr_idx != -1:
            matches.append((prev_idx, best_curr_idx, best_iou))
            used_curr_indices.add(best_curr_idx)

    return matches


class FaceFusionDirectAnonymizer:
    def __init__(self, synthetic_face_path, model_path='./models/insightface', config_path=None):
        self.synthetic_face_path = synthetic_face_path

        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "module_parameters" / "facefusion_direct_parameters.yaml"

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.face_detector_score = config.get('face_detector_score', 0.5)
        self.face_landmarker_score = config.get('face_landmarker_score', 0.5)
        self.max_faces = config.get('max_faces', None)
        self.face_swapper_model = config.get('face_swapper_model', 'inswapper_128')
        self.face_mask_blur = config.get('face_mask_blur', 0.7)
        self.skip_nsfw = config.get('skip_nsfw', False)
        self.enable_face_enhancer = config.get('enable_face_enhancer', True)
        self.face_enhancer_model = config.get('face_enhancer_model', 'gfpgan_1.4')
        self.face_enhancer_blend = config.get('face_enhancer_blend', 100)
        self.enable_expression_restorer = config.get('enable_expression_restorer', False)
        self.expression_restorer_model = config.get('expression_restorer_model', 'live_portrait')
        self.expression_restorer_factor = config.get('expression_restorer_factor', 80)
        self.expression_restorer_areas = config.get('expression_restorer_areas', ['upper-face', 'lower-face'])
        self.execution_providers = config.get('execution_providers', ['CPUExecutionProvider'])

        self.iou_filter = config.get('iou_filter', False)
        self.iou_threshold = config.get('iou_threshold', 0.3)
        self.iou_skip_threshold = config.get('iou_skip_threshold', 10) 
        self.previous_bboxes = []
        self.frames_since_last_swap = 0
        # other models beside inswapper_128 not tested, but they are available in facefusion oficially
        available_models = ['blendswap_256', 'inswapper_128', 'inswapper_128_fp16',
                           'simswap_256', 'simswap_512_unofficial', 'uniface_256']
        if self.face_swapper_model not in available_models:
            self.face_swapper_model = 'inswapper_128'


        state_manager.init_item('download_providers', ['github', 'huggingface'])
        state_manager.init_item('log_level', 'info')
        state_manager.init_item('source_paths', [str(Path(synthetic_face_path).absolute())])
        state_manager.init_item('execution_providers', self.execution_providers)
        state_manager.init_item('execution_device_ids', ['0'])
        state_manager.init_item('execution_thread_count', 4)
        state_manager.init_item('face_detector_model', 'yolo_face')
        state_manager.init_item('face_detector_size', '640x640')
        state_manager.init_item('face_detector_score', self.face_detector_score)
        state_manager.init_item('face_detector_margin', (0, 0, 0, 0))
        state_manager.init_item('face_detector_angles', [0, 90, 180, 270])
        state_manager.init_item('face_landmarker_model', '2dfan4')
        state_manager.init_item('face_landmarker_score', self.face_landmarker_score)
        state_manager.init_item('face_selector_mode', 'many')
        state_manager.init_item('face_selector_order', 'large-small')
        state_manager.init_item('face_selector_age_start', 0)
        state_manager.init_item('face_selector_age_end', 100)
        state_manager.init_item('face_selector_gender', None)
        state_manager.init_item('skip_nsfw', self.skip_nsfw)
        state_manager.init_item('face_mask_types', ['box'])
        state_manager.init_item('face_mask_blur', self.face_mask_blur)
        state_manager.init_item('face_mask_padding', (0, 0, 0, 0))

        if self.face_swapper_model == 'blendswap_256':
            face_recognizer_model = 'arcface_blendswap'
            pixel_boost_default = '256x256'
        elif self.face_swapper_model in ['inswapper_128', 'inswapper_128_fp16']:
            face_recognizer_model = 'arcface_inswapper'
            pixel_boost_default = '128x128'
        elif self.face_swapper_model in ['simswap_256', 'simswap_512_unofficial']:
            face_recognizer_model = 'arcface_simswap'
            pixel_boost_default = '256x256' if self.face_swapper_model == 'simswap_256' else '512x512'
        elif self.face_swapper_model == 'uniface_256':
            face_recognizer_model = 'arcface_uniface'
            pixel_boost_default = '256x256'

        state_manager.init_item('face_recognizer_model', face_recognizer_model)
        state_manager.init_item('face_swapper_model', self.face_swapper_model)
        state_manager.init_item('face_swapper_pixel_boost', pixel_boost_default)
        state_manager.init_item('face_swapper_weight', 100)
        state_manager.init_item('video_memory_strategy', 'moderate')
        state_manager.init_item('face_mask_areas', [])
        state_manager.init_item('face_mask_regions', [])

        face_detector.pre_check()
        face_landmarker.pre_check()
        face_recognizer.pre_check()
        face_classifier.pre_check()

        if self.enable_face_enhancer:
            state_manager.init_item('face_enhancer_model', self.face_enhancer_model)
            state_manager.init_item('face_enhancer_blend', self.face_enhancer_blend)
            face_enhancer.pre_check()

        if self.enable_expression_restorer:
            state_manager.init_item('expression_restorer_model', self.expression_restorer_model)
            state_manager.init_item('expression_restorer_factor', self.expression_restorer_factor)
            state_manager.init_item('expression_restorer_areas', self.expression_restorer_areas)
            from facefusion.processors.modules.expression_restorer import core as expression_restorer
            expression_restorer.pre_check()

        face_swapper.pre_check()

        source_frame = cv2.imread(str(synthetic_face_path))

        self.source_faces = face_analyser.get_many_faces([source_frame])
        if len(self.source_faces) == 0:
            raise ValueError(f"No face detected in source: {synthetic_face_path}")

        self.source_face = self.source_faces[0]

    def _draw_debug_visualization(self, image, all_detected_bboxes, filtered_bboxes, final_bboxes, iou_values):
        # function to check BB for IoU filtering
        debug_img = image.copy()

        if len(self.previous_bboxes) > 0:
            for idx, bbox in enumerate(self.previous_bboxes):
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(debug_img, f'Prev {idx}', (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for idx, bbox in enumerate(all_detected_bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(debug_img, f'Det {idx}', (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        for idx, (bbox, iou) in enumerate(zip(filtered_bboxes, iou_values)):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(debug_img, f'Filt {idx} IoU:{iou:.2f}', (x1, y1 - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for idx, bbox in enumerate(final_bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(debug_img, f'Final {idx}', (x1 + 5, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        legend_y = 30
        cv2.putText(debug_img, 'Green: Previous frame', (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(debug_img, 'Blue: Detected', (10, legend_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(debug_img, 'Red: IoU Filtered', (10, legend_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(debug_img, 'Yellow: Final Swapped', (10, legend_y + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return debug_img

    def anonymize(self, image, detections, draw_debug_bboxes=False):
        target_faces = face_analyser.get_many_faces([image])

        if len(target_faces) == 0:
            raise RuntimeError("No faces detected")

        if self.max_faces is not None and len(target_faces) > self.max_faces:
            target_faces = target_faces[:self.max_faces]

        all_detected_bboxes = [face.bounding_box.tolist() for face in target_faces]
        filtered_bboxes = []
        iou_values = []

        iou_failed = False
        # prevent freeze
        skip_iou_check = self.frames_since_last_swap >= self.iou_skip_threshold

        if self.iou_filter and len(self.previous_bboxes) > 0 and not skip_iou_check:
            filtered_faces = []
            current_bboxes = [face.bounding_box.tolist() for face in target_faces]

            for idx, face in enumerate(target_faces):
                current_bbox = current_bboxes[idx]
                max_iou = 0.0
                for prev_bbox in self.previous_bboxes:
                    iou = calculate_iou(current_bbox, prev_bbox)
                    max_iou = max(max_iou, iou)

                if max_iou >= self.iou_threshold:
                    filtered_faces.append(face)
                    filtered_bboxes.append(current_bbox)
                    iou_values.append(max_iou)

            target_faces = filtered_faces

            if len(target_faces) == 0:
                iou_failed = True
                if draw_debug_bboxes:
                    print(f"  [DEBUG] IoU filter rejected all faces - will use previous frame")
                    debug_img = self._draw_debug_visualization(
                        image, all_detected_bboxes, filtered_bboxes, [], iou_values
                    )
                    raise IoUFilterException("IoU filter rejected all faces - use previous frame", debug_img)
        elif skip_iou_check and draw_debug_bboxes:
            print(f"  [DEBUG] Skipping IoU filter ({self.frames_since_last_swap} frames since last swap >= {self.iou_skip_threshold})")

        result_frame = image.copy()
        bounding_boxes = []

        for target_face in target_faces:
            try:
                if not isinstance(target_face.landmark_set, dict) or '5/68' not in target_face.landmark_set:
                    raise RuntimeError("Invalid landmarks")

                result_frame = face_swapper.swap_face(
                    source_face=self.source_face,
                    target_face=target_face,
                    temp_vision_frame=result_frame
                )

                if self.enable_expression_restorer:
                    from facefusion.processors.modules.expression_restorer import core as expression_restorer
                    result_frame = expression_restorer.restore_expression(
                        target_face=target_face,
                        target_vision_frame=result_frame,
                        temp_vision_frame=result_frame
                    )

                if self.enable_face_enhancer:
                    result_frame = face_enhancer.enhance_face(
                        target_face=target_face,
                        temp_vision_frame=result_frame
                    )

                bounding_boxes.append(target_face.bounding_box.tolist())

            except Exception as e:
                if draw_debug_bboxes:
                    print(f"  [DEBUG] Face swap failed: {e}")
                continue

        if iou_failed:
            self.frames_since_last_swap += 1
            raise RuntimeError("IoU filter rejected all faces - use previous frame")

        # swap -> reset counter and update previous bboxes
        if self.iou_filter:
            self.previous_bboxes = bounding_boxes
            self.frames_since_last_swap = 0

        if np.array_equal(result_frame, image):
            raise RuntimeError("FaceFusion returned unchanged image")

        if draw_debug_bboxes:
            debug_img = self._draw_debug_visualization(
                image, all_detected_bboxes, filtered_bboxes, bounding_boxes, iou_values
            )
            print(f"  [DEBUG] Prev: {len(self.previous_bboxes)}, Detected: {len(all_detected_bboxes)}, "
                  f"Filtered: {len(filtered_bboxes)}, Final: {len(bounding_boxes)}, IoU filter: {self.iou_filter}")
            return result_frame, bounding_boxes, debug_img

        return result_frame, bounding_boxes

    def get_face_count(self, image):
        return len(face_analyser.get_many_faces([image]))

    def clear_cache(self):
        face_swapper.clear_inference_pool()
        if self.enable_face_enhancer:
            face_enhancer.clear_inference_pool()