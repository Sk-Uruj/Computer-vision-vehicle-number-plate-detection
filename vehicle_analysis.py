import cv2
import time
import csv
import os
import re
import sys
import queue
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List, Any

import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError(f"Failed to import Ultralytics YOLO: {e}")

try:
    import easyocr
except Exception as e:
    raise RuntimeError(f"Failed to import EasyOCR: {e}")


@dataclass
class AppConfig:
    source: str
    vehicle_model_path: str = "yolov8n.pt"
    plate_model_path: str = "best.pt"
    csv_log_path: str = "vehicle_compliance_log.csv"
    loop_video: bool = False
    frame_queue_size: int = 4
    ocr_queue_size: int = 128
    reconnect_delay_sec: float = 3.0
    max_reconnect_attempts_before_backoff: int = 5
    reconnect_backoff_max_sec: float = 30.0
    vehicle_conf: float = 0.35
    plate_conf: float = 0.30
    vehicle_iou: float = 0.50
    plate_iou: float = 0.45
    track_persist: bool = True
    tracker_name: str = "bytetrack.yaml"
    easyocr_langs: Tuple[str, ...] = ("en",)
    ocr_min_confidence: float = 0.70
    plate_crop_padding_ratio: float = 0.15
    plate_resize_scale: float = 3.0
    display_window_name: str = "Vehicle ANPR Analytics"
    font_scale: float = 0.6
    line_thickness: int = 2
    box_thickness: int = 2
    track_ttl_sec: float = 3.0
    stable_plate_hits_required: int = 3
    plate_match_history_size: int = 10
    plate_text_stability_threshold: float = 0.75
    draw_debug: bool = True
    debug_dir: str = "debug_outputs"


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("vehicle_analysis")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(threadName)s | %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


LOGGER = setup_logging()
VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,2}[0-9]{4}$")


def safe_makedirs(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def safe_makedirs_for_file(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def now_ts() -> float:
    return time.time()


def format_timestamp(ts: Optional[float] = None) -> str:
    ts = ts if ts is not None else now_ts()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def clean_plate_text(text: str) -> str:
    if not text:
        return ""
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def majority_vote(values: List[str]) -> str:
    if not values:
        return ""
    freq: Dict[str, int] = {}
    for v in values:
        if not v:
            continue
        freq[v] = freq.get(v, 0) + 1
    if not freq:
        return ""
    return max(freq.items(), key=lambda kv: kv[1])[0]


def bbox_xyxy_to_int(b):
    return [int(round(x)) for x in b]


def plate_rejection_reason(text: str, confidence: float, min_conf: float) -> str:
    if not text:
        return "empty OCR"
    if confidence < min_conf:
        return f"confidence too low ({confidence:.3f} < {min_conf:.3f})"
    if not PLATE_REGEX.match(text):
        return f"regex mismatch ({text})"
    return ""


@dataclass
class FramePacket:
    frame: np.ndarray
    timestamp: float
    frame_id: int


@dataclass
class OCRJob:
    track_id: int
    plate_crop: np.ndarray
    vehicle_bbox: Tuple[int, int, int, int]
    plate_bbox: Tuple[int, int, int, int]
    timestamp: float
    frame_id: int


@dataclass
class OCRResult:
    track_id: int
    raw_text: str
    clean_text: str
    confidence: float
    timestamp: float
    frame_id: int


@dataclass
class TrackState:
    track_id: int
    vehicle_class_id: int
    vehicle_class_name: str
    last_vehicle_bbox: Tuple[int, int, int, int]
    last_seen_ts: float
    first_seen_ts: float
    plate_history: List[str] = field(default_factory=list)
    plate_conf_history: List[float] = field(default_factory=list)
    confirmed_plate: str = ""
    confirmed_plate_conf: float = 0.0
    stable_plate_hits: int = 0
    logged: bool = False
    last_ocr_requested_frame_id: int = -1
    last_ocr_result_frame_id: int = -1
    last_plate_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)


class DebugSaver:
    def __init__(self, debug_dir: str):
        self.debug_dir = debug_dir
        self.frames_dir = os.path.join(debug_dir, "frames")
        self.vehicles_dir = os.path.join(debug_dir, "vehicles")
        self.plates_dir = os.path.join(debug_dir, "plates")
        safe_makedirs(self.frames_dir)
        safe_makedirs(self.vehicles_dir)
        safe_makedirs(self.plates_dir)

    def save_image(self, folder: str, prefix: str, track_id: int, frame_id: int, img: np.ndarray) -> None:
        try:
            if img is None or img.size == 0:
                return
            filename = f"{prefix}_track{track_id}_frame{frame_id}_{int(time.time()*1000)}.jpg"
            path = os.path.join(folder, filename)
            cv2.imwrite(path, img)
        except Exception as e:
            LOGGER.exception(f"Debug save error: {e}")


class FrameReader(threading.Thread):
    def __init__(self, source: str, frame_queue: "queue.Queue[FramePacket]", stop_event: threading.Event,
                 reconnect_delay_sec: float, max_reconnect_attempts_before_backoff: int,
                 reconnect_backoff_max_sec: float, loop_video: bool = False):
        super().__init__(daemon=True, name="FrameReader")
        self.source = source
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.reconnect_delay_sec = reconnect_delay_sec
        self.max_reconnect_attempts_before_backoff = max_reconnect_attempts_before_backoff
        self.reconnect_backoff_max_sec = reconnect_backoff_max_sec
        self.loop_video = loop_video
        self.cap = None
        self.frame_id = 0
        self.is_file_source = os.path.isfile(source)

    def _open_capture(self) -> bool:
        try:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.cap = cv2.VideoCapture(self.source) if self.is_file_source else cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            opened = self.cap.isOpened()
            if not opened:
                LOGGER.warning(f"Failed to open source: {self.source}")
            else:
                LOGGER.info(f"Opened video source: {self.source}")
            return opened
        except Exception as e:
            LOGGER.exception(f"Capture open error: {e}")
            return False

    def _put_latest(self, packet: FramePacket) -> None:
        try:
            if self.frame_queue.full():
                try:
                    _ = self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put_nowait(packet)
        except Exception as e:
            LOGGER.exception(f"Frame queue put error: {e}")

    def run(self) -> None:
        while not self.stop_event.is_set():
            if self.cap is None or not self.cap.isOpened():
                if not self._open_capture():
                    time.sleep(self.reconnect_delay_sec)
                    continue
            try:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    if self.is_file_source and self.loop_video:
                        try:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        except Exception:
                            break
                    LOGGER.warning("Source read failed; reconnecting")
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
                    time.sleep(self.reconnect_delay_sec)
                    continue
                self.frame_id += 1
                self._put_latest(FramePacket(frame=frame, timestamp=now_ts(), frame_id=self.frame_id))
            except Exception as e:
                LOGGER.exception(f"Frame read loop error: {e}")
                try:
                    if self.cap is not None:
                        self.cap.release()
                except Exception:
                    pass
                self.cap = None
                time.sleep(self.reconnect_delay_sec)


class OCRWorker(threading.Thread):
    def __init__(self, ocr_queue: "queue.Queue[OCRJob]", result_queue: "queue.Queue[OCRResult]",
                 stop_event: threading.Event, langs: Tuple[str, ...]):
        super().__init__(daemon=True, name="OCRWorker")
        self.ocr_queue = ocr_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.reader = easyocr.Reader(list(langs), gpu=False)

    def _preprocess_for_ocr(self, crop: np.ndarray) -> np.ndarray:
        try:
            if crop is None or crop.size == 0:
                return crop
            if len(crop.shape) == 3:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = crop.copy()

            h, w = gray.shape[:2]
            gray = cv2.resize(gray, (max(1, int(w * 3.0)), max(1, int(h * 3.0))), interpolation=cv2.INTER_CUBIC)
            gray = cv2.bilateralFilter(gray, 9, 75, 75)

            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]], dtype=np.float32)
            sharp = cv2.filter2D(gray, -1, kernel)

            thr = cv2.adaptiveThreshold(
                sharp, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31, 11
            )
            return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            LOGGER.exception(f"OCR preprocessing error: {e}")
            return crop

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                job = self.ocr_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                pre = self._preprocess_for_ocr(job.plate_crop)
                results = self.reader.readtext(pre, detail=1, paragraph=False)

                best_text = ""
                best_conf = 0.0
                for item in results:
                    if len(item) >= 3:
                        txt = clean_plate_text(str(item[1]))
                        conf = float(item[2])
                        if conf > best_conf and txt:
                            best_text = txt
                            best_conf = conf

                result = OCRResult(job.track_id, best_text, clean_plate_text(best_text), best_conf, job.timestamp, job.frame_id)
                try:
                    self.result_queue.put_nowait(result)
                except queue.Full:
                    try:
                        _ = self.result_queue.get_nowait()
                        self.result_queue.put_nowait(result)
                    except Exception:
                        pass
            except Exception as e:
                LOGGER.exception(f"OCR worker error: {e}")
            finally:
                try:
                    self.ocr_queue.task_done()
                except Exception:
                    pass


class ComplianceLogger:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.lock = threading.Lock()
        safe_makedirs_for_file(csv_path)
        self._ensure_header()

    def _ensure_header(self) -> None:
        try:
            if not os.path.exists(self.csv_path):
                with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", "track_id", "vehicle_class", "plate_text", "plate_confidence",
                                     "vehicle_bbox_x1", "vehicle_bbox_y1", "vehicle_bbox_x2", "vehicle_bbox_y2",
                                     "plate_bbox_x1", "plate_bbox_y1", "plate_bbox_x2", "plate_bbox_y2"])
        except Exception as e:
            LOGGER.exception(f"CSV header init error: {e}")

    def write_record(self, record: Dict[str, Any]) -> None:
        with self.lock:
            try:
                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        record.get("timestamp", ""),
                        record.get("track_id", ""),
                        record.get("vehicle_class", ""),
                        record.get("plate_text", ""),
                        record.get("plate_confidence", ""),
                        *record.get("vehicle_bbox", ("", "", "", "")),
                        *record.get("plate_bbox", ("", "", "", "")),
                    ])
            except Exception as e:
                LOGGER.exception(f"CSV write error: {e}")


class VehicleAnalysisEngine:
    def __init__(self, config: AppConfig):
        self.cfg = config
        self.stop_event = threading.Event()
        self.frame_queue: "queue.Queue[FramePacket]" = queue.Queue(maxsize=config.frame_queue_size)
        self.ocr_queue: "queue.Queue[OCRJob]" = queue.Queue(maxsize=config.ocr_queue_size)
        self.ocr_result_queue: "queue.Queue[OCRResult]" = queue.Queue()
        self.compliance_logger = ComplianceLogger(config.csv_log_path)
        self.vehicle_model = None
        self.plate_model = None
        self.tracks: Dict[int, TrackState] = {}
        self.track_lock = threading.Lock()
        self.debug_saver = DebugSaver(config.debug_dir)
        self.frame_reader = FrameReader(config.source, self.frame_queue, self.stop_event,
                                        config.reconnect_delay_sec, config.max_reconnect_attempts_before_backoff,
                                        config.reconnect_backoff_max_sec, config.loop_video)
        self.ocr_worker = OCRWorker(self.ocr_queue, self.ocr_result_queue, self.stop_event, config.easyocr_langs)

    def load_models(self) -> None:
        self.vehicle_model = YOLO(self.cfg.vehicle_model_path)
        self.plate_model = YOLO(self.cfg.plate_model_path)
        LOGGER.info("Models loaded successfully")
        LOGGER.info(f"Saving crops to: {os.path.abspath(self.cfg.debug_dir)}")

    def start(self) -> None:
        self.load_models()
        self.frame_reader.start()
        self.ocr_worker.start()
        self._main_loop()

    def _update_track_state(self, track_id: int, vehicle_class_id: int, bbox: Tuple[int, int, int, int]) -> TrackState:
        with self.track_lock:
            ts = now_ts()
            state = self.tracks.get(track_id)
            if state is None:
                state = TrackState(track_id, vehicle_class_id, VEHICLE_CLASS_IDS.get(vehicle_class_id, "vehicle"), bbox, ts, ts)
                self.tracks[track_id] = state
            else:
                state.last_vehicle_bbox = tuple(int(0.7 * state.last_vehicle_bbox[i] + 0.3 * bbox[i]) for i in range(4))
                state.last_seen_ts = ts
            return state

    def _prune_tracks(self) -> None:
        ts = now_ts()
        with self.track_lock:
            stale_ids = [tid for tid, st in self.tracks.items() if (ts - st.last_seen_ts) > self.cfg.track_ttl_sec]
            for tid in stale_ids:
                self.tracks.pop(tid, None)

    def _enqueue_ocr_job(self, track_id: int, plate_crop: np.ndarray, vehicle_bbox, plate_bbox, frame_id: int) -> None:
        try:
            job = OCRJob(track_id, plate_crop, vehicle_bbox, plate_bbox, now_ts(), frame_id)
            if self.ocr_queue.full():
                try:
                    _ = self.ocr_queue.get_nowait()
                except queue.Empty:
                    pass
            self.ocr_queue.put_nowait(job)
        except Exception as e:
            LOGGER.exception(f"OCR enqueue error: {e}")

    def _handle_ocr_results(self) -> None:
        while True:
            try:
                result = self.ocr_result_queue.get_nowait()
            except queue.Empty:
                break

            try:
                with self.track_lock:
                    state = self.tracks.get(result.track_id)
                    if state is None:
                        continue

                    state.last_ocr_result_frame_id = result.frame_id

                    reason = plate_rejection_reason(result.clean_text, result.confidence, self.cfg.ocr_min_confidence)
                    if reason:
                        LOGGER.info(
                            f"[REJECTED] track={result.track_id} frame={result.frame_id} "
                            f"text='{result.clean_text}' conf={result.confidence:.3f} reason={reason}"
                        )
                        continue

                    state.plate_history.append(result.clean_text)
                    state.plate_conf_history.append(result.confidence)

                    if len(state.plate_history) > self.cfg.plate_match_history_size:
                        state.plate_history.pop(0)
                    if len(state.plate_conf_history) > self.cfg.plate_match_history_size:
                        state.plate_conf_history.pop(0)

                    candidate = majority_vote(state.plate_history)
                    vote_ratio = state.plate_history.count(candidate) / max(1, len(state.plate_history)) if candidate else 0.0
                    avg_conf = float(np.mean(state.plate_conf_history)) if state.plate_conf_history else 0.0

                    if not candidate:
                        LOGGER.info(f"[REJECTED] track={result.track_id} frame={result.frame_id} reason=no candidate")
                        continue
                    if not PLATE_REGEX.match(candidate):
                        LOGGER.info(
                            f"[REJECTED] track={result.track_id} frame={result.frame_id} "
                            f"candidate='{candidate}' reason=regex mismatch"
                        )
                        continue
                    if vote_ratio < self.cfg.plate_text_stability_threshold:
                        LOGGER.info(
                            f"[REJECTED] track={result.track_id} frame={result.frame_id} "
                            f"candidate='{candidate}' reason=unstable vote_ratio={vote_ratio:.2f}"
                        )
                        continue
                    if avg_conf < self.cfg.ocr_min_confidence:
                        LOGGER.info(
                            f"[REJECTED] track={result.track_id} frame={result.frame_id} "
                            f"candidate='{candidate}' reason=avg confidence too low ({avg_conf:.3f})"
                        )
                        continue

                    if candidate == state.confirmed_plate:
                        state.stable_plate_hits += 1
                    else:
                        state.confirmed_plate = candidate
                        state.confirmed_plate_conf = avg_conf
                        state.stable_plate_hits = 1

                    LOGGER.info(
                        f"[ACCEPTED-CANDIDATE] track={result.track_id} frame={result.frame_id} "
                        f"candidate='{candidate}' hits={state.stable_plate_hits} "
                        f"vote_ratio={vote_ratio:.2f} avg_conf={avg_conf:.2f}"
                    )

                    if state.stable_plate_hits >= self.cfg.stable_plate_hits_required and not state.logged:
                        state.logged = True
                        self.compliance_logger.write_record({
                            "timestamp": format_timestamp(result.timestamp),
                            "track_id": state.track_id,
                            "vehicle_class": state.vehicle_class_name,
                            "plate_text": state.confirmed_plate,
                            "plate_confidence": round(state.confirmed_plate_conf, 4),
                            "vehicle_bbox": state.last_vehicle_bbox,
                            "plate_bbox": state.last_plate_bbox,
                        })
                        LOGGER.info(f"[CONFIRMED] Logged track_id={state.track_id}, plate={state.confirmed_plate}")

            except Exception as e:
                LOGGER.exception(f"OCR result handling error: {e}")
            finally:
                try:
                    self.ocr_result_queue.task_done()
                except Exception:
                    pass

    def _crop_with_padding(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], pad_ratio: float) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        pad_x = int(bw * pad_ratio)
        pad_y = int(bh * pad_ratio)
        xx1 = clamp(x1 - pad_x, 0, w - 1)
        yy1 = clamp(y1 - pad_y, 0, h - 1)
        xx2 = clamp(x2 + pad_x, 0, w - 1)
        yy2 = clamp(y2 + pad_y, 0, h - 1)
        return frame[yy1:yy2, xx1:xx2].copy()

    def _detect_and_track_vehicles(self, frame: np.ndarray):
        try:
            return self.vehicle_model.track(source=frame, persist=self.cfg.track_persist, conf=self.cfg.vehicle_conf,
                                            iou=self.cfg.vehicle_iou, classes=list(VEHICLE_CLASS_IDS.keys()),
                                            tracker=self.cfg.tracker_name, verbose=False)
        except Exception as e:
            LOGGER.exception(f"Vehicle inference error: {e}")
            return []

    def _detect_plates(self, crop: np.ndarray):
        try:
            return self.plate_model.predict(source=crop, conf=self.cfg.plate_conf, iou=self.cfg.plate_iou, verbose=False)
        except Exception as e:
            LOGGER.exception(f"Plate inference error: {e}")
            return []

    def _main_loop(self) -> None:
        cv2.namedWindow(self.cfg.display_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.cfg.display_window_name, 1280, 720)
        while not self.stop_event.is_set():
            self._handle_ocr_results()
            self._prune_tracks()
            try:
                packet = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            frame = packet.frame
            frame_id = packet.frame_id

            try:
                self.debug_saver.save_image(self.debug_saver.frames_dir, "frame", 0, frame_id, frame)

                results = self._detect_and_track_vehicles(frame)
                display_frame = frame.copy()

                if results and len(results) > 0:
                    r = results[0]
                    boxes = getattr(r, "boxes", None)
                    if boxes is not None:
                        for box in boxes:
                            try:
                                cls_id = int(box.cls[0]) if hasattr(box.cls, "__len__") else int(box.cls)
                                if cls_id not in VEHICLE_CLASS_IDS:
                                    continue

                                if hasattr(box, "id") and box.id is not None:
                                    track_id = int(box.id[0]) if hasattr(box.id, "__len__") else int(box.id)
                                else:
                                    track_id = -1
                                if track_id < 0:
                                    continue

                                xyxy = box.xyxy[0].cpu().numpy().tolist()
                                vehicle_bbox = tuple(bbox_xyxy_to_int(xyxy))
                                state = self._update_track_state(track_id, cls_id, vehicle_bbox)

                                x1, y1, x2, y2 = vehicle_bbox
                                x1 = clamp(x1, 0, frame.shape[1] - 1)
                                y1 = clamp(y1, 0, frame.shape[0] - 1)
                                x2 = clamp(x2, 0, frame.shape[1] - 1)
                                y2 = clamp(y2, 0, frame.shape[0] - 1)

                                vehicle_crop = self._crop_with_padding(frame, vehicle_bbox, 0.0)
                                self.debug_saver.save_image(self.debug_saver.vehicles_dir, "vehicle", track_id, frame_id, vehicle_crop)

                                plate_results = self._detect_plates(vehicle_crop)
                                best_plate_bbox = None
                                best_plate_conf = 0.0

                                if plate_results and len(plate_results) > 0:
                                    pr = plate_results[0]
                                    pboxes = getattr(pr, "boxes", None)
                                    if pboxes is not None:
                                        for pbox in pboxes:
                                            try:
                                                pxyxy = pbox.xyxy[0].cpu().numpy().tolist()
                                                pconf = float(pbox.conf[0]) if hasattr(pbox.conf, "__len__") else float(pbox.conf)
                                                if pconf >= best_plate_conf:
                                                    best_plate_conf = pconf
                                                    best_plate_bbox = tuple(bbox_xyxy_to_int(pxyxy))
                                            except Exception:
                                                continue

                                if best_plate_bbox is not None:
                                    px1, py1, px2, py2 = best_plate_bbox
                                    plate_crop = self._crop_with_padding(vehicle_crop, best_plate_bbox, self.cfg.plate_crop_padding_ratio)
                                    self.debug_saver.save_image(self.debug_saver.plates_dir, "plate", track_id, frame_id, plate_crop)

                                    if plate_crop.size > 0:
                                        state.last_plate_bbox = best_plate_bbox
                                        if state.last_ocr_requested_frame_id != frame_id:
                                            state.last_ocr_requested_frame_id = frame_id
                                            self._enqueue_ocr_job(track_id, plate_crop, vehicle_bbox, best_plate_bbox, frame_id)

                                    cv2.rectangle(display_frame, (x1 + px1, y1 + py1), (x1 + px2, y1 + py2), (255, 0, 0), 2)

                                label = f"{state.vehicle_class_name} ID:{track_id}"
                                if state.confirmed_plate:
                                    label += f" | {state.confirmed_plate}"

                                color = (0, 255, 0) if state.confirmed_plate else (0, 200, 255)
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, self.cfg.box_thickness)
                                cv2.putText(display_frame, label, (x1, max(20, y1 - 10)),
                                            cv2.FONT_HERSHEY_SIMPLEX, self.cfg.font_scale, color, self.cfg.line_thickness)

                            except Exception as e:
                                LOGGER.exception(f"Per-box processing error: {e}")

                status = f"Frames:{frame_id} | Tracks:{len(self.tracks)} | OCRQ:{self.ocr_queue.qsize()}"
                cv2.putText(display_frame, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.imshow(self.cfg.display_window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.stop_event.set()
                    break

            except Exception as e:
                LOGGER.exception(f"Main loop error: {e}")

        self.shutdown()

    def shutdown(self) -> None:
        self.stop_event.set()
        try:
            self.frame_reader.join(timeout=3.0)
        except Exception:
            pass
        try:
            self.ocr_worker.join(timeout=3.0)
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        LOGGER.info("Shutdown complete")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Vehicle ANPR + Analytics for RTSP or Video")
    parser.add_argument("--source", required=True, help="RTSP URL or local video file path")
    parser.add_argument("--vehicle-model", default="yolov8n.pt", help="Vehicle YOLO model path")
    parser.add_argument("--plate-model", default="best.pt", help="Plate YOLO model path")
    parser.add_argument("--csv", default="vehicle_compliance_log.csv", help="Compliance CSV path")
    parser.add_argument("--loop-video", action="store_true", help="Loop video if source is a file")
    parser.add_argument("--frame-queue-size", type=int, default=4)
    parser.add_argument("--ocr-queue-size", type=int, default=128)
    parser.add_argument("--vehicle-conf", type=float, default=0.35)
    parser.add_argument("--plate-conf", type=float, default=0.30)
    parser.add_argument("--debug-dir", default="debug_outputs", help="Directory to save crops/frames")
    args = parser.parse_args()

    cfg = AppConfig(
        source=args.source,
        vehicle_model_path=args.vehicle_model,
        plate_model_path=args.plate_model,
        csv_log_path=args.csv,
        loop_video=args.loop_video,
        frame_queue_size=args.frame_queue_size,
        ocr_queue_size=args.ocr_queue_size,
        vehicle_conf=args.vehicle_conf,
        plate_conf=args.plate_conf,
        debug_dir=args.debug_dir,
    )

    engine = VehicleAnalysisEngine(cfg)
    try:
        engine.start()
    except KeyboardInterrupt:
        LOGGER.info("KeyboardInterrupt received")
        engine.shutdown()
    except Exception as e:
        LOGGER.exception(f"Fatal error: {e}")
        engine.shutdown()
        raise


if __name__ == "__main__":
    main()