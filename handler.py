import runpod
import os
import websocket
import base64
import json
import uuid
import logging
import urllib.request
import urllib.parse
import binascii  # Base64 에러 처리를 위해 import
import subprocess
import librosa
import shutil
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def truncate_base64_for_log(base64_str, max_length=50):
    """Base64 문자열을 로깅용으로 앞부분만 잘라서 반환"""
    if not base64_str:
        return "None"
    if len(base64_str) <= max_length:
        return base64_str
    return f"{base64_str[:max_length]}... (총 {len(base64_str)} 문자)"


server_address = os.getenv("SERVER_ADDRESS", "127.0.0.1")
client_id = str(uuid.uuid4())


def download_file_from_url(url, output_path):
    """URL에서 파일을 다운로드하는 함수"""
    try:
        # wget을 사용하여 파일 다운로드
        result = subprocess.run(
            ["wget", "-O", output_path, "--no-verbose", "--timeout=30", url],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            logger.info(
                f"✅ URL에서 파일을 성공적으로 다운로드했습니다: {url} -> {output_path}"
            )
            return output_path
        else:
            logger.error(f"❌ wget 다운로드 실패: {result.stderr}")
            raise Exception(f"URL 다운로드 실패: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.error("❌ 다운로드 시간 초과")
        raise Exception("다운로드 시간 초과")
    except Exception as e:
        logger.error(f"❌ 다운로드 중 오류 발생: {e}")
        raise Exception(f"다운로드 중 오류 발생: {e}")


def save_base64_to_file(base64_data, temp_dir, output_filename):
    """Base64 데이터를 파일로 저장하는 함수"""
    try:
        # Base64 문자열 디코딩
        decoded_data = base64.b64decode(base64_data)

        # 디렉토리가 존재하지 않으면 생성
        os.makedirs(temp_dir, exist_ok=True)

        # 파일로 저장
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, "wb") as f:
            f.write(decoded_data)

        logger.info(f"✅ Base64 입력을 '{file_path}' 파일로 저장했습니다.")
        return file_path
    except (binascii.Error, ValueError) as e:
        logger.error(f"❌ Base64 디코딩 실패: {e}")
        raise Exception(f"Base64 디코딩 실패: {e}")


COMFY_INPUT_DIR = "/ComfyUI/input"
os.makedirs(COMFY_INPUT_DIR, exist_ok=True)


def get_file_ext(url_or_data, default_ext=".jpg"):
    """URL or string se file extension extract karein"""
    try:
        parsed = urllib.parse.urlparse(str(url_or_data)).path
        ext = os.path.splitext(parsed)[1].lower()
        if ext in [".png", ".jpg", ".jpeg", ".webp", ".mp4", ".wav", ".mp3", ".m4a", ".flac"]:
            return ext
    except Exception:
        pass
    return default_ext


def process_input(input_data, filename, input_type):
    """
    ComfyUI ke /ComfyUI/input/ directory me file save/copy karein.
    Returns (absolute_file_path, filename_for_comfy)
    """
    os.makedirs(COMFY_INPUT_DIR, exist_ok=True)
    target_path = os.path.abspath(os.path.join(COMFY_INPUT_DIR, filename))

    if input_type == "path":
        logger.info(f"📁 경로 입력 처리: {input_data}")
        if os.path.abspath(input_data) != target_path:
            shutil.copy2(input_data, target_path)
        return target_path, filename
    elif input_type == "url":
        logger.info(f"🌐 URL 입력 처리: {input_data}")
        download_file_from_url(input_data, target_path)
        return target_path, filename
    elif input_type == "base64":
        logger.info("🔢 Base64 입력 처리")
        save_base64_to_file(input_data, COMFY_INPUT_DIR, filename)
        return target_path, filename
    else:
        raise Exception(f"지원하지 않는 입력 타입: {input_type}")


def queue_prompt(prompt, input_type="image", person_count="single"):
    # Offload fail-safe right before /prompt (also with WORKER_GUARD_ENABLED=0); logs "[OFFLOAD] final: ..."
    from worker_guard.job_guard import final_offload_check
    final_offload_check(prompt)

    url = f"http://{server_address}:8188/prompt"
    logger.info(f"Queueing prompt to: {url}")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")

    # 디버깅을 위해 워크플로우 내용 로깅
    logger.info(f"워크플로우 노드 수: {len(prompt)}")
    if input_type == "image":
        logger.info(
            f"이미지 노드(284) 설정: {prompt.get('284', {}).get('inputs', {}).get('image', 'NOT_FOUND')}"
        )
    else:
        logger.info(
            f"비디오 노드(228) 설정: {prompt.get('228', {}).get('inputs', {}).get('video', 'NOT_FOUND')}"
        )
    logger.info(
        f"오디오 노드(125) 설정: {prompt.get('125', {}).get('inputs', {}).get('audio', 'NOT_FOUND')}"
    )
    logger.info(
        f"텍스트 노드(241) 설정: {prompt.get('241', {}).get('inputs', {}).get('positive_prompt', 'NOT_FOUND')}"
    )
    logger.info(
        f"네거티브 프롬프트(241) 설정: {prompt.get('241', {}).get('inputs', {}).get('negative_prompt', 'NOT_FOUND')}"
    )
    if person_count == "multi":
        if "307" in prompt:
            logger.info(
                f"두 번째 오디오 노드(307) 설정: {prompt.get('307', {}).get('inputs', {}).get('audio', 'NOT_FOUND')}"
            )
        elif "313" in prompt:
            logger.info(
                f"두 번째 오디오 노드(313) 설정: {prompt.get('313', {}).get('inputs', {}).get('audio', 'NOT_FOUND')}"
            )

    req = urllib.request.Request(url, data=data)
    req.add_header("Content-Type", "application/json")

    try:
        response = urllib.request.urlopen(req)
        result = json.loads(response.read())
        logger.info(f"프롬프트 전송 성공: {result}")
        return result
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        logger.error(f"HTTP 에러 발생: {e.code} - {e.reason}")
        logger.error(f"응답 내용: {error_body}")
        diag_info = ""
        try:
            vol_content = os.listdir("/runpod-volume") if os.path.exists("/runpod-volume") else "DOES_NOT_EXIST"
            comfy_models = {}
            if os.path.exists("/ComfyUI/models"):
                for sub in os.listdir("/ComfyUI/models"):
                    p = os.path.join("/ComfyUI/models", sub)
                    if os.path.isdir(p):
                        comfy_models[sub] = os.listdir(p)[:5]
            diag_info = f" | Diagnostics: /runpod-volume={vol_content}, /ComfyUI/models={comfy_models}"
        except Exception as de:
            diag_info = f" | Diagnostics error: {de}"
        raise Exception(f"ComfyUI HTTP {e.code} Error ({e.reason}): {error_body}{diag_info}")
    except Exception as e:
        logger.error(f"프롬프트 전송 중 오류: {e}")
        raise



def get_image(filename, subfolder, folder_type):
    url = f"http://{server_address}:8188/view"
    logger.info(f"Getting image from: {url}")
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"{url}?{url_values}") as response:
        return response.read()


def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    logger.info(f"Getting history from: {url}")
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())


class ComfyExecutionError(Exception):
    """ComfyUI 노드 실행 실패 (websocket execution_error / execution_interrupted)"""

    def __init__(self, message_type, data):
        self.details = {
            "type": message_type,
            "node_id": data.get("node_id"),
            "node_type": data.get("node_type"),
            "exception_type": data.get("exception_type"),
            "exception_message": (data.get("exception_message") or "").strip(),
            "traceback": data.get("traceback", []),
        }
        super().__init__(
            f"ComfyUI {message_type} at node {self.details['node_id']} "
            f"({self.details['node_type']}): {self.details['exception_type']}: "
            f"{self.details['exception_message']}"
        )


def get_videos(ws, prompt, input_type="image", person_count="single"):
    prompt_id = queue_prompt(prompt, input_type, person_count)["prompt_id"]
    logger.info(f"워크플로우 실행 시작: prompt_id={prompt_id}")

    output_videos = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message["type"] == "executing":
                data = message["data"]
                if data["node"] is not None:
                    logger.info(f"노드 실행 중: {data['node']}")
                if data["node"] is None and data["prompt_id"] == prompt_id:
                    logger.info("워크플로우 실행 완료")
                    break
            elif message["type"] in ("execution_error", "execution_interrupted"):
                data = message.get("data", {})
                if data.get("prompt_id") == prompt_id:
                    raise ComfyExecutionError(message["type"], data)
        else:
            continue

    logger.info(f"히스토리 조회 중: prompt_id={prompt_id}")
    history = get_history(prompt_id)[prompt_id]
    logger.info(f"출력 노드 수: {len(history['outputs'])}")

    for node_id in history["outputs"]:
        node_output = history["outputs"][node_id]
        videos_output = []
        if "gifs" in node_output:
            logger.info(
                f"노드 {node_id}에서 {len(node_output['gifs'])}개의 비디오 발견"
            )
            for idx, video in enumerate(node_output["gifs"]):
                # fullpath를 그대로 반환 (base64 인코딩하지 않음)
                video_path = video["fullpath"]
                logger.info(f"비디오 파일 경로: {video_path}")

                # 파일 존재 여부 및 크기 확인
                if os.path.exists(video_path):
                    file_size = os.path.getsize(video_path)
                    logger.info(
                        f"비디오 {idx+1} 발견: {video_path} (크기: {file_size} bytes)"
                    )
                else:
                    logger.warning(f"비디오 파일이 존재하지 않습니다: {video_path}")

                videos_output.append(video_path)
        else:
            logger.info(f"노드 {node_id}에 비디오 출력 없음")
        output_videos[node_id] = videos_output

    logger.info(f"총 {len(output_videos)}개 노드에서 비디오 파일 경로 수집 완료")
    return output_videos


def load_workflow(workflow_path):
    with open(workflow_path, "r") as file:
        return json.load(file)


def get_workflow_path(input_type, person_count):
    """input_type과 person_count에 따라 적절한 워크플로우 파일 경로를 반환"""
    if input_type == "image":
        if person_count == "single":
            return "/I2V_single.json"
        else:  # multi
            return "/I2V_multi.json"
    else:  # video
        if person_count == "single":
            return "/V2V_single.json"
        else:  # multi
            return "/V2V_multi.json"


def get_audio_duration(audio_path):
    """오디오 파일의 길이(초)를 반환"""
    try:
        duration = librosa.get_duration(path=audio_path)
        return duration
    except Exception as e:
        logger.warning(f"오디오 길이 계산 실패 ({audio_path}): {e}")
        return None


def calculate_max_frames_from_audio(wav_path, wav_path_2=None, fps=25):
    """오디오 길이를 기반으로 max_frames를 계산"""
    durations = []

    # 첫 번째 오디오 길이 계산
    duration1 = get_audio_duration(wav_path)
    if duration1 is not None:
        durations.append(duration1)
        logger.info(f"첫 번째 오디오 길이: {duration1:.2f}초")

    # 두 번째 오디오 길이 계산 (multi person인 경우)
    if wav_path_2:
        duration2 = get_audio_duration(wav_path_2)
        if duration2 is not None:
            durations.append(duration2)
            logger.info(f"두 번째 오디오 길이: {duration2:.2f}초")

    if not durations:
        logger.warning("오디오 길이를 계산할 수 없습니다. 기본값 81을 사용합니다.")
        return 81

    # 가장 긴 오디오 길이를 기준으로 max_frames 계산
    max_duration = max(durations)
    max_frames = int(max_duration * fps) + 81

    logger.info(
        f"가장 긴 오디오 길이: {max_duration:.2f}초, 계산된 max_frames: {max_frames}"
    )
    return max_frames


def handler(job):
    job_input = job.get("input", {})

    if job_input.get("task") == "tts":
        from tts.tts_handler import handle_tts
        return handle_tts(job_input)

    if "cmd" in job_input:
        import subprocess
        cmd = job_input["cmd"]
        logger.info(f"🔧 Running diagnostic command: {cmd}")
        output = subprocess.getoutput(cmd)
        return {"cmd": cmd, "output": output}

    # v24 job modes ("mode": "fast480" sets the size); a job without "mode" is not changed
    from worker_guard.modes import prepare_job_input, job_mode, UPSCALE_ONLY
    job_input, mode_error = prepare_job_input(job_input)
    if mode_error:
        logger.error(mode_error)
        return {"error": mode_error}
    if job_mode(job_input) == UPSCALE_ONLY:
        # v26: worker_guard runs upscale_only jobs itself; they never reach this handler with the guard on
        return {"error": "[MODE] upscale_only needs the worker guard (WORKER_GUARD_ENABLED=1)"}

    # job_input을 로깅할 때 base64 데이터는 truncate해서 출력
    log_input = job_input.copy()
    for key in ["image_base64", "video_base64", "wav_base64", "wav_base64_2"]:
        if key in log_input:
            log_input[key] = truncate_base64_for_log(log_input[key])

    logger.info(f"Received job input: {log_input}")
    task_id = f"task_{uuid.uuid4()}"

    # 입력 타입과 인물 수 확인
    input_type = job_input.get("input_type", "image")  # "image" 또는 "video"
    person_count = job_input.get("person_count", "single")  # "single" 또는 "multi"

    logger.info(f"워크플로우 타입: {input_type}, 인물 수: {person_count}")

    # 워크플로우 파일 경로 결정
    workflow_path = get_workflow_path(input_type, person_count)
    logger.info(f"사용할 워크플로우: {workflow_path}")

    # 이미지/비디오 입력 처리
    media_path = None
    media_node_name = None
    if input_type == "image":
        if "image_path" in job_input:
            ext = get_file_ext(job_input["image_path"], ".jpg")
            media_path, media_node_name = process_input(
                job_input["image_path"], f"{task_id}_input_image{ext}", "path"
            )
        elif "image_url" in job_input:
            ext = get_file_ext(job_input["image_url"], ".png")
            media_path, media_node_name = process_input(
                job_input["image_url"], f"{task_id}_input_image{ext}", "url"
            )
        elif "image_base64" in job_input:
            media_path, media_node_name = process_input(
                job_input["image_base64"], f"{task_id}_input_image.png", "base64"
            )
        else:
            default_img = "/examples/image.jpg"
            target = os.path.join(COMFY_INPUT_DIR, f"{task_id}_default_image.jpg")
            if os.path.exists(default_img):
                shutil.copy2(default_img, target)
                media_path, media_node_name = target, f"{task_id}_default_image.jpg"
            else:
                media_path, media_node_name = default_img, "image.jpg"
            logger.info("기본 이미지 파일을 사용합니다: /examples/image.jpg")
    else:  # video
        if "video_path" in job_input:
            ext = get_file_ext(job_input["video_path"], ".mp4")
            media_path, media_node_name = process_input(
                job_input["video_path"], f"{task_id}_input_video{ext}", "path"
            )
        elif "video_url" in job_input:
            ext = get_file_ext(job_input["video_url"], ".mp4")
            media_path, media_node_name = process_input(
                job_input["video_url"], f"{task_id}_input_video{ext}", "url"
            )
        elif "video_base64" in job_input:
            media_path, media_node_name = process_input(
                job_input["video_base64"], f"{task_id}_input_video.mp4", "base64"
            )
        else:
            default_vid = "/examples/image.jpg"
            target = os.path.join(COMFY_INPUT_DIR, f"{task_id}_default_video.mp4")
            if os.path.exists(default_vid):
                shutil.copy2(default_vid, target)
                media_path, media_node_name = target, f"{task_id}_default_video.mp4"
            else:
                media_path, media_node_name = default_vid, "image.jpg"
            logger.info("기본 비디오 파일을 사용합니다: /examples/image.jpg")

    # 오디오 입력 처리
    wav_path = None
    wav_node_name = None
    wav_path_2 = None
    wav_node_name_2 = None

    if "wav_path" in job_input:
        ext = get_file_ext(job_input["wav_path"], ".wav")
        wav_path, wav_node_name = process_input(
            job_input["wav_path"], f"{task_id}_input_audio{ext}", "path"
        )
    elif "wav_url" in job_input:
        ext = get_file_ext(job_input["wav_url"], ".wav")
        wav_path, wav_node_name = process_input(
            job_input["wav_url"], f"{task_id}_input_audio{ext}", "url"
        )
    elif "wav_base64" in job_input:
        wav_path, wav_node_name = process_input(
            job_input["wav_base64"], f"{task_id}_input_audio.wav", "base64"
        )
    else:
        default_audio = "/examples/audio.mp3"
        target = os.path.join(COMFY_INPUT_DIR, f"{task_id}_default_audio.mp3")
        if os.path.exists(default_audio):
            shutil.copy2(default_audio, target)
            wav_path, wav_node_name = target, f"{task_id}_default_audio.mp3"
        else:
            wav_path, wav_node_name = default_audio, "audio.mp3"
        logger.info("기본 오디오 파일을 사용합니다: /examples/audio.mp3")

    # 다중 인물용 두 번째 오디오 처리
    if person_count == "multi":
        if "wav_path_2" in job_input:
            ext = get_file_ext(job_input["wav_path_2"], ".wav")
            wav_path_2, wav_node_name_2 = process_input(
                job_input["wav_path_2"], f"{task_id}_input_audio_2{ext}", "path"
            )
        elif "wav_url_2" in job_input:
            ext = get_file_ext(job_input["wav_url_2"], ".wav")
            wav_path_2, wav_node_name_2 = process_input(
                job_input["wav_url_2"], f"{task_id}_input_audio_2{ext}", "url"
            )
        elif "wav_base64_2" in job_input:
            wav_path_2, wav_node_name_2 = process_input(
                job_input["wav_base64_2"], f"{task_id}_input_audio_2.wav", "base64"
            )
        else:
            wav_path_2, wav_node_name_2 = wav_path, wav_node_name
            logger.info("두 번째 오디오가 없어 첫 번째 오디오를 사용합니다.")


    # 필수 필드 검증 및 기본값 설정
    prompt_text = job_input.get("prompt", "A person talking naturally")
    width = job_input.get("width", 512)
    height = job_input.get("height", 512)

    # max_frame 설정 (입력이 없으면 오디오 길이 기반으로 자동 계산)
    max_frame = job_input.get("max_frame")
    if max_frame is None:
        logger.info(
            "max_frame이 입력되지 않았습니다. 오디오 길이를 기반으로 자동 계산합니다."
        )
        max_frame = calculate_max_frames_from_audio(
            wav_path, wav_path_2 if person_count == "multi" else None
        )
    else:
        logger.info(f"사용자 지정 max_frame: {max_frame}")

    logger.info(
        f"워크플로우 설정: prompt='{prompt_text}', width={width}, height={height}, max_frame={max_frame}"
    )
    logger.info(f"미디어 경로: {media_path}")
    logger.info(f"오디오 경로: {wav_path}")
    if person_count == "multi":
        logger.info(f"두 번째 오디오 경로: {wav_path_2}")

    prompt = load_workflow(workflow_path)

    # ------------------------------------------------------------------
    # 동적 Force Offload 설정
    # ------------------------------------------------------------------
    # 1. 입력에서 force_offload 읽기 (기본값 True: 작은 GPU에서 OOM 방지)
    force_offload = job_input.get("force_offload", True)
    logger.info(f"🔧 설정: force_offload={force_offload}")

    # 2. WanVideoSampler 노드에 force_offload 파라미터 주입
    sampler_node_id = None
    preferred_id = "128"

    # 효율성을 위해 먼저 선호 ID(128) 확인
    if preferred_id in prompt and prompt[preferred_id].get("class_type") == "WanVideoSampler":
        sampler_node_id = preferred_id
    else:
        # ID가 다른 경우 class type으로 검색 (폴백)
        for node_id, node_data in prompt.items():
            if node_data.get("class_type") == "WanVideoSampler":
                sampler_node_id = node_id
                break

    # sampler 노드를 찾은 경우 force_offload 파라미터 주입
    if sampler_node_id:
        # setdefault를 사용하여 'inputs' 딕셔너리가 없으면 생성
        inputs = prompt[sampler_node_id].setdefault("inputs", {})
        inputs["force_offload"] = force_offload
    else:
        logger.warning("⚠️ 경고: WanVideoSampler 노드를 찾을 수 없습니다. 워크플로우 기본값을 사용합니다.")
    # ------------------------------------------------------------------
    # 동적 Block Swap 및 VRAM 최적화 설정 (24GB VRAM OOM / Bus error 방지)
    # ------------------------------------------------------------------
    # 기본값 0: 모델 전체를 GPU에 상주 (24GB VRAM 기준). OOM 발생 시 입력으로 blocks_to_swap 지정
    blocks_to_swap = int(job_input.get("blocks_to_swap", 0))
    offload = blocks_to_swap > 0
    load_device = job_input.get("load_device", "offload_device" if offload else "main_device")
    for node_id, node_data in prompt.items():
        ctype = node_data.get("class_type")
        if ctype == "WanVideoBlockSwap":
            inp = node_data.setdefault("inputs", {})
            inp["blocks_to_swap"] = blocks_to_swap
            inp["offload_img_emb"] = offload
            inp["offload_txt_emb"] = offload
            logger.info(f"✅ 노드 {node_id} (WanVideoBlockSwap) 업데이트됨: blocks_to_swap={blocks_to_swap}")
        elif ctype == "WanVideoModelLoader":
            inp = node_data.setdefault("inputs", {})
            inp["load_device"] = load_device
            logger.info(f"✅ 노드 {node_id} (WanVideoModelLoader) 업데이트됨: load_device={load_device}")
            if job_input.get("base_precision"):
                inp["base_precision"] = job_input["base_precision"]
            elif inp.get("base_precision") == "fp16_fast":
                inp["base_precision"] = "bf16"
        elif ctype == "WanVideoLoraSelect":
            inp = node_data.setdefault("inputs", {})
            inp["low_mem_load"] = True
        elif job_input.get("vae_tiling") and ctype in ("WanVideoImageToVideoMultiTalk", "WanVideoDecode", "WanVideoEncode"):
            # Test input (TESTING.md step 2c): tiled VAE for the per-window decode/encode and the final decode
            inp = node_data.setdefault("inputs", {})
            inp["tiled_vae" if ctype == "WanVideoImageToVideoMultiTalk" else "enable_vae_tiling"] = True
            logger.info(f"✅ 노드 {node_id} ({ctype}) VAE tiling on")
    # ------------------------------------------------------------------

    # 파일 존재 여부 확인
    if not os.path.exists(media_path):
        logger.error(f"미디어 파일이 존재하지 않습니다: {media_path}")
        return {"error": f"미디어 파일을 찾을 수 없습니다: {media_path}"}

    if not os.path.exists(wav_path):
        logger.error(f"오디오 파일이 존재하지 않습니다: {wav_path}")
        return {"error": f"오디오 파일을 찾을 수 없습니다: {wav_path}"}

    if person_count == "multi" and wav_path_2 and not os.path.exists(wav_path_2):
        logger.error(f"두 번째 오디오 파일이 존재하지 않습니다: {wav_path_2}")
        return {"error": f"두 번째 오디오 파일을 찾을 수 없습니다: {wav_path_2}"}

    logger.info(f"미디어 파일 크기: {os.path.getsize(media_path)} bytes")
    logger.info(f"오디오 파일 크기: {os.path.getsize(wav_path)} bytes")
    if person_count == "multi" and wav_path_2:
        logger.info(f"두 번째 오디오 파일 크기: {os.path.getsize(wav_path_2)} bytes")

    # 워크플로우 노드 설정
    if input_type == "image":
        # I2V 워크플로우: 이미지 입력 설정 (ComfyUI /ComfyUI/input 기준 파일명 전달)
        prompt["284"]["inputs"]["image"] = media_node_name
    else:
        # V2V 워크플로우: 비디오 입력 설정
        prompt["228"]["inputs"]["video"] = media_node_name

    # 공통 설정
    prompt["125"]["inputs"]["audio"] = wav_node_name
    prompt["241"]["inputs"]["positive_prompt"] = prompt_text

    # 네거티브 프롬프트 + NAG: cfg 1.0에서는 샘플러가 negative 패스를 건너뛰므로
    # WanVideoApplyNAG로 적용. negative_prompt가 없으면 워크플로우는 기존과 동일.
    negative_prompt = job_input.get("negative_prompt")
    if negative_prompt:
        prompt["241"]["inputs"]["negative_prompt"] = negative_prompt
        if "|" in prompt_text:
            # '|' 다중 프롬프트는 split_cross_attn_ffn 경로를 타며 NAG 미지원
            logger.warning("⚠️ [NAG] off: '|' 다중 프롬프트는 NAG를 지원하지 않습니다.")
        elif sampler_node_id:
            nag_inputs = {
                "original_text_embeds": ["241", 0],
                "nag_text_embeds": ["241", 1],  # negative 전용 임베딩 (NAG용 출력)
                "nag_scale": float(job_input.get("nag_scale", 11.0)),
                "nag_tau": float(job_input.get("nag_tau", 2.5)),
                "nag_alpha": float(job_input.get("nag_alpha", 0.25)),
            }
            prompt["900"] = {
                "inputs": nag_inputs,
                "class_type": "WanVideoApplyNAG",
                "_meta": {"title": "WanVideo Apply NAG"},
            }
            prompt[sampler_node_id]["inputs"]["text_embeds"] = ["900", 0]
            logger.info(
                f"[NAG] on: scale={nag_inputs['nag_scale']}, tau={nag_inputs['nag_tau']}, alpha={nag_inputs['nag_alpha']}"
            )

    prompt["245"]["inputs"]["value"] = width
    prompt["246"]["inputs"]["value"] = height

    prompt["270"]["inputs"]["value"] = max_frame

    # 다중 인물용 두 번째 오디오 설정
    if person_count == "multi":
        # 워크플로우 타입에 따라 두 번째 오디오 노드 설정
        if input_type == "image":  # I2V_multi.json의 경우
            if "307" in prompt:
                prompt["307"]["inputs"]["audio"] = wav_node_name_2
        else:  # V2V_multi.json의 경우
            if "313" in prompt:
                prompt["313"]["inputs"]["audio"] = wav_node_name_2


    ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
    logger.info(f"Connecting to WebSocket: {ws_url}")

    # 먼저 HTTP 연결이 가능한지 확인
    http_url = f"http://{server_address}:8188/"
    logger.info(f"Checking HTTP connection to: {http_url}")

    # HTTP 연결 확인 (최대 1분)
    max_http_attempts = 180
    for http_attempt in range(max_http_attempts):
        try:
            import urllib.request

            response = urllib.request.urlopen(http_url, timeout=5)
            logger.info(f"HTTP 연결 성공 (시도 {http_attempt+1})")
            break
        except Exception as e:
            logger.warning(
                f"HTTP 연결 실패 (시도 {http_attempt+1}/{max_http_attempts}): {e}"
            )
            if http_attempt == max_http_attempts - 1:
                # ComfyUI 프로세스가 죽은 원인을 응답에 포함 (comfyui.log 마지막 부분)
                log_tail = subprocess.getoutput("tail -60 /ComfyUI/comfyui.log")
                raise Exception(
                    "ComfyUI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.\n"
                    f"--- comfyui.log (tail) ---\n{log_tail}"
                )
            time.sleep(1)

    ws = websocket.WebSocket()
    # 웹소켓 연결 시도 (최대 3분)
    max_attempts = int(180 / 5)  # 3분 (1초에 한 번씩 시도)
    for attempt in range(max_attempts):
        try:
            ws.connect(ws_url)
            logger.info(f"웹소켓 연결 성공 (시도 {attempt+1})")
            break
        except Exception as e:
            logger.warning(f"웹소켓 연결 실패 (시도 {attempt+1}/{max_attempts}): {e}")
            if attempt == max_attempts - 1:
                raise Exception("웹소켓 연결 시간 초과 (3분)")
            time.sleep(5)
    try:
        videos = get_videos(ws, prompt, input_type, person_count)
    except ComfyExecutionError as e:
        logger.error(f"❌ {e}\n{''.join(e.details['traceback'])}")
        return {"error": str(e), "details": e.details}
    except Exception as e:
        comfy_log_tail = ""
        try:
            with open("/ComfyUI/comfyui.log", "r", errors="ignore") as f:
                comfy_log_tail = "".join(f.readlines()[-150:])
        except Exception:
            pass
        # OOM kill 여부 (cgroup v2 / v1) 및 GPU 상태
        memory_events = subprocess.getoutput(
            "cat /sys/fs/cgroup/memory.events 2>/dev/null || cat /sys/fs/cgroup/memory/memory.oom_control 2>/dev/null"
        )
        gpu_state = subprocess.getoutput(
            "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv 2>&1"
        )
        comfy_alive = subprocess.getoutput("pgrep -f 'python main.py' || echo 'ComfyUI process not running'")
        logger.error(
            f"❌ ComfyUI 실행 중 오류/크래시 발생: {e}\nComfyUI log:\n{comfy_log_tail}\n"
            f"memory.events:\n{memory_events}\nGPU:\n{gpu_state}\nComfyUI PID: {comfy_alive}"
        )
        return {
            "error": f"{type(e).__name__}: {str(e)}",
            "comfyui_log": comfy_log_tail,
            "memory_events": memory_events,
            "gpu_state": gpu_state,
            "comfyui_pid": comfy_alive,
        }
    finally:
        ws.close()
    logger.info("웹소켓 연결 종료")

    # 비디오가 없는 경우 처리
    output_video_path = None
    logger.info("출력 비디오 검색 중...")

    for node_id in videos:
        if videos[node_id]:
            output_video_path = videos[node_id][0]
            logger.info(f"노드 {node_id}에서 출력 비디오 발견: {output_video_path}")
            break
        else:
            logger.info(f"노드 {node_id}는 비어있음")

    if not output_video_path:
        logger.error("출력 비디오를 찾을 수 없습니다. 모든 노드가 비어있습니다.")
        return {"error": "비디오를 찾을 수 없습니다."}

    # 비디오 파일 존재 여부 확인
    if not os.path.exists(output_video_path):
        logger.error(f"출력 비디오 파일이 존재하지 않습니다: {output_video_path}")
        return {"error": f"비디오 파일을 찾을 수 없습니다: {output_video_path}"}

    # network_volume 파라미터 확인
    use_network_volume = job_input.get("network_volume", False)
    logger.info(f"네트워크 볼륨 사용 여부: {use_network_volume}")

    if use_network_volume:
        # 네트워크 볼륨 사용: 파일 복사
        logger.info("네트워크 볼륨에 비디오 복사 시작")
        try:
            # 결과 비디오 파일 경로 생성
            output_filename = f"infinitetalk_{task_id}.mp4"
            output_path = f"/runpod-volume/{output_filename}"
            logger.info(f"원본 파일: {output_video_path}")
            logger.info(f"대상 경로: {output_path}")

            # 원본 파일 크기 확인
            source_file_size = os.path.getsize(output_video_path)
            logger.info(f"원본 파일 크기: {source_file_size} bytes")

            # 파일 복사 (shutil.copy2는 메타데이터도 함께 복사)
            shutil.copy2(output_video_path, output_path)
            logger.info("파일 복사 완료")

            # 복사된 파일 크기 확인
            copied_file_size = os.path.getsize(output_path)
            logger.info(f"복사된 파일 크기: {copied_file_size} bytes")

            if source_file_size == copied_file_size:
                logger.info(
                    f"✅ 결과 비디오를 '{output_path}'에 성공적으로 복사했습니다"
                )
            else:
                logger.warning(
                    f"⚠️ 파일 크기가 일치하지 않습니다: 원본={source_file_size}, 복사본={copied_file_size}"
                )

            result = {"video_path": output_path}
            if job_input.get("_return_video_file"):
                # v26: the 480p file for the upscale stage (worker_guard removes this key)
                result["_video_file"] = output_video_path
            return result

        except Exception as e:
            logger.error(f"❌ 비디오 복사 실패: {e}")
            return {"error": f"비디오 복사 실패: {e}"}
    else:
        # 네트워크 볼륨 미사용: Base64 인코딩하여 반환
        logger.info("Base64 인코딩 시작")
        logger.info(f"비디오 파일 경로: {output_video_path}")

        try:
            # 파일 크기 확인
            file_size = os.path.getsize(output_video_path)
            logger.info(f"원본 파일 크기: {file_size} bytes")

            # 파일을 읽어 base64 인코딩
            with open(output_video_path, "rb") as f:
                video_data = base64.b64encode(f.read()).decode("utf-8")

            encoded_size = len(video_data)
            logger.info(f"Base64 인코딩 완료: {encoded_size} 문자")
            logger.info(
                f"✅ Base64 인코딩된 비디오 반환: {truncate_base64_for_log(video_data)}"
            )
            result = {"video": video_data}
            if job_input.get("_return_video_file"):
                # v26: the 480p file for the upscale stage (worker_guard removes this key)
                result["_video_file"] = output_video_path
            return result

        except Exception as e:
            logger.error(f"❌ Base64 인코딩 실패: {e}")
            return {"error": f"Base64 인코딩 실패: {e}"}


from worker_guard.job_guard import install as install_job_guard
handler, get_videos = install_job_guard(handler, get_videos)
runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": lambda current_concurrency: 1
})

