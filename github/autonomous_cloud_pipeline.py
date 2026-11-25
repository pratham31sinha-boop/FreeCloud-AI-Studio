"""Autonomous 24/7 Cloud Video Pipeline for Serverless GitHub Actions Execution.

Runs completely independently of any user PC or tunnel:
  1. `plan`           - Splits target duration into parallel Cloud Chapters and sequential Extend-Clip Acts.
  2. `render-chapter` - Renders sequential acts using ZeroGPU AI Video Diffusion (LTX-Video / Wan 2.2 / Hunyuan 1.5)
                        with DIS optical-flow 1080p interpolation & stereo audio, chaining exact last frames.
  3. `publish-release`- Stitches all chapters with FFmpeg +faststart and publishes MP4 + Poster + Catalog JSON
                        to GitHub Releases (`cloud-gallery`) for permanent 24/7 browser streaming.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] autonomous_cloud: %(message)s")
logger = logging.getLogger("freecloud.autonomous")


STORY_BEATS = [
    ("Opening Establishing Motion", "Sweeping cinema opening revealing subject in full motion under atmospheric lighting"),
    ("Dynamic Acceleration", "Intensifying forward momentum and dynamic camera tracking close-up"),
    ("Climax Peak Action", "Explosive peak choreography with dramatic lighting reflections and kinetic force"),
    ("Twilight Resolution", "Heroic final departure gliding smoothly toward the glowing horizon"),
]


def build_autonomous_plan(
    job_id: str,
    prompt: str,
    duration_seconds: float,
    model_name: str,
    anime_style: str,
    width: int,
    height: int,
    fps: float = 24.0,
) -> Dict[str, Any]:
    """Divide job cleanly into parallel GitHub VM chapters and sequential Extend-Clip acts."""
    dur = max(4.0, min(600.0, float(duration_seconds)))
    # Up to 4 sequential ~6s acts per VM (~24s per VM), up to 8 parallel VMs (192s–600s)
    num_chapters = max(1, min(8, int(math.ceil(dur / 24.0))))
    per_chapter_dur = dur / float(num_chapters)

    relay_cycle = ["wan_2_1", "minimax_h3", "hunyuan_video", "wan_2_1"]
    chapters: List[Dict[str, Any]] = []
    t_cursor = 0.0

    for ch_idx in range(num_chapters):
        n_acts = max(1, min(4, int(round(per_chapter_dur / 6.0))))
        act_dur = round(per_chapter_dur / float(n_acts), 2)
        acts: List[Dict[str, Any]] = []
        for a_idx in range(n_acts):
            beat_title, beat_desc = STORY_BEATS[a_idx % len(STORY_BEATS)]
            act_model = relay_cycle[(ch_idx + a_idx) % len(relay_cycle)] if model_name == "auto_relay" else model_name
            acts.append(
                {
                    "act_index": a_idx,
                    "start_sec": round(t_cursor, 2),
                    "end_sec": round(t_cursor + act_dur, 2),
                    "duration_sec": act_dur,
                    "model_name": act_model,
                    "beat_title": beat_title,
                    "beat_desc": beat_desc,
                }
            )
            t_cursor += act_dur

        chapters.append(
            {
                "chapter_index": ch_idx,
                "chapter_title": f"CHAPTER {ch_idx + 1}/{num_chapters}" if num_chapters > 1 else None,
                "chapter_subtitle": prompt[:64],
                "acts": acts,
            }
        )

    return {
        "job_id": job_id,
        "prompt": prompt,
        "duration_seconds": dur,
        "model_name": model_name,
        "anime_style": anime_style,
        "width": width,
        "height": height,
        "fps": fps,
        "chapters": chapters,
    }


def _extract_last_frame_bgr(mp4_path: Path) -> Optional[Any]:
    import cv2

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return None
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n - 1))
        ok, frame = cap.read()
        return frame if ok and frame is not None and frame.size > 0 else None
    finally:
        cap.release()


def render_chapter_autonomous(
    plan: Dict[str, Any],
    chapter_index: int,
    out_dir: Path,
) -> Path:
    """Render one full chapter sequentially with real Cloud Diffusion I2V Extend-Clip chaining."""
    from backend.animation.anime_engine import AnimeSceneRenderer
    from backend.continuity.video_code import build_extended_clip_prompt, extract_handoff_bundle_from_mp4
    from backend.generation.cloud_diffusion import CloudVideoDiffusionClient, stream_diffusion_frames
    from backend.storage.video_processor import video_processor
    from github.worker_entrypoint import WorkerRunner, _build_gamma_lut, _classify_audio_theme

    out_dir.mkdir(parents=True, exist_ok=True)
    chapter = plan["chapters"][chapter_index]
    job_id = plan["job_id"]
    prompt = plan["prompt"]
    anime_style = plan["anime_style"]
    width = int(plan["width"])
    height = int(plan["height"])
    fps = float(plan.get("fps", 24.0))
    total_chunks = sum(len(c["acts"]) for c in plan["chapters"])

    runner = WorkerRunner(backend_url="http://localhost:8000", worker_id=f"cloud_ch_{chapter_index}")
    diff_client = CloudVideoDiffusionClient()
    cloud_mode = os.getenv("FREECLOUD_CLOUD_DIFFUSION", "auto").lower()

    act_files: List[Path] = []
    engines_used: List[str] = []
    prev_video_code: Optional[Dict[str, Any]] = None
    prev_last_bgr: Optional[Any] = None

    seed_base = int(hashlib.sha256(f"{job_id}_{chapter_index}".encode()).hexdigest()[:8], 16)

    for a_idx, act in enumerate(chapter["acts"]):
        act_dur = float(act["duration_sec"])
        total_frames = max(12, int(round(act_dur * fps)))
        act_model = act["model_name"]
        chunk_id = f"{job_id}_ch{chapter_index}_act{a_idx}"
        act_mp4 = out_dir / f"{chunk_id}.mp4"

        if a_idx == 0 or prev_video_code is None:
            act_prompt = (
                f"[Act {a_idx + 1} | {act['start_sec']}s-{act['end_sec']}s] "
                f"{prompt} -- {act['beat_title']}: {act['beat_desc']}"
            )
            is_extended = False
        else:
            act_prompt = build_extended_clip_prompt(
                original_prompt=prompt,
                next_scene_segment=f"{act['beat_title']}: {act['beat_desc']}",
                prev_chunk_index=a_idx - 1,
                next_chunk_index=a_idx,
                video_code=prev_video_code,
            )
            is_extended = True

        # Color grade & stereo audio synced from previous Video-as-Code handoff
        gamma, gain = (0.95, 1.05) if "wan" in act_model else ((0.89, 1.11) if "hunyuan" in act_model else (0.92, 1.08))
        gamma_lut = _build_gamma_lut(gamma, gain)
        prev_audio = (prev_video_code or {}).get("audio_continuity", {})
        init_phase = float(prev_audio.get("end_phase_rad", 0.0))
        init_rpm = float(prev_audio.get("end_rpm_hz", 85.0))
        audio_theme = _classify_audio_theme(prompt.lower())
        temp_wav = str(out_dir / f"{chunk_id}_audio.wav")
        has_audio, end_phase, end_rpm = runner._synthesize_stereo_audio(
            audio_theme, act_dur, temp_wav, init_phase_rad=init_phase, init_rpm_hz=init_rpm
        )

        diffusion_frames = None
        used_engine = "local_fallback"
        if cloud_mode != "off":
            diffusion_frames, eng = diff_client.synthesize_chunk_frames(
                prompt=act_prompt,
                anime_style=anime_style,
                model_name=act_model,
                seed=seed_base + a_idx * 97,
                total_frames=total_frames,
                duration_sec=act_dur,
                anchor_bgr=prev_last_bgr if is_extended else None,
                is_extended=is_extended,
            )
            if eng:
                used_engine = eng

        engines_used.append(used_engine)
        proc = runner._open_ffmpeg_pipe(width, height, fps, temp_wav if has_audio else None, str(act_mp4))

        def _write_frame(_i: int, frame: Any) -> None:
            if proc.stdin:
                proc.stdin.write(frame.tobytes())

        try:
            if diffusion_frames:
                state = stream_diffusion_frames(
                    diffusion_frames=diffusion_frames,
                    total_frames=total_frames,
                    width=width,
                    height=height,
                    seq_idx=a_idx,
                    is_chapter_start=(a_idx == 0 and len(plan["chapters"]) > 1),
                    is_chapter_end=(a_idx == len(chapter["acts"]) - 1 and len(plan["chapters"]) > 1),
                    total_chunks=total_chunks,
                    chapter_title=chapter.get("chapter_title"),
                    chapter_subtitle=chapter.get("chapter_subtitle"),
                    gamma_lut=gamma_lut,
                    on_frame=_write_frame,
                )
                state["kinetic_theme"] = f"cloud_diffusion_{used_engine}"
            else:
                renderer = AnimeSceneRenderer(width=width, height=height, fps=fps)
                state = renderer.render_sequence(
                    prompt=prompt,
                    total_frames=total_frames,
                    seq_idx=a_idx,
                    total_chunks=total_chunks,
                    chunk_id=chunk_id,
                    duration_sec=act_dur,
                    model_name=act_model,
                    is_chapter_start=(a_idx == 0 and len(plan["chapters"]) > 1),
                    is_chapter_end=(a_idx == len(chapter["acts"]) - 1 and len(plan["chapters"]) > 1),
                    chapter_title=chapter.get("chapter_title"),
                    chapter_subtitle=chapter.get("chapter_subtitle"),
                    prev_video_code=prev_video_code,
                    gamma_lut=gamma_lut,
                    on_frame=_write_frame,
                )
        finally:
            if proc.stdin:
                proc.stdin.close()
            ret = proc.wait()
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
            if ret != 0:
                raise RuntimeError(f"FFmpeg exited with code {ret} rendering {chunk_id}")

        state.update(
            {
                "gamma": gamma,
                "contrast_gain": gain,
                "audio_theme": audio_theme,
                "audio_end_phase_rad": round(end_phase, 4),
                "audio_end_rpm_hz": round(end_rpm, 2),
            }
        )

        # Extract true final frame + Video-as-Code bundle directly for the next Extend-Clip act
        handoff_dir = out_dir / "handoff"
        handoff_dir.mkdir(exist_ok=True)
        _, prev_video_code = extract_handoff_bundle_from_mp4(
            artifact_path=act_mp4,
            anchor_output_path=handoff_dir / f"anchor_{a_idx + 1}.png",
            code_output_path=handoff_dir / f"code_{a_idx}.json",
            job_id=job_id,
            chunk_index=a_idx,
            chunk_id=chunk_id,
            model_engine=act_model,
            prompt=act_prompt,
            end_timestamp_sec=act["end_sec"],
            total_duration_sec=plan["duration_seconds"],
            master_seed=seed_base,
            character_ids=["default"],
            worker_state=state,
        )
        prev_last_bgr = _extract_last_frame_bgr(act_mp4)
        act_files.append(act_mp4)

    chapter_mp4 = out_dir / f"chapter_{chapter_index:02d}.mp4"
    video_processor.concatenate_video_chunks(act_files, chapter_mp4)

    # Clean intermediate act files cleanly before artifact upload
    for af in act_files:
        if af != chapter_mp4 and af.exists():
            af.unlink()

    meta_path = out_dir / f"chapter_{chapter_index:02d}_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "chapter_index": chapter_index,
                "engines_used": engines_used,
                "duration_sec": sum(float(a["duration_sec"]) for a in chapter["acts"]),
            },
            indent=2,
        )
    )
    logger.info("Chapter %d assembled -> %s (engines=%s)", chapter_index, chapter_mp4, engines_used)
    return chapter_mp4


def _gh_request(url: str, token: str, method: str = "GET", json_body: Any = None, raw_bytes: bytes = None, content_type: str = None) -> Any:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "FreeCloud-Autonomous-Studio",
    }
    data = None
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif raw_bytes is not None:
        data = raw_bytes
        headers["Content-Type"] = content_type or "application/octet-stream"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=180) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
        return json.loads(raw) if raw else {}


def ensure_gallery_release(repo: str, tag: str, token: str) -> Dict[str, Any]:
    """Fetch existing gallery release or create it cleanly."""
    api_base = f"https://api.github.com/repos/{repo}/releases"
    try:
        return _gh_request(f"{api_base}/tags/{tag}", token)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
    initial_body = json.dumps({"schema": "freecloud-gallery/v1", "items": []}, indent=2)
    return _gh_request(
        api_base,
        token,
        method="POST",
        json_body={
            "tag_name": tag,
            "name": "FreeCloud AI Studio — 24/7 Cloud Video Gallery",
            "body": initial_body,
            "draft": False,
            "prerelease": False,
        },
    )


def upload_release_asset(release: Dict[str, Any], file_path: Path, content_type: str, token: str) -> str:
    """Upload asset to GitHub Release, replacing any same-named previous asset cleanly."""
    fname = file_path.name
    for asset in release.get("assets", []):
        if asset.get("name") == fname and asset.get("url"):
            _gh_request(asset["url"], token, method="DELETE")

    upload_url = release["upload_url"].split("{")[0] + "?name=" + urllib.parse.quote(fname)
    resp = _gh_request(
        upload_url,
        token,
        method="POST",
        raw_bytes=file_path.read_bytes(),
        content_type=content_type,
    )
    return resp["browser_download_url"]


def parse_release_catalog(body_text: str) -> List[Dict[str, Any]]:
    """Extract clean gallery items list from GitHub Release body JSON."""
    if not body_text:
        return []
    try:
        # Handle raw JSON or JSON fenced block
        cleaned = body_text.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
        data = json.loads(cleaned)
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            return data["items"]
    except Exception:
        pass
    return []


def publish_to_github_gallery(
    repo: str,
    tag: str,
    token: str,
    job_id: str,
    video_path: Path,
    poster_path: Optional[Path],
    prompt: str,
    duration_seconds: float,
    resolution: str,
    model_name: str,
    anime_style: str,
    engines_used: List[str],
    run_url: str = "",
) -> Dict[str, Any]:
    """Publish rendered 1080p MP4 + poster + catalog entry into the 24/7 GitHub Releases gallery."""
    release = ensure_gallery_release(repo, tag, token)
    mp4_url = upload_release_asset(release, video_path, "video/mp4", token)
    poster_url = ""
    if poster_path and poster_path.is_file():
        poster_url = upload_release_asset(release, poster_path, "image/jpeg", token)

    items = [item for item in parse_release_catalog(release.get("body", "")) if item.get("job_id") != job_id]
    new_entry = {
        "job_id": job_id,
        "prompt": prompt[:240],
        "duration_seconds": round(float(duration_seconds), 2),
        "resolution": resolution,
        "model_name": model_name,
        "anime_style": anime_style,
        "engines_used": engines_used,
        "video_url": mp4_url,
        "poster_url": poster_url,
        "size_mb": round(video_path.stat().st_size / (1024 * 1024), 2),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_url": run_url,
    }
    items.insert(0, new_entry)
    items = items[:40]

    new_body = json.dumps({"schema": "freecloud-gallery/v1", "items": items}, indent=2)
    _gh_request(release["url"], token, method="PATCH", json_body={"body": new_body})
    logger.info("Published %s to 24/7 GitHub Release CDN: %s", job_id, mp4_url)
    return new_entry


def main() -> None:
    parser = argparse.ArgumentParser(description="FreeCloud Autonomous 24/7 Cloud Studio Pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--job-id", default="")
    p_plan.add_argument("--prompt", required=True)
    p_plan.add_argument("--duration-seconds", type=float, default=24.0)
    p_plan.add_argument("--model", default="auto_relay")
    p_plan.add_argument("--anime-style", default="wan_2_1_realistic")
    p_plan.add_argument("--resolution", default="1920x1080")
    p_plan.add_argument("--plan-out", default="/tmp/cloud_plan.json")
    p_plan.add_argument("--github-output", default=os.getenv("GITHUB_OUTPUT", ""))

    p_ch = sub.add_parser("render-chapter")
    p_ch.add_argument("--plan-json", required=True)
    p_ch.add_argument("--chapter-index", type=int, required=True)
    p_ch.add_argument("--out-dir", required=True)

    p_pub = sub.add_parser("publish-release")
    p_pub.add_argument("--plan-json", required=True)
    p_pub.add_argument("--chapters-dir", required=True)
    p_pub.add_argument("--repo", default=os.getenv("GITHUB_REPOSITORY", "pratham31sinha-boop/FreeCloud-AI-Studio"))
    p_pub.add_argument("--tag", default="cloud-gallery")
    p_pub.add_argument("--run-url", default="")

    args = parser.parse_args()

    if args.cmd == "plan":
        jid = (args.job_id or "").strip() or f"fc_cloud_{hashlib.sha256(f'{time.time()}_{args.prompt}'.encode()).hexdigest()[:10]}"
        w_str, h_str = (args.resolution or "1920x1080").lower().split("x")
        plan = build_autonomous_plan(
            job_id=jid,
            prompt=args.prompt,
            duration_seconds=args.duration_seconds,
            model_name=args.model,
            anime_style=args.anime_style,
            width=int(w_str),
            height=int(h_str),
        )
        Path(args.plan_out).write_text(json.dumps(plan, indent=2))
        indices = [c["chapter_index"] for c in plan["chapters"]]
        if args.github_output:
            with open(args.github_output, "a", encoding="utf-8") as gh_out:
                gh_out.write(f"job_id={jid}\n")
                gh_out.write(f"matrix={json.dumps(indices)}\n")
        print(json.dumps({"job_id": jid, "chapters": len(indices), "acts": sum(len(c["acts"]) for c in plan["chapters"])}))

    elif args.cmd == "render-chapter":
        plan = json.loads(Path(args.plan_json).read_text())
        render_chapter_autonomous(plan, args.chapter_index, Path(args.out_dir))

    elif args.cmd == "publish-release":
        import cv2
        from backend.storage.video_processor import video_processor

        plan = json.loads(Path(args.plan_json).read_text())
        cdir = Path(args.chapters_dir)
        chapter_mp4s = sorted(cdir.rglob("chapter_*.mp4"))
        if not chapter_mp4s:
            raise RuntimeError(f"No rendered chapter MP4s found in {cdir}")

        engines_used: List[str] = []
        for mp in sorted(cdir.rglob("chapter_*_meta.json")):
            meta = json.loads(mp.read_text())
            engines_used.extend(meta.get("engines_used", []))

        job_id = plan["job_id"]
        final_mp4 = cdir / f"{job_id}_1080p.mp4"
        video_processor.concatenate_video_chunks(chapter_mp4s, final_mp4)

        poster_jpg = cdir / f"{job_id}_poster.jpg"
        cap = cv2.VideoCapture(str(final_mp4))
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(total_f - 1, total_f // 3)))
        ok, frm = cap.read()
        cap.release()
        if ok and frm is not None:
            thumb = cv2.resize(frm, (960, 540), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(poster_jpg), thumb, [cv2.IMWRITE_JPEG_QUALITY, 90])

        token = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PAT_TOKEN") or ""
        if not token:
            raise RuntimeError("GITHUB_TOKEN or GITHUB_PAT_TOKEN required to publish Release assets")

        entry = publish_to_github_gallery(
            repo=args.repo,
            tag=args.tag,
            token=token,
            job_id=job_id,
            video_path=final_mp4,
            poster_path=poster_jpg if poster_jpg.is_file() else None,
            prompt=plan["prompt"],
            duration_seconds=plan["duration_seconds"],
            resolution=f"{plan['width']}x{plan['height']}",
            model_name=plan["model_name"],
            anime_style=plan["anime_style"],
            engines_used=engines_used,
            run_url=args.run_url,
        )
        summary_path = os.getenv("GITHUB_STEP_SUMMARY")
        if summary_path:
            with open(summary_path, "a", encoding="utf-8") as sf:
                sf.write(f"## FreeCloud AI Studio — 24/7 Cloud Render Completed\n")
                sf.write(f"- **Job ID:** `{job_id}`\n")
                sf.write(f"- **Video CDN:** {entry['video_url']}\n")
                sf.write(f"- **Engines:** `{', '.join(engines_used)}`\n")
        print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
