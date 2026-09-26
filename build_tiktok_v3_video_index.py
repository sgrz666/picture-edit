#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Build TikTok V3 video index

Inspired by large-scale video dataset preprocessing:
HumanVid / Motion-X style pipeline.

Only build metadata.
No frame extraction.
No filtering.
"""


from pathlib import Path
import json
import subprocess
from tqdm import tqdm


RAW_ROOT = Path(
    "datasets/TikTokDataset/TikTok_Raw_Videos"
)

OUT = Path(
    "datasets/TikTokDataset/TikTok_v3/manifests/videos.jsonl"
)


def probe_video(path):

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path)
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    info = json.loads(result.stdout)


    video_stream = None

    for s in info["streams"]:
        if s["codec_type"]=="video":
            video_stream=s
            break


    if video_stream is None:
        return None


    width = video_stream.get("width")
    height = video_stream.get("height")


    fps_str = video_stream.get(
        "avg_frame_rate",
        "0/1"
    )


    try:
        a,b=fps_str.split("/")
        fps=float(a)/float(b)
    except:
        fps=0


    frames = video_stream.get(
        "nb_frames",
        None
    )


    duration=float(
        info["format"].get(
            "duration",
            0
        )
    )


    return {

        "fps":fps,

        "frames":
            int(frames)
            if frames
            else None,

        "width":width,

        "height":height,

        "duration":duration

    }



def main():

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    videos=[]


    for p in RAW_ROOT.rglob("*"):

        if p.suffix.lower() in [
            ".mp4",
            ".mov",
            ".avi"
        ]:
            videos.append(p)


    print(
        "videos:",
        len(videos)
    )


    with open(
        OUT,
        "w"
    ) as f:


        for p in tqdm(videos):

            meta=probe_video(p)


            if meta is None:
                continue


            row={

                "video_id":
                    p.stem,

                "video_path":
                    str(p),

                **meta

            }


            f.write(
                json.dumps(
                    row
                )
                + "\n"
            )


    print()
    print(
        "saved:",
        OUT
    )



if __name__=="__main__":
    main()
