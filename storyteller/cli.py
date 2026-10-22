from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ROOT, load_book_config, load_config
from .pipeline import run
from .project import approve, create, load, retry
from .report import summarize, write_report
from .runtime import StageRunner, prerequisites
from .storage import FileLock, contained, file_hash, read_json


def parser():
    cli = argparse.ArgumentParser(prog="shuoshu", description="说书人：离线克隆有声书批处理")
    cli.add_argument("--config", type=Path, help="配置 JSON 路径，默认 config.local.json")
    commands = cli.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="导入 TXT 与参考录音，创建项目")
    init.add_argument("book", type=Path)
    init.add_argument("--voice", required=True, type=Path)
    init.add_argument("--out", required=True, type=Path)
    init.add_argument("--encoding")
    init.add_argument("--pronunciations", type=Path)
    init.add_argument("--reference-start", type=float, default=0, help="参考录音截取起点，最多使用 15 秒")
    init.add_argument("--chapter-pattern", help="自定义章节标题正则，匹配整行")
    for name, description in (("run", "生成、校对、自动重试并导出；中断后仍用此命令"),
                              ("status", "查看进度"), ("report", "刷新本地复听报告")):
        sub = commands.add_parser(name, help=description)
        sub.add_argument("project", type=Path)
    trial = commands.add_parser("preview", help="生成开头短试读，确认音色前不制作全书")
    trial.add_argument("project", type=Path)
    confirmation = commands.add_parser("confirm", help="检查目录或听完试读后确认")
    confirmation.add_argument("project", type=Path)
    confirmation.add_argument("--chapters", action="store_true")
    confirmation.add_argument("--voice", action="store_true")
    confirmation.add_argument("--note", default="已检查并确认")
    plan = commands.add_parser("plan", help="制作前调整章节；不更改原文")
    plan.add_argument("project", type=Path)
    group = plan.add_mutually_exclusive_group()
    group.add_argument("--chapter-pattern")
    group.add_argument("--starts", help="章节标题所在原稿行号，逗号分隔；空字符串表示整本一章")
    studio = commands.add_parser("studio", help="打开本地制作面板：章节、试读、生成、复听和纠错")
    studio.add_argument("project", type=Path)
    studio.add_argument("--port", type=int, default=0)
    studio.add_argument("--no-browser", action="store_true")
    again = commands.add_parser("retry", help="只重做指定片段，然后用 run 继续")
    again.add_argument("project", type=Path)
    again.add_argument("segments", nargs="+")
    accept = commands.add_parser("approve", help="人工复听后接受可疑片段，然后用 run 导出")
    accept.add_argument("project", type=Path)
    accept.add_argument("segments", nargs="+")
    accept.add_argument("--note", default="人工复听确认通过")
    doctor = commands.add_parser("doctor", help="检查环境；完全离线，不下载文件")
    doctor.add_argument("--load-models", action="store_true", help="依次加载 TTS 和 ASR 验证离线启动")
    doctor.add_argument("--verify-hashes", action="store_true", help="校验全部模型文件 SHA-256，可能耗时较长")
    return cli


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        cfg = load_book_config(args.project.resolve(), args.config) if hasattr(args, "project") else load_config(args.config)
        if args.command == "init":
            project = create(args.book.resolve(), args.voice.resolve(), args.out.resolve(), cfg,
                             args.encoding, args.pronunciations, args.reference_start, args.chapter_pattern)
            write_report(args.out.resolve(), project)
            print(f"已导入 {len(project['chapters'])} 章，{len(project['segments'])} 段。")
            print(f"参考录音已转为单声道 WAV，使用起点 {args.reference_start:g} 秒后的最多 15 秒。")
            print("已保存本书制作配置。请先检查章节并生成试读，确认后开始全书。")
            print(f'打开制作面板：.\\run.ps1 studio "{args.out.resolve()}"')
            print(f'命令行试读：.\\run.ps1 preview "{args.out.resolve()}"')
        elif args.command == "run":
            from .preparation import require_confirmation
            require_confirmation(args.project.resolve())
            summary = run(args.project, cfg)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            print(f"复听报告：{args.project.resolve() / 'review.html'}")
            return 2 if summary["review_segments"] else 0
        elif args.command == "preview":
            from .preparation import preview
            preview(args.project.resolve(), cfg)
            project = load(args.project.resolve())
            sample = project["preflight"]["preview"]
            print(f"试读音频：{args.project.resolve() / sample['audio']}")
            print(f'听完后确认：.\\run.ps1 confirm "{args.project.resolve()}" --voice --chapters')
            return 2 if any(s["state"] in {"review", "error"} for s in project["segments"] if s["id"] in sample["segments"]) else 0
        elif args.command == "confirm":
            from .preparation import confirm
            result = confirm(args.project.resolve(), chapters=args.chapters, voice=args.voice, note=args.note)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "plan":
            from .preparation import replan
            starts = [int(item.strip()) for item in args.starts.split(",") if item.strip()] if args.starts is not None else None
            project = replan(args.project.resolve(), cfg, pattern=args.chapter_pattern, starts=starts)
            write_report(args.project.resolve(), project)
            print(f"章节已更新：{len(project['chapters'])} 章，{len(project['segments'])} 段；请重新确认。")
        elif args.command == "studio":
            from .studio import serve
            serve(args.project.resolve(), args.config, args.port, not args.no_browser)
        elif args.command == "status":
            summary = summarize(load(args.project.resolve()))
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        elif args.command in {"retry", "approve", "report"}:
            root = args.project.resolve()
            with FileLock(root / ".project.lock"):
                if args.command == "retry":
                    retry(root, args.segments)
                elif args.command == "approve":
                    approve(root, args.segments, args.note)
                summary = write_report(root, load(root))
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        elif args.command == "doctor":
            missing = prerequisites(cfg)
            if missing:
                print("尚未就绪：\n" + "\n".join("- " + item for item in missing))
                return 1
            if args.verify_hashes:
                for key in ("index_model", "asr_model", "asr_vad_model"):
                    model = Path(cfg[key])
                    for entry in read_json(model / "asset-manifest.json")["files"]:
                        if file_hash(contained(model, entry["path"])) != entry["sha256"]:
                            raise RuntimeError(f"模型校验失败：{key}/{entry['path']}")
            print("文件检查通过。")
            if args.load_models:
                root = ROOT / ".runtime/doctor"
                root.mkdir(parents=True, exist_ok=True)
                with FileLock(ROOT / ".runtime/gpu.lock"):
                    runner = StageRunner(cfg)
                    for stage in ("synth", "check"):
                        ready = False
                        for event in runner.events(stage, root, []):
                            ready |= event["event"] == "ready"
                        if not ready:
                            raise RuntimeError(f"{stage} 未返回模型就绪信号。")
                        print(f"{stage} 离线加载成功，进程已退出释放显存。")
            else:
                print("尚未加载模型验证。使用 doctor --load-models 检查真实离线启动。")
        return 0
    except KeyboardInterrupt:
        print("\n任务已中断，已保存进度。使用同一 run 命令续跑。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
