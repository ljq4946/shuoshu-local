from __future__ import annotations

import csv
import html
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from .storage import write_json

LABELS = {"pending": "等待生成", "synthesizing": "正在生成", "generated": "等待校对",
          "checking": "正在校对", "passed": "机器校验通过", "approved": "人工复听通过",
          "review": "需要复听", "error": "生成失败", "pause": "场景停顿（保留原文，不合成）"}


def summarize(project: dict) -> dict:
    """Build status from one saved snapshot without writing project files."""
    counts = Counter(s["state"] for s in project["segments"])
    duration = sum((s.get("check") or {}).get("duration", 0) for s in project["segments"])
    processing = sum(run.get("elapsed_seconds", 0) for run in project["runs"])
    issues = [s for s in project["segments"] if s["state"] in {"review", "error"}]
    historical_segments = project["segments"] + project.get("retired_segments", [])
    ever_review = sum(bool(s.get("ever_required_review") or s.get("approval") or s["state"] in {"review", "error"})
                      for s in historical_segments)
    exported_duration = sum(item.get("duration", 0) for item in project["exports"].values())
    summary = {"title": project["title"], "segments": len(project["segments"]), "states": dict(counts),
               "audio_hours": round(duration / 3600, 4), "processing_hours": round(processing / 3600, 4),
               "processing_hours_per_audio_hour": round(processing / duration, 3) if duration else None,
               "review_segments": len(issues),
               "review_segments_per_audio_hour": round(len(issues) * 3600 / duration, 2) if duration else None,
               "manual_review_segments": ever_review,
               "manual_review_segments_per_audio_hour": round(ever_review * 3600 / duration, 2) if duration else None,
               "manual_actions": sum(len(s.get("manual_actions", [])) for s in historical_segments),
               "exported_audio_hours": round(exported_duration / 3600, 4),
               "total_synthesis_attempts": sum(s["total_attempts"] for s in historical_segments),
               "ready_chapters": len(project["exports"]), "chapters": len(project["chapters"])}
    summary["preflight"] = project.get("preflight", {"chapters_confirmed": True, "voice_confirmed": True, "legacy": True})
    return summary


def write_report(root: Path, project: dict) -> dict:
    summary = summarize(project)
    duration = sum((s.get("check") or {}).get("duration", 0) for s in project["segments"])
    issues = [s for s in project["segments"] if s["state"] in {"review", "error"}]
    ever_review = summary['manual_review_segments']
    exported_duration = sum(item.get("duration", 0) for item in project["exports"].values())
    write_json(root / "summary.json", summary)
    with (root / "review.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["片段", "状态", "原文", "识别结果", "原因", "音频"])
        for segment in issues:
            check = segment.get("check") or {}
            row = [segment["id"], LABELS[segment["state"]], segment["text"], check.get("recognized", ""),
                   "；".join(check.get("reasons", [])) or segment.get("error", ""), segment["audio"] or ""]
            # Neutralize spreadsheet formula cells in a user-supplied manuscript.
            writer.writerow(["'" + item if item.startswith(("=", "+", "-", "@")) else item for item in row])
    esc = html.escape
    cards = []
    for segment in sorted(project["segments"], key=lambda s: s["state"] not in {"review", "error"}):
        check = segment.get("check") or {}
        reasons = "；".join(check.get("reasons", [])) or segment.get("error", "")
        audio = f'<audio controls preload="none" src="{quote(segment["audio"], safe="/")}"></audio>' if segment["audio"] else ""
        quoted_root = "'" + str(root).replace("'", "''") + "'"
        command = f'.\\run.ps1 approve {quoted_root} {segment["id"]}'
        redo = f'.\\run.ps1 retry {quoted_root} {segment["id"]}'
        cards.append(f'<article class="{segment["state"]}"><header><b>{segment["id"]}</b><span>{LABELS[segment["state"]]}</span></header>'
                     f'<p>{esc(segment["text"])}</p><p class="muted">识别：{esc(check.get("recognized", "尚未校对"))}</p>'
                     f'<p class="reason">{esc(reasons)}</p>{audio}<details><summary>复听后的操作与生成记录</summary>'
                     f'<p>确认读对了：<code>{esc(command)}</code></p><p>重新生成：<code>{esc(redo)}</code></p>'
                     f'<p>累计生成 {segment["total_attempts"]} 次；文字差异率 {check.get("cer", 0):.1%}</p></details></article>')
    links = []
    for chapter in project["chapters"]:
        output = project["exports"].get(chapter["id"])
        if output:
            links.append(f'<li>{esc(chapter["title"])} · <a href="exports/{quote(output["mp3"])}">MP3</a> · '
                         f'<a href="exports/{quote(output["wav"])}">WAV 母版</a></li>')
        else:
            links.append(f'<li>{esc(chapter["title"])} · 尚未导出</li>')
    document = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(project['title'])} · 说书人</title><style>
body{{max-width:1000px;margin:36px auto;padding:0 22px;background:#f6f4ee;color:#24332f;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}}
h1{{font-size:32px;margin-bottom:4px}}.muted{{color:#65716a}}.metrics{{display:flex;gap:16px;flex-wrap:wrap;margin:24px 0}}
.metrics div{{background:#e4ebe2;padding:14px 24px;border-radius:12px}}article{{padding:22px;background:#fff;border:1px solid #e0e4dc;border-radius:12px;margin:16px 0}}
header{{display:flex;justify-content:space-between;gap:20px}}header span{{color:#65716a}}.review,.error{{border-left:5px solid #c57b2d}}
.reason{{color:#9c5418}}audio{{width:100%;margin:8px 0}}code{{overflow-wrap:anywhere;font-size:13px}}details{{font-size:14px}}
a{{color:#216955}}button,input{{font:inherit;padding:9px 12px;border:1px solid #adb8ae;border-radius:8px;background:white}}input{{width:min(90%,450px)}}
body.issues article:not(.review):not(.error){{display:none}}.hidden{{display:none!important}}</style>
<h1>{esc(project['title'])}</h1><p class="muted">说书人 · 本地制作报告 · 所有音频留在此项目目录</p>
<p>需要直接批准、重做、编辑读音或查看实时进度：运行 <code>.\\run.ps1 studio {esc(chr(39) + str(root).replace(chr(39), chr(39)*2) + chr(39))}</code> 打开本地制作面板。</p>
<div class="metrics"><div>{len(project['segments'])} 个片段</div><div>{duration / 60:.1f} 分钟音频</div><div>{len(issues)} 段待复听</div><div>{len(project['exports'])}/{len(project['chapters'])} 章已导出</div></div>
<p>机器校验用于筛查可疑漏读、重复和截断，不能保证读音、音色与情绪正确。仅通过校验或人工复听的完整章节会导出。</p>
<p class="muted">累计有 {ever_review} 段需要人工处理，完成复听后仍计入统计。已导出 {exported_duration / 60:.1f} 分钟；上方音频时长也包含尚待复听的片段。</p>
<ul>{''.join(links)}</ul><p><button id="issues">只看异常</button> <input id="search" aria-label="搜索片段" placeholder="搜索片段编号或文字"></p>
{''.join(cards)}<script>
document.getElementById('issues').onclick=function(){{document.body.classList.toggle('issues');this.textContent=document.body.classList.contains('issues')?'显示全部':'只看异常'}};
document.getElementById('search').oninput=function(){{const q=this.value.toLowerCase();document.querySelectorAll('article').forEach(a=>a.classList.toggle('hidden',!a.textContent.toLowerCase().includes(q)))}};
</script></html>'''
    (root / "review.html").write_text(document, encoding="utf-8")
    return summary
