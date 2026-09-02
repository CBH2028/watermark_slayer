"""Generate a self-contained GitHub star-history chart for the README."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any


API_ROOT = "https://api.github.com"
STAR_ACCEPT = "application/vnd.github.star+json"


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def request_json(url: str, token: str, accept: str) -> tuple[Any, str]:
    headers = {
        "Accept": accept,
        "User-Agent": "pdf-size-reducer-star-chart",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
            return payload, response.headers.get("Link", "")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API returned HTTP {exc.code}: {detail}") from exc


def next_link(link_header: str) -> str | None:
    for part in link_header.split(","):
        if 'rel="next"' not in part:
            continue
        match = re.search(r"<([^>]+)>", part)
        if match:
            return match.group(1)
    return None


def load_history(repository: str, token: str) -> tuple[dict[str, Any], list[datetime]]:
    repo, _ = request_json(
        f"{API_ROOT}/repos/{repository}", token, "application/vnd.github+json"
    )
    url = f"{API_ROOT}/repos/{repository}/stargazers?per_page=100"
    starred_at: list[datetime] = []
    while url:
        page, links = request_json(url, token, STAR_ACCEPT)
        for item in page:
            timestamp = item.get("starred_at")
            if timestamp:
                starred_at.append(parse_timestamp(timestamp))
        url = next_link(links)
    starred_at.sort()
    return repo, starred_at


def nice_ceiling(value: int) -> int:
    if value <= 5:
        return max(1, value)
    magnitude = 10 ** (len(str(value)) - 1)
    for multiplier in (1, 2, 5, 10):
        ceiling = multiplier * magnitude
        if ceiling >= value:
            return ceiling
    return value


def make_svg(
    repository: str,
    created_at: datetime,
    stars: list[datetime],
    generated_on: datetime,
) -> str:
    width, height = 960, 460
    left, right, top, bottom = 78, 46, 132, 66
    chart_width = width - left - right
    chart_height = height - top - bottom

    day_end = datetime.combine(
        generated_on.astimezone(timezone.utc).date(),
        time(23, 59, 59),
        tzinfo=timezone.utc,
    )
    start = min(created_at, stars[0] if stars else created_at)
    if day_end - start < timedelta(days=6):
        start = day_end - timedelta(days=6)
    span = max((day_end - start).total_seconds(), 1.0)
    y_max = nice_ceiling(len(stars))

    def x_for(moment: datetime) -> float:
        ratio = (moment - start).total_seconds() / span
        return left + max(0.0, min(1.0, ratio)) * chart_width

    def y_for(count: int) -> float:
        return top + chart_height - (count / y_max) * chart_height

    points: list[tuple[float, float]] = [(left, y_for(0))]
    count = 0
    for timestamp in stars:
        count += 1
        x = x_for(timestamp)
        points.append((x, y_for(count - 1)))
        points.append((x, y_for(count)))
    points.append((left + chart_width, y_for(count)))
    line_path = " ".join(
        ("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}"
        for index, (x, y) in enumerate(points)
    )
    area_path = (
        line_path
        + f" L {left + chart_width:.2f} {top + chart_height:.2f}"
        + f" L {left:.2f} {top + chart_height:.2f} Z"
    )

    grid: list[str] = []
    if y_max <= 5:
        y_ticks = list(range(y_max, -1, -1))
    else:
        y_ticks = [round(y_max * (1 - index / 4)) for index in range(5)]
    for value in y_ticks:
        y = y_for(value)
        grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + chart_width}" '
            f'y2="{y:.2f}" stroke="#E7E7F1" stroke-width="1"/>'
        )
        grid.append(
            f'<text x="{left - 18}" y="{y + 5:.2f}" text-anchor="end" '
            f'class="axis">{value}</text>'
        )
    for index in range(5):
        ratio = index / 4
        moment = start + timedelta(seconds=span * ratio)
        x = left + chart_width * ratio
        anchor = "start" if index == 0 else "end" if index == 4 else "middle"
        grid.append(
            f'<text x="{x:.2f}" y="{top + chart_height + 34}" '
            f'text-anchor="{anchor}" class="axis">{moment:%b %d}</text>'
        )

    safe_repo = html.escape(repository)
    total = len(stars)
    final_x, final_y = points[-1]
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="960" height="460" viewBox="0 0 960 460" role="img" aria-labelledby="title desc">
  <title id="title">GitHub star history for {safe_repo}</title>
  <desc id="desc">{total} current stars, generated from the official GitHub Stargazers API.</desc>
  <defs>
    <linearGradient id="card" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#FFFFFF"/>
      <stop offset="1" stop-color="#F7F6FF"/>
    </linearGradient>
    <linearGradient id="area" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#635BFF" stop-opacity="0.28"/>
      <stop offset="1" stop-color="#635BFF" stop-opacity="0.02"/>
    </linearGradient>
    <linearGradient id="line" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#827CFF"/>
      <stop offset="1" stop-color="#5149E8"/>
    </linearGradient>
    <filter id="shadow" x="-10%" y="-15%" width="120%" height="140%">
      <feDropShadow dx="0" dy="8" stdDeviation="12" flood-color="#3E3A70" flood-opacity="0.12"/>
    </filter>
    <style>
      text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; }}
      .axis {{ fill: #777782; font-size: 13px; }}
    </style>
  </defs>
  <rect x="14" y="14" width="932" height="428" rx="26" fill="url(#card)" stroke="#E4E2F5" filter="url(#shadow)"/>
  <circle cx="56" cy="56" r="20" fill="#EFEEFF"/>
  <path d="M56 43.8l3.7 7.5 8.3 1.2-6 5.8 1.4 8.2-7.4-3.9-7.4 3.9 1.4-8.2-6-5.8 8.3-1.2z" fill="#635BFF"/>
  <text x="88" y="52" fill="#202024" font-size="20" font-weight="700">GitHub Star History</text>
  <text x="88" y="75" fill="#777782" font-size="13">{safe_repo}</text>
  <text x="904" y="55" text-anchor="end" fill="#202024" font-size="28" font-weight="750">{total}</text>
  <text x="904" y="76" text-anchor="end" fill="#777782" font-size="12">CURRENT STARS</text>
  {''.join(grid)}
  <path d="{area_path}" fill="url(#area)"/>
  <path d="{line_path}" fill="none" stroke="url(#line)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="{final_x:.2f}" cy="{final_y:.2f}" r="7" fill="#FFFFFF" stroke="#635BFF" stroke-width="4"/>
  <text x="78" y="421" fill="#92929D" font-size="11">OFFICIAL GITHUB STARGAZERS API</text>
  <text x="882" y="421" text-anchor="end" fill="#92929D" font-size="11">UPDATED {day_end:%Y-%m-%d} UTC</text>
</svg>
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository",
        default=os.environ.get("GITHUB_REPOSITORY", "CBH2028/watermark_slayer"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/images/star-history.svg")
    )
    args = parser.parse_args()
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN", "")
    repo, stars = load_history(args.repository, token)
    svg = make_svg(
        args.repository,
        parse_timestamp(repo["created_at"]),
        stars,
        datetime.now(timezone.utc),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8", newline="\n")
    print(f"Wrote {args.output} with {len(stars)} stars")


if __name__ == "__main__":
    main()
