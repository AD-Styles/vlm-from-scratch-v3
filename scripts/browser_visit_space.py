"""실제 Chromium 브라우저로 라이브 HF Space 방문 + 응답 캡처 + 스크린샷."""
import sys, io, time, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
from playwright.sync_api import sync_playwright

# HF Spaces 는 iframe 안에 gradio 가 embed 됨 → 직접 gradio host 로 navigate (iframe 회피)
SPACE_URL = "https://ad-styles-mini-llava-v3-demo.hf.space"
ASSETS = Path("assets").resolve()

TEST_CASES = [
    # English → English
    (str(ASSETS / "source_dog.jpg"), "Is there a cat in the image?", "no"),
    (str(ASSETS / "source_dog.jpg"), "What color is the main subject?", "white"),
    (str(ASSETS / "source_pikachu.png"), "What color is this character?", "yellow"),
    # Korean → Korean (m2m100)
    (str(ASSETS / "source_dog.jpg"), "이 동물의 종류는 무엇인가요?", "개"),
    (str(ASSETS / "source_dog.jpg"), "이 이미지에 고양이가 있나요?", "아니요"),
    (str(ASSETS / "source_dog.jpg"), "주요 피사체의 색상은 무엇인가요?", "흰색"),
    (str(ASSETS / "source_pikachu.png"), "이 캐릭터의 색은 무엇인가요?", "노란색"),
]

OUT_DIR = Path("eval_results/browser_screenshots")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print(f"Visit: {SPACE_URL}")
    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 1800})
        page = context.new_page()

        # Direct gradio host — no iframe wrapper
        print(f"  loading {SPACE_URL} (domcontentloaded) ...")
        page.goto(SPACE_URL, wait_until="domcontentloaded", timeout=60000)
        print("  page loaded, waiting 30s for Gradio JS init ...")
        time.sleep(30)
        gradio_frame = page.main_frame  # No iframe needed
        print(f"  using main frame: {gradio_frame.url}")
        page.screenshot(path=str(OUT_DIR / "00_initial.png"), full_page=True)

        for i, (img_path, question, expected) in enumerate(TEST_CASES, 1):
            print(f"\n--- Case {i}: {Path(img_path).stem} | {question[:40]} ---")
            try:
                file_input = gradio_frame.locator('input[type="file"]').first
                file_input.set_input_files(img_path)
                time.sleep(3)
                print(f"    image uploaded")

                textbox = gradio_frame.locator('textarea').first
                textbox.fill(question)
                time.sleep(1)
                print(f"    question entered")

                # 응답 버튼 — 텍스트로 찾거나 primary class
                try:
                    btn = gradio_frame.get_by_role("button", name="🚀 응답 생성").first
                    btn.click(timeout=5000)
                except Exception:
                    btns = gradio_frame.locator('button').all()
                    for b in btns:
                        txt = b.inner_text().strip()
                        if "응답" in txt or "Submit" in txt or "Run" in txt:
                            b.click()
                            break
                # m2m100 사전 로드됨 → 모든 inference 동일 시간
                wait_s = 40
                print(f"    clicked submit, waiting {wait_s}s for inference ...")
                time.sleep(wait_s)

                # 응답 textarea 들 모두 확인
                tas = gradio_frame.locator('textarea').all()
                output_text = ""
                for j, t in enumerate(tas):
                    val = t.input_value()
                    if val and val.strip() != question.strip():
                        if val.strip() not in {"", "⚠️ 이미지를 먼저 업로드해 주세요.", "⚠️ 질문을 입력해 주세요."}:
                            output_text = val
                            break

                screenshot_path = OUT_DIR / f"{i:02d}_case_{i}.png"
                page.screenshot(path=str(screenshot_path), full_page=True)

                print(f"    UI 응답: {output_text[:150]!r}")
                print(f"    screenshot: {screenshot_path.name}")

                results.append({
                    "case": i,
                    "image": Path(img_path).name,
                    "question": question,
                    "expected": expected,
                    "ui_response": output_text,
                    "screenshot": str(screenshot_path),
                })

                # reset by reload
                page.reload(wait_until="domcontentloaded", timeout=30000)
                time.sleep(15)
                gradio_frame = page.main_frame  # main frame 재할당

            except Exception as e:
                print(f"    [ERROR] {type(e).__name__}: {e}")
                results.append({
                    "case": i,
                    "image": Path(img_path).name,
                    "question": question,
                    "expected": expected,
                    "ui_response": f"[error: {e}]",
                    "screenshot": None,
                })

        browser.close()

    out_json = OUT_DIR / "browser_test_results.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {out_json}")
    print("\n=== 요약 ===")
    for r in results:
        match = "✅" if r["expected"].lower().strip() in r["ui_response"].lower() else "❌"
        print(f"  {match} case {r['case']}: expected '{r['expected']}' → got '{r['ui_response'][:60]}'")


if __name__ == "__main__":
    main()
