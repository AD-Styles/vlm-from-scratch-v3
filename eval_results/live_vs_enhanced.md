# 라이브 Space vs 로컬 Enhanced — 실제 응답 비교

사용자 challenge: 수치만 말하지 말고 실제 사용해보고 답변 확인.

테스트 방법:
- 라이브 Space (baseline): https://huggingface.co/spaces/AD-Styles/mini-llava-v3-demo (gradio_client API)
- 로컬 Enhanced: src/enhanced_inference.py (CLIP grounding + extraction + OOD gate)
- 같은 이미지, 같은 질문 → raw 응답 그대로 비교

| # | 이미지 | 질문 | 기대 | LIVE Space (baseline) | Enhanced (local) | path |
|---|---|---|---|---|---|---|
| 1 | source_dog | What is in this image? | dog | Dog | Dog | vlm_raw |
| 2 | source_dog | Is there a dog in the image? | yes | yes | yes | clip_grounding_yesno |
| 3 | source_dog | Is there a cat in the image? | no | no | no | clip_grounding_yesno |
| 4 | source_dog | Is there a person in the image? | no | no | no | clip_grounding_yesno |
| 5 | source_dog | Is there a car in the image? | no | no | no | clip_grounding_yesno |
| 6 | source_dog | What color is the main subject? | white | white | white | clip_color |
| 7 | source_dog | 이 이미지에 무엇이 보이나요? | 개 | 이 그림에서 우리는 개를 볼 수 있으며 배경은 흰색입니다. | 이 그림에서 우리는 개를 볼 수 있으며 배경은 흰색입니다. | vlm_raw |
| 8 | source_dog | 이 동물의 종류는 무엇인가요? | 개 | 개 | 개 | vlm_raw |
| 9 | source_pikachu | What is in this image? | cartoon | A picture of a | A picture of a | vlm_raw |
| 10 | source_pikachu | Is there a real animal in the image? | no | no | no | clip_grounding_yesno |
| 11 | source_pikachu | What color is this character? | yellow | yellow | yellow | clip_color |
| 12 | source_pikachu | 이 캐릭터의 색은 무엇인가요? | 노란색 | 노란색 | 노란색 | clip_color |