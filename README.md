# e4s_faceswap
e4s faceswap (swap_source_image_to_target_video)

### extract_style_vector.py -- terminal
'''
python extract_style_vector.py --source /path/ --style_vector_path /path/ --faceParser_ckpts /path/ --config_path""
'''

1) 이미지 로드 및 전처리:
소스 이미지를 1024x1024 크기로 리사이즈합니다.
이미지를 256x256 크기로 다시 리사이즈하여 모델 입력으로 사용합니다.

2) 마스크 생성:
3) Face Parsing 모델을 사용하여 이미지의 얼굴 마스크를 생성합니다.

4) 텐서 변환:
이미지를 텐서로 변환하고 정규화합니다.
마스크를 텐서로 변환하고 one-hot 인코딩합니다.

5) 스타일 벡터 추출:
E4S 모델을 사용하여 이미지의 스타일 벡터를 추출합니다.

6) 스타일 벡터 저장:
추출한 스타일 벡터를 지정된 경로에 저장합니다.
