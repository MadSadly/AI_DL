# ============================================================
# app.py - LNG 탱크 용접 결함 분류 Streamlit 프로토타입 v2
# 실행: streamlit run app.py
# ============================================================

import streamlit as st
import numpy as np
import cv2
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io

st.set_page_config(page_title="LNG 탱크 용접 결함 분류기", page_icon="🔧", layout="wide")
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False


@st.cache_resource
def load_model_and_info():
    for path in ['model/best_model_final.keras', 'best_model_final.keras',
                 'model/best_model.keras', 'best_model.keras']:
        if os.path.exists(path):
            model = load_model(path)
            break
    else:
        return None, None

    for info_path in ['class_info.json', 'model/class_info.json']:
        if os.path.exists(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                return model, json.load(f)

    return model, {
        'class_indices': {'bad_weld': 0, 'blowhole': 1, 'good': 2},
        'class_kor': {'bad_weld': '용접불량', 'blowhole': '블로우홀', 'good': '양품'},
        'img_size': 224, 'best_model': 'EfficientNetV2B0', 'val_accuracy': 0.0
    }


def preprocess_image(image_pil, img_size=224):
    img_rgb = np.array(image_pil.convert('RGB'))
    img_res = cv2.resize(img_rgb, (img_size, img_size))
    return np.expand_dims(img_res.astype(np.float32) / 255.0, axis=0), img_res


def get_gradcam(model, img_array):
    last_conv = None
    for layer in reversed(model.layers[0].layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer
            break
    if last_conv is None:
        return None, None, None

    grad_model = tf.keras.Model(inputs=model.layers[0].input,
                                outputs=[last_conv.output, model.output])
    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        tape.watch(inputs)
        conv_out, preds = grad_model(inputs)
        pred_idx = tf.argmax(preds[0])
        loss = preds[:, pred_idx]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis]).numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap, int(pred_idx.numpy()), preds[0].numpy()


def gradcam_fig(img_rgb, heatmap, pred_kor, alpha=0.4):
    h, w = img_rgb.shape[:2]
    hm = cv2.resize(heatmap, (w, h))
    hm_col = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 1 - alpha, hm_col, alpha, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_rgb);      axes[0].set_title('입력 이미지', fontsize=13, fontweight='bold'); axes[0].axis('off')
    im = axes[1].imshow(hm, cmap='jet')
    axes[1].set_title('Grad-CAM 히트맵\n(빨강=주목 영역)', fontsize=11); axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    axes[2].imshow(overlay);      axes[2].set_title(f'합성 | 예측: {pred_kor}', fontsize=11); axes[2].axis('off')
    plt.tight_layout()
    return fig


def prob_fig(probs, idx_to_class, class_kor, pred_idx):
    labels = [class_kor.get(idx_to_class[i], idx_to_class[i]) for i in range(len(probs))]
    colors = ['#FF6B6B' if i == pred_idx else '#90CAF9' for i in range(len(probs))]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.barh(labels, probs * 100, color=colors, edgecolor='white')
    for bar, p in zip(bars, probs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{p*100:.1f}%', va='center', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 115); ax.set_xlabel('예측 확률 (%)'); ax.set_title('클래스별 예측 확률', fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.5); plt.tight_layout()
    return fig


# ── UI 시작 ──────────────────────────────────────────────────
st.title("🔧 LNG 탱크 용접 결함 분류 시스템")
st.markdown("**EfficientNetV2B0 전이학습 기반** | 클래스: 용접불량 / 블로우홀 / 양품")
st.divider()

tab1, tab2, tab3 = st.tabs(["🔍 결함 분류", "📊 모델 비교", "📈 학습 결과"])

# ============================================================
# TAB 1: 결함 분류
# ============================================================
with tab1:
    model, class_info = load_model_and_info()

    if model is None:
        st.error("❌ 모델 파일 없음 — Colab 실행 후 best_model_final.keras를 복사하세요.")
        st.stop()

    class_indices = class_info['class_indices']
    class_kor     = class_info['class_kor']
    IMG_SIZE      = class_info.get('img_size', 224)
    idx_to_class  = {v: k for k, v in class_indices.items()}
    val_acc_ref   = class_info.get('val_accuracy', 0.0)

    c1, c2, c3 = st.columns(3)
    c1.metric("사용 모델", class_info.get('best_model', 'EfficientNetV2B0'))
    c2.metric("검증 정확도", f"{val_acc_ref*100:.1f}%")
    c3.metric("입력 크기", f"{IMG_SIZE}×{IMG_SIZE}")
    st.divider()

    with st.sidebar:
        st.header("⚙️ 설정")
        alpha     = st.slider("Grad-CAM 투명도", 0.1, 0.9, 0.4, 0.05)
        threshold = st.slider("불량 판정 임계값 (%)", 50, 95, 70, 5)
        st.divider()
        st.subheader("📋 클래스 설명")
        st.markdown("- 🔴 **용접불량**: 용접 형상·품질 불량\n- 🟠 **블로우홀**: 기공(구멍) 결함\n- 🟢 **양품**: 정상 용접")

    st.subheader("📤 이미지 업로드")
    uploaded = st.file_uploader("용접 이미지 (JPG, PNG)", type=['jpg', 'jpeg', 'png'])

    b1, b2, b3, _ = st.columns([1, 1, 1, 3])
    example_cls = None
    if b1.button("🔴 용접불량 예시"): example_cls = 'bad_weld'
    if b2.button("🟠 블로우홀 예시"): example_cls = 'blowhole'
    if b3.button("🟢 양품 예시"):     example_cls = 'good'

    if example_cls:
        for base in ['Joint_dataset_mini/val', 'data/val']:
            p = os.path.join(base, example_cls)
            if os.path.exists(p):
                files = [f for f in os.listdir(p) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if files:
                    uploaded = open(os.path.join(p, files[0]), 'rb')
                    st.info(f"예시: {class_kor.get(example_cls)} ({files[0]})")
                    break

    if uploaded:
        st.divider()
        image_pil = Image.open(uploaded)

        col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
        with col_img2:
            st.subheader("📷 업로드된 이미지")
            st.image(image_pil, caption="입력 이미지", use_column_width=True)
        st.divider()

        with st.spinner("🔄 분석 중..."):
            img_arr, img_rgb = preprocess_image(image_pil, IMG_SIZE)
            probs     = model.predict(img_arr, verbose=0)[0]
            pred_idx  = np.argmax(probs)
            pred_cls  = idx_to_class[pred_idx]
            pred_kor  = class_kor.get(pred_cls, pred_cls)
            conf      = probs[pred_idx] * 100

            try:
                heatmap, _, _ = get_gradcam(model, img_arr)
                gradcam_ok = heatmap is not None
            except:
                gradcam_ok = False

        if pred_cls == 'good':
            st.success(f"## ✅ 판정: **{pred_kor}** (신뢰도: {conf:.1f}%)")
        elif conf >= threshold:
            st.error(f"## ❌ 판정: **{pred_kor}** (신뢰도: {conf:.1f}%)")
        else:
            st.warning(f"## ⚠️ 판정: **{pred_kor}** — 신뢰도 낮음 ({conf:.1f}%)")

        st.subheader("📊 클래스별 예측 확률")
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.table({'클래스': [class_kor.get(idx_to_class[i]) for i in range(len(probs))],
                      '확률':   [f"{p*100:.2f}%" for p in probs],
                      '판정':   ['◀ 선택됨' if i == pred_idx else '' for i in range(len(probs))]})
        with col_b:
            f = prob_fig(probs, idx_to_class, class_kor, pred_idx)
            st.pyplot(f); plt.close(f)

        if gradcam_ok:
            st.subheader("🔍 Grad-CAM XAI — 모델 판단 근거")
            st.markdown("> 빨간 영역: 분류에 가장 크게 기여한 부위")
            f = gradcam_fig(img_rgb, heatmap, pred_kor, alpha)
            st.pyplot(f)
            buf = io.BytesIO()
            f.savefig(buf, format='png', dpi=150, bbox_inches='tight'); buf.seek(0)
            st.download_button("📥 Grad-CAM 다운로드", buf, f"gradcam_{pred_cls}.png", "image/png")
            plt.close(f)

# ============================================================
# TAB 2: 모델 비교
# ============================================================
with tab2:
    st.subheader("📊 4개 모델 성능 비교")
    st.markdown("Basic CNN · VGG16 · ResNet50 · EfficientNetV2B0 — 동일 조건(10 Epoch) 비교")

    img_path = 'results/01_model_comparison.png'
    if os.path.exists(img_path):
        st.image(img_path, use_column_width=True)
        if os.path.exists('results/02_comparison_curves.png'):
            st.subheader("📈 학습 곡선 비교")
            st.image('results/02_comparison_curves.png', use_column_width=True)
    else:
        st.info("📌 Colab 실행 후 `results/` 폴더에 이미지를 복사하면 실제 결과가 표시됩니다.")
        # 샘플 차트
        fig, ax = plt.subplots(figsize=(9, 5))
        names = ['Basic_CNN', 'VGG16', 'ResNet50', 'EfficientNetV2']
        accs  = [72.5, 84.3, 88.1, 93.7]
        colors = ['#90CAF9', '#90CAF9', '#90CAF9', '#FF6B6B']
        bars = ax.bar(names, accs, color=colors, edgecolor='white', linewidth=1.2)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.5,
                    f'{acc:.1f}%', ha='center', fontweight='bold')
        ax.set_title('4개 모델 성능 비교 (예시 데이터)', fontsize=13, fontweight='bold')
        ax.set_ylabel('검증 정확도 (%)'); ax.set_ylim([0, 108])
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.subheader("📋 모델 특성 비교")
    st.table({'모델':        ['Basic CNN', 'VGG16', 'ResNet50', 'EfficientNetV2B0'],
              '파라미터':    ['~200만', '~1.38억', '~2,550만', '~2,100만'],
              '전이학습':    ['❌', '✅', '✅', '✅'],
              '소량데이터':  ['△ 불리', '△ 주의', '○ 무난', '✅ 최적'],
              '선택여부':    ['기준선', '', '', '⭐ 최종 선택']})

# ============================================================
# TAB 3: 학습 결과
# ============================================================
with tab3:
    st.subheader("📈 최종 모델 (EfficientNetV2B0) 학습 결과")
    result_map = {
        '학습 곡선':       'results/03_learning_curves.png',
        'Confusion Matrix': 'results/04_confusion_matrix.png',
        '샘플 비교':       'results/05_sample_comparison.png',
        'ROC AUC 곡선':    'results/06_roc_curve.png',
        '픽셀 밝기 분포':  'results/07_brightness_boxplot.png',
        't-SNE 시각화':    'results/08_tsne.png',
        'Grad-CAM XAI':    'results/09_gradcam.png',
    }
    any_found = False
    for title, path in result_map.items():
        if os.path.exists(path):
            st.subheader(title); st.image(path, use_column_width=True); any_found = True

    if not any_found:
        st.info("📌 Colab 실행 완료 후 시각화 이미지를 `results/` 폴더에 복사하면 여기서 확인할 수 있습니다.")

st.divider()
st.markdown("<div style='text-align:center; color:gray; font-size:12px;'>"
            "LNG 탱크 용접 결함 분류 | EfficientNetV2B0 Transfer Learning | "
            "데이터: 미래아이티컨소시엄 (CC BY-NC)</div>", unsafe_allow_html=True)