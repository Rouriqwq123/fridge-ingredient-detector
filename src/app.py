"""
Day 10: Gradio Web 应用
上传冰箱照片 → YOLOv8 检测食材 → 推荐菜谱

Usage: python src/app.py
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

import gradio as gr

# 确保 src 目录在 Python 路径里
sys.path.insert(0, str(Path(__file__).parent))
from recommend import RecipeRecommender

# ============================================================
# 配置
# ============================================================
MODEL_PATH  = Path("runs/detect/runs/train/fridge_v12/weights/best.pt")
CONF_THRESH = 0.25   # 置信度阈值
MAX_MISSING = 3      # 最多缺几种食材
MAX_MINUTES = 60     # 最长烹饪时间
TOP_K       = 5      # 推荐几道菜

# ============================================================
# 全局加载模型（只加载一次）
# ============================================================
print("🔧 加载模型和菜谱数据...")
model       = YOLO(str(MODEL_PATH))
recommender = RecipeRecommender()
print("✅ 加载完成！")

# ============================================================
# 核心推理函数
# ============================================================
def detect_and_recommend(image, conf_thresh, max_missing, max_minutes):
    """
    输入：PIL Image
    输出：标注图片, 食材列表文本, 菜谱推荐文本
    """
    if image is None:
        return None, "请上传图片", "请先上传冰箱照片"

    # PIL → numpy (BGR for OpenCV)
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # YOLOv8 推理
    results = model(img_bgr, conf=conf_thresh, verbose=False)
    result  = results[0]

    # 解析检测结果
    detected = {}  # {class_name: max_confidence}
    for box in result.boxes:
        cls_id  = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf    = float(box.conf[0])
        if cls_name not in detected or conf > detected[cls_name]:
            detected[cls_name] = conf

    # 生成标注图片
    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # 食材列表文本
    if not detected:
        ingredient_text = "❌ 未检测到任何食材\n\n请尝试：\n• 拍摄更清晰的照片\n• 降低置信度阈值"
        recipe_text = "请先成功检测到食材"
        return annotated_rgb, ingredient_text, recipe_text

    # 按置信度排序
    sorted_items = sorted(detected.items(), key=lambda x: x[1], reverse=True)
    ingredient_lines = [f"✅ {name:<20} {conf:.1%}" for name, conf in sorted_items]
    ingredient_text  = f"🔍 检测到 {len(detected)} 种食材：\n\n" + "\n".join(ingredient_lines)

    # 菜谱推荐
    ingredient_list = list(detected.keys())
    recipes = recommender.recommend(
        ingredient_list,
        top_k=TOP_K,
        max_minutes=max_minutes,
        max_missing=max_missing,
    )

    if not recipes:
        recipe_text = "❌ 没有找到合适的菜谱\n\n建议：\n• 增加 '最多缺少食材数'\n• 延长 '最长烹饪时间'"
    else:
        lines = [f"🍽️ 根据你冰箱里的食材，推荐以下 {len(recipes)} 道菜：\n"]
        for i, r in enumerate(recipes, 1):
            lines.append(f"{'─'*45}")
            lines.append(f"🍳 #{i}  {r['name']}")
            lines.append(
                f"   ✅ 已有 {r['matched']} 种  |  "
                f"还缺 {r['missing']} 种  |  "
                f"⏱️ {r['minutes']} 分钟"
            )
            if r["missing_ings"]:
                lines.append(f"   🛒 还需购买: {', '.join(r['missing_ings'][:3])}")
            lines.append(f"   📝 食材: {', '.join(r['ingredients'][:6])}" +
                         ("..." if len(r["ingredients"]) > 6 else ""))
            lines.append(f"\n   做法：")
            for j, step in enumerate(r["steps"][:3], 1):
                step_short = step[:100] + "..." if len(step) > 100 else step
                lines.append(f"   {j}. {step_short}")
            lines.append("")
        recipe_text = "\n".join(lines)

    return annotated_rgb, ingredient_text, recipe_text


# ============================================================
# Gradio 界面
# ============================================================
def build_ui():
    with gr.Blocks(
        title="🧊 冰箱食材识别 → 菜谱推荐",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown("""
        # 🧊 冰箱食材识别 → 菜谱推荐
        上传一张冰箱照片，AI 自动识别食材并推荐你能做的菜！
        """)

        with gr.Row():
            # 左栏：输入
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="📸 上传冰箱照片",
                    height=400,
                )
                with gr.Accordion("⚙️ 高级设置", open=False):
                    conf_slider = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                        label="置信度阈值（越低检测越多，但可能有误检）"
                    )
                    missing_slider = gr.Slider(
                        minimum=0, maximum=5, value=3, step=1,
                        label="最多缺少食材数（越大推荐越多）"
                    )
                    minutes_slider = gr.Slider(
                        minimum=10, maximum=120, value=60, step=10,
                        label="最长烹饪时间（分钟）"
                    )
                detect_btn = gr.Button("🔍 识别食材 + 推荐菜谱", variant="primary", size="lg")

            # 右栏：输出
            with gr.Column(scale=1):
                image_output = gr.Image(
                    label="🎯 检测结果",
                    height=400,
                )

        with gr.Row():
            ingredient_output = gr.Textbox(
                label="🥬 识别到的食材",
                lines=10,
                max_lines=20,
            )
            recipe_output = gr.Textbox(
                label="🍳 推荐菜谱",
                lines=20,
                max_lines=40,
            )

        # 示例图片
        gr.Markdown("### 💡 示例")
        gr.Examples(
            examples=[],  # 可以在这里加示例图片路径
            inputs=image_input,
        )

        # 绑定按钮
        detect_btn.click(
            fn=detect_and_recommend,
            inputs=[image_input, conf_slider, missing_slider, minutes_slider],
            outputs=[image_output, ingredient_output, recipe_output],
        )

        # 上传图片自动触发
        image_input.change(
            fn=detect_and_recommend,
            inputs=[image_input, conf_slider, missing_slider, minutes_slider],
            outputs=[image_output, ingredient_output, recipe_output],
        )

        gr.Markdown("""
        ---
        **技术栈**: YOLOv8 (Ultralytics) + Food.com 菜谱数据集 + Gradio
        """)

    return demo


# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,        # 改成 True 可以生成公网链接
        inbrowser=True,     # 自动打开浏览器
    )
