"""
Day 8: 菜谱推荐模块 v2
优先推荐缺少食材最少的菜谱

Usage:
  python src/recommend.py
  from src.recommend import RecipeRecommender
"""

import ast
import json
from pathlib import Path

import pandas as pd


# ============================================================
# 配置
# ============================================================
RECIPES_CSV  = Path("data/raw/recipes/RAW_recipes.csv")
CACHE_PATH   = Path("data/processed/recipe_index.json")
MAX_RECIPES  = 50000


# ============================================================
# 主类
# ============================================================
class RecipeRecommender:

    def __init__(self):
        self.recipes = []
        self.index   = {}
        self._load()

    # ----------------------------------------------------------
    # 加载数据
    # ----------------------------------------------------------
    def _load(self):
        if CACHE_PATH.exists():
            print("  📦 加载菜谱缓存...")
            with open(CACHE_PATH, encoding="utf-8") as f:
                data = json.load(f)
            self.recipes = data["recipes"]
            self.index   = data["index"]
            print(f"  ✅ 已加载 {len(self.recipes):,} 条菜谱")
        else:
            print("  📖 首次加载，建立索引（约需30秒）...")
            self._build_index()

    def _build_index(self):
        if not RECIPES_CSV.exists():
            raise FileNotFoundError(f"找不到菜谱数据: {RECIPES_CSV}")

        df = pd.read_csv(RECIPES_CSV, nrows=MAX_RECIPES)
        df = df.dropna(subset=["name", "ingredients", "steps"])

        self.recipes = []
        self.index   = {}

        for _, row in df.iterrows():
            try:
                ingredients = ast.literal_eval(row["ingredients"])
                steps       = ast.literal_eval(row["steps"])
            except Exception:
                continue

            recipe = {
                "name":        row["name"].strip().title(),
                "ingredients": ingredients,
                "steps":       steps[:5],
                "minutes":     int(row["minutes"]) if pd.notna(row["minutes"]) else 30,
                "n_steps":     int(row["n_steps"])  if pd.notna(row["n_steps"])  else len(steps),
            }

            idx = len(self.recipes)
            self.recipes.append(recipe)

            for ing in ingredients:
                key = ing.lower().strip()
                if key not in self.index:
                    self.index[key] = []
                self.index[key].append(idx)

        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump({"recipes": self.recipes, "index": self.index},
                      f, ensure_ascii=False)

        print(f"  ✅ 索引建立完成: {len(self.recipes):,} 条菜谱，"
              f"{len(self.index):,} 种食材")

    # ----------------------------------------------------------
    # 推荐核心逻辑
    # ----------------------------------------------------------
    def recommend(self,
                  detected_ingredients: list,
                  top_k: int = 5,
                  max_minutes: int = 60,
                  max_missing: int = 3) -> list:
        """
        根据检测到的食材推荐菜谱

        Args:
            detected_ingredients: YOLO 检测到的食材列表
            top_k:               返回前 k 个推荐
            max_minutes:         最长烹饪时间（分钟）
            max_missing:         最多允许缺少几种食材（核心改进）

        Returns:
            list of dict
        """
        if not detected_ingredients:
            return []

        query = {ing.lower().strip() for ing in detected_ingredients}

        # 统计每个菜谱匹配了多少食材
        match_count = {}
        for ing in query:
            # 精确匹配
            if ing in self.index:
                for idx in self.index[ing]:
                    match_count[idx] = match_count.get(idx, 0) + 1
            # 模糊匹配
            for key in self.index:
                if (ing in key or key in ing) and ing != key:
                    for idx in self.index[key]:
                        if idx not in match_count:
                            match_count[idx] = 0
                        match_count[idx] += 0.5

        if not match_count:
            return []

        candidates = []
        for idx, matched in match_count.items():
            recipe  = self.recipes[idx]
            total   = len(recipe["ingredients"])
            missing = total - matched

            # 过滤：烹饪时间太长
            if recipe["minutes"] > max_minutes:
                continue
            # 过滤：缺少食材太多（核心改进）
            if missing > max_missing:
                continue
            # 过滤：匹配数太少（至少匹配1种）
            if matched < 1:
                continue

            # Jaccard 相似度
            jaccard = matched / (len(query) + total - matched)

            # 找出缺少的食材
            query_list   = [ing.lower().strip() for ing in detected_ingredients]
            missing_ings = [
                ing for ing in recipe["ingredients"]
                if not any(q in ing.lower() or ing.lower() in q
                           for q in query_list)
            ]

            candidates.append({
                "name":         recipe["name"],
                "matched":      int(matched),
                "total":        total,
                "missing":      int(missing),
                "missing_ings": missing_ings[:5],  # 最多显示5个缺少的食材
                "score":        round(jaccard, 3),
                "minutes":      recipe["minutes"],
                "steps":        recipe["steps"],
                "ingredients":  recipe["ingredients"],
            })

        if not candidates:
            # 放宽条件重试
            return self.recommend(
                detected_ingredients,
                top_k=top_k,
                max_minutes=max_minutes * 2,
                max_missing=max_missing + 2
            )

        # 排序：缺少食材少 → 匹配度高 → 烹饪时间短
        candidates.sort(
            key=lambda x: (x["missing"], -x["score"], x["minutes"])
        )
        return candidates[:top_k]

    # ----------------------------------------------------------
    # 格式化输出
    # ----------------------------------------------------------
    def format_result(self, recommendations: list) -> str:
        if not recommendations:
            return "❌ 没有找到匹配的菜谱，请尝试更多食材"

        lines = []
        for i, r in enumerate(recommendations, 1):
            lines.append(f"\n{'='*50}")
            lines.append(f"🍳 #{i}  {r['name']}")
            lines.append(
                f"   ✅ 已有: {r['matched']} 种  "
                f"| 还缺: {r['missing']} 种  "
                f"| 匹配度: {r['score']:.1%}  "
                f"| 烹饪时间: {r['minutes']} 分钟"
            )

            if r["missing_ings"]:
                lines.append(f"\n   ⚠️  还需要购买:")
                for ing in r["missing_ings"]:
                    lines.append(f"     • {ing}")

            lines.append(f"\n   所需全部食材:")
            for ing in r["ingredients"]:
                lines.append(f"     • {ing}")

            lines.append(f"\n   做法 (前{len(r['steps'])}步):")
            for j, step in enumerate(r["steps"], 1):
                step_text = step[:150] + "..." if len(step) > 150 else step
                lines.append(f"     {j}. {step_text}")

        return "\n".join(lines)


# ============================================================
# 测试
# ============================================================
def main():
    print()
    print("🧊 Fridge AI - 菜谱推荐测试 v2")
    print("   改进：优先推荐缺少食材最少的菜谱")
    print()

    # 删除旧缓存，重新建立
    if CACHE_PATH.exists():
        CACHE_PATH.unlink()
        print("  🗑️  已删除旧缓存，重新建立索引...")

    recommender = RecipeRecommender()

    test_cases = [
        (["tomato", "egg", "onion", "garlic"],          "最多缺3种"),
        (["chicken", "carrot", "potato", "onion"],      "最多缺3种"),
        (["apple", "banana", "strawberry"],              "最多缺3种"),
        (["rice", "pork", "garlic", "ginger", "onion"], "最多缺2种"),
    ]

    for ingredients, desc in test_cases:
        print(f"\n🥬 检测到的食材 ({desc}): {ingredients}")
        results = recommender.recommend(
            ingredients,
            top_k=3,
            max_minutes=60,
            max_missing=3
        )
        print(recommender.format_result(results))
        print()

    print("=" * 50)
    print("✅ 推荐模块测试完成！")
    print()
    print("下一步 (Day 10): 搭建 Gradio 界面")
    print("  python src/app.py")


if __name__ == "__main__":
    main()
