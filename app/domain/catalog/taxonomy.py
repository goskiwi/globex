"""目录的结构化枚举，不根据查询文字推断分类或材质。"""
from typing import Literal, get_args

Category = Literal["旅行装备", "户外运动", "数码配件", "家居生活", "美妆个护", "厨房餐饮", "办公学习", "母婴宠物"]
MaterialTag = Literal["合成聚合物", "天然材料", "天然纤维", "玻璃", "纸", "金属", "陶瓷"]
CATEGORIES = get_args(Category)
MATERIAL_TAGS = get_args(MaterialTag)
