# -*- coding: utf-8 -*-
"""生成可复现的演示商品集（1000 SPU / 1705 SKU）。

这是一次性数据构建工具，运行后得到应提交到仓库的 ``data/catalog-v2.jsonl``；
运行时只读取 JSONL，业务代码不再依赖硬编码商品表。
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.catalog_seed_data import build_demo_seed_products


_OUT = Path(__file__).resolve().parents[1] / "data" / "catalog-v2.jsonl"
_PLATFORMS = ("amazon", "ebay", "etsy", "walmart")
_CURRENCIES = ("CNY", "USD", "EUR", "JPY", "SGD")
_CATEGORIES = (
    ("旅行装备", "旅行收纳 轻便 出行", "旅行收纳包", 0.48, (32, 20, 42)),
    ("数码配件", "数码 快充 便携", "数码扩展坞", 0.12, (12, 6, 3)),
    ("家居生活", "家居 天然材质 轻器物", "家居收纳盒", 0.35, (18, 12, 10)),
    ("户外运动", "户外 防水 耐用 徒步", "户外保温杯", 0.42, (9, 9, 25)),
    ("美妆个护", "个护 低敏 旅行装", "旅行洗护套装", 0.08, (6, 6, 15)),
    ("厨房餐饮", "厨房 食品接触 便携", "便携餐盒", 0.5, (18, 12, 7)),
    ("办公学习", "办公 护眼 轻薄", "桌面收纳架", 0.7, (28, 24, 18)),
    ("母婴宠物", "母婴 宠物 安全 易清洁", "宠物出行包", 0.9, (45, 28, 30)),
)
_MATERIALS = (
    ("天然纤维", "帆布棉麻"),
    ("合成聚合物", "再生尼龙"),
    ("金属", "铝合金"),
    ("陶瓷", "高温陶瓷"),
    ("玻璃", "耐热玻璃"),
)
_ORIGINS = ("CN", "US", "DE", "JP", "KR", "VN", "PT", "SG")
_DESTINATIONS = (
    ["CN"], ["US"], ["EU"], ["JP"], ["SG"],
    ["CN", "US"], ["CN", "EU"], ["US", "EU", "JP"], ["CN", "US", "EU", "JP", "SG"],
)


def _legacy_record(product, index: int) -> dict:
    # “再生”不改变材料属性；记忆棉、超纤、PVC/PU 等同样不能标为天然材质。
    synthetic_markers = ("尼龙", "记忆棉", "超纤", "聚酯", "涤纶", "PVC", "PU", "塑料")
    material = "合成聚合物" if any(marker in product.description for marker in synthetic_markers) else "天然材料"
    description = product.description.replace("无塑料感", "含再生尼龙")
    highlights = [{"label": h.label, "detail": h.detail} for h in product.highlights]
    if product.product_id == "P1001":
        highlights = [
            {"label": "材质", "detail": "帆布+再生尼龙（含合成聚合物）"},
            *highlights[1:],
        ]
    return {
        "product_id": product.product_id,
        "title": product.title,
        "brand": product.brand,
        "category": product.category,
        "origin_country": product.origin_country,
        "description": description,
        "highlights": highlights,
        "ships_to": product.ships_to,
        "skus": [
            {"sku_id": sku.sku_id, "spec": sku.spec, "price_major": sku.price.to_major_units(), "currency": sku.price.currency, "stock": sku.stock}
            for sku in product.skus
        ],
        "source_platform": _PLATFORMS[index % len(_PLATFORMS)],
        "external_product_id": f"{_PLATFORMS[index % len(_PLATFORMS)]}-{product.product_id}",
        "canonical_product_id": f"CAN-LEGACY-{index:03d}",
        "material_tags": [material],
        "weight_kg": product.weight_kg,
        "dimensions_cm": product.dimensions_cm,
        "package_dimensions_cm": product.package_dimensions_cm,
        "tax_category": product.category,
        "rating_summary": {"average": round(4.0 + (index % 9) / 10, 1), "review_count": 30 + index * 11},
        "updated_at": "2026-08-01",
    }


def _price(index: int, currency: str) -> float:
    base = (39, 89, 169, 329, 699, 1299)[index % 6]
    if currency == "CNY":
        return float(base)
    if currency == "USD":
        return round(base / 7.1, 2)
    if currency == "EUR":
        return round(base / 7.8, 2)
    if currency == "JPY":
        return round(base / 0.048)
    return round(base / 5.3, 2)


def _generated_record(index: int) -> dict:
    category, keywords, noun, weight_kg, dimensions = _CATEGORIES[index % len(_CATEGORIES)]
    material_tag, material_text = _MATERIALS[index % len(_MATERIALS)]
    platform = _PLATFORMS[index % len(_PLATFORMS)]
    currency = _CURRENCIES[index % len(_CURRENCIES)]
    product_id = f"P{2000 + index:04d}"
    multi_sku = index < 200
    fully_out = index < 50
    partially_out = 50 <= index < 100
    skus = []
    for variant in range(2 if multi_sku else 1):
        stock = 0 if fully_out or (partially_out and variant == 0) else 8 + (index * 7 + variant * 11) % 180
        skus.append(
            {
                "sku_id": f"{product_id}-S{variant + 1}",
                "spec": ("标准版" if variant == 0 else "升级版"),
                "price_major": round(_price(index + variant, currency) * (1 if variant == 0 else 1.12), 2),
                "currency": currency,
                "stock": stock,
            },
        )
    title = f"Atlas {noun} {index:03d}"
    brand = f"Atlas-{index % 24:02d}"
    description = f"{keywords} {material_text} 多场景使用 轻量 耐用 评测候选 {index:03d}"
    # 为“无合成聚合物旅行三件套”保留一个真实可检索的天然材质正例，
    # 避免把含再生尼龙的 P1001 错标为“无塑料”。
    if index == 120:
        title = "PureCanvas 纯棉旅行三件套（收纳袋+颈枕+眼罩）"
        brand = "PureCanvas"
        description = "旅行三件套 纯棉 帆布 天然材质 不含合成聚合物 长途飞行 评测候选"

    evaluation_tags = ["hard_negative"] if index < 75 else []
    if evaluation_tags:
        description += " 标题近似款：关键属性故意不匹配，用于检验属性过滤而非标题碰撞。"

    return {
        "product_id": product_id,
        "title": title,
        "brand": brand,
        "category": category,
        "origin_country": _ORIGINS[index % len(_ORIGINS)],
        "description": description,
        "highlights": [
            {"label": "材质", "detail": material_text},
            {"label": "测试属性", "detail": f"候选分组 {index // 5}"},
        ],
        "ships_to": _DESTINATIONS[index % len(_DESTINATIONS)],
        "skus": skus,
        "source_platform": platform,
        "external_product_id": f"{platform}-{product_id}",
        "canonical_product_id": f"CAN-{index // 5:03d}",
        "material_tags": [material_tag],
        "weight_kg": weight_kg,
        "dimensions_cm": dict(zip(("length","width","height"), dimensions)),
        "package_dimensions_cm": {},
        "tax_category": category,
        "rating_summary": {"average": round(3.8 + (index % 12) / 10, 1), "review_count": 20 + index * 13},
        "updated_at": f"2026-08-{1 + index % 28:02d}",
        "evaluation_tags": evaluation_tags,
    }


@dataclass(frozen=True)
class DemoSpec:
    """一个型号的单一设定；重量文案和结构化值都从这里生成。"""
    text: str
    weight_kg: float | None = None
    dimensions_cm: tuple[int, int, int] | None = None

    def label(self) -> str:
        values = {"weight_g": f"{self.weight_kg * 1000:g}"} if self.weight_kg is not None else {}
        return self.text.format(**values)


# 每行是一种商品，不同档位有真实的功能差异；四个平台卖同一型号时共享 canonical ID。
# 全部为合成演示数据，品牌、规格、价格均不代表平台真实在售商品。
# category, noun, material, purpose, four specification tiers
_EXPANDED_FAMILIES = (
    ("旅行装备", "登山背包", "合成聚合物", "山路步行时背负装备，有胸带和腰带分担肩部重量", (DemoSpec("15L 无防雨罩", 0.52, (28, 16, 40)), DemoSpec("25L 无防雨罩", 0.68, (30, 20, 47)), DemoSpec("35L 配防雨罩", 0.82, (32, 23, 53)), DemoSpec("45L 配防雨罩及独立水袋仓", 1.1, (34, 26, 57)))),
    ("旅行装备", "航空颈枕", "天然纤维", "坐着休息时稳定头部，外套为可拆洗纯棉", (DemoSpec("无侧向支撑", 0.12, (28, 25, 10)), DemoSpec("单侧支撑", 0.15, (30, 26, 11)), DemoSpec("双侧支撑", 0.18, (31, 28, 12)), DemoSpec("双侧支撑且前部有托下巴结构", 0.22, (32, 29, 13)))),
    ("旅行装备", "行李箱", "金属", "机场转机拖行，铝合金箱体配万向轮", (DemoSpec("28寸 托运尺寸", 5.2, (50, 32, 76)), DemoSpec("24寸 托运尺寸", 4.3, (44, 28, 65)), DemoSpec("20寸 仅密码锁", 2.8, (36, 23, 55)), DemoSpec("20寸 TSA锁静音轮", 3.1, (36, 23, 55)))),
    ("旅行装备", "压缩收纳袋", "合成聚合物", "行李内分开衣物，拉链可重复开合", (DemoSpec("2件 无压缩层", 0.15, (32, 24, 4)), DemoSpec("3件 无压缩层", 0.22, (34, 26, 5)), DemoSpec("4件 双层压缩", 0.3, (36, 28, 6)), DemoSpec("6件 双层压缩干湿分离", 0.42, (40, 30, 7)))),
    ("数码配件", "氮化镓充电器", "合成聚合物", "出差给便携电脑和手机补电，支持USB-C PD", (DemoSpec("30W 单口", 0.065, (4, 3, 3)), DemoSpec("45W 双口", 0.085, (5, 4, 3)), DemoSpec("65W 双口", 0.12, (6, 4, 4)), DemoSpec("100W 三口全球插脚", 0.19, (7, 5, 4)))),
    ("数码配件", "蓝牙耳机", "合成聚合物", "乘地铁听播客，麦克风支持语音通话", (DemoSpec("仅通话降噪 20小时", 0.045, (6, 5, 3)), DemoSpec("仅通话降噪 30小时", 0.05, (6, 5, 3)), DemoSpec("主动降噪35dB 40小时", 0.055, (7, 5, 3)), DemoSpec("主动降噪45dB 50小时可折叠", 0.22, (20, 18, 8)))),
    ("数码配件", "USB-C扩展坞", "金属", "笔记本连接外接显示器和有线网络，铝合金外壳", (DemoSpec("HDMI1080P 无网口", 0.055, (10, 3, 1)), DemoSpec("HDMI4K30Hz 无网口", 0.065, (11, 4, 1)), DemoSpec("HDMI4K60Hz 千兆网口", 0.085, (12, 4, 2)), DemoSpec("双HDMI4K60Hz 千兆网口100W回充", 0.12, (14, 5, 2)))),
    ("数码配件", "移动电源", "合成聚合物", "离开插座后为手机供电，内置锂电池", (DemoSpec("5000mAh 10W", 0.11, (10, 6, 1)), DemoSpec("10000mAh 18W", 0.19, (13, 7, 2)), DemoSpec("20000mAh 30W", 0.3, (15, 8, 3)), DemoSpec("20000mAh 65W自带USB-C线", 0.35, (16, 8, 3)))),
    ("家居生活", "遮光窗帘", "天然纤维", "日间睡眠减少窗外光线，纯棉面料可机洗", (DemoSpec("遮光率50%", 0.6, (140, 200, 1)), DemoSpec("遮光率70%", 0.8, (140, 200, 1)), DemoSpec("遮光率90%", 1, (140, 200, 1)), DemoSpec("遮光率99%双层隔热", 1.4, (140, 200, 2)))),
    ("家居生活", "粗陶马克杯", "陶瓷", "书桌喝咖啡，粗陶手工釉面中性色", (DemoSpec("200ml 无把手", 0.2, (8, 8, 8)), DemoSpec("280ml 无把手", 0.25, (9, 9, 9)), DemoSpec("350ml 带把手", 0.3, (12, 9, 10)), DemoSpec("450ml 带把手可进洗碗机", 0.38, (13, 10, 12)))),
    ("家居生活", "衣物收纳箱", "天然纤维", "换季整理衣柜，棉麻表层收纳衣物", (DemoSpec("10L 不可叠放", 0.3, (30, 25, 16)), DemoSpec("20L 不可叠放", 0.5, (40, 30, 20)), DemoSpec("40L 可叠放", 0.85, (50, 35, 28)), DemoSpec("60L 可叠放防尘带透明窗口", 1.1, (60, 40, 32)))),
    ("家居生活", "折叠晾衣架", "金属", "阳台晒衣物，铝合金支撑架可折叠", (DemoSpec("承重5kg", 1.1, (60, 45, 80)), DemoSpec("承重10kg", 1.5, (80, 50, 90)), DemoSpec("承重20kg", 2, (100, 60, 100)), DemoSpec("承重30kg可伸缩带轮", 2.8, (120, 65, 110)))),
    ("户外运动", "露营灯", "合成聚合物", "帐篷和营地照明，Type-C充电", (DemoSpec("100流明 不防水", 0.1, (6, 6, 10)), DemoSpec("200流明 IPX4", 0.15, (8, 8, 12)), DemoSpec("400流明 IPX5", 0.22, (9, 9, 15)), DemoSpec("800流明 IPX6磁吸挂钩", 0.3, (11, 11, 18)))),
    ("户外运动", "登山杖", "金属", "上下坡借力减轻膝盖负担，铝合金杖身", (DemoSpec("{weight_g}g 不可折叠", 0.45, (125, 4, 4)), DemoSpec("{weight_g}g 三节伸缩", 0.35, (125, 4, 4)), DemoSpec("{weight_g}g 三节折叠", 0.28, (125, 4, 4)), DemoSpec("{weight_g}g 五节折叠快锁", 0.22, (125, 4, 4)))),
    ("户外运动", "羽绒睡袋", "合成聚合物", "户外过夜保暖，填充鸭绒外层尼龙", (DemoSpec("舒适温15℃", 0.65, (210, 75, 4)), DemoSpec("舒适温5℃", 0.9, (210, 80, 6)), DemoSpec("舒适温0℃", 1.2, (215, 80, 8)), DemoSpec("舒适温-10℃防风帽", 1.5, (220, 85, 10)))),
    ("户外运动", "真空保温壶", "金属", "徒步带热水，316不锈钢内胆", (DemoSpec("350ml 保温4小时", 0.21, (7, 7, 19)), DemoSpec("500ml 保温6小时", 0.28, (8, 8, 23)), DemoSpec("750ml 保温12小时", 0.36, (9, 9, 27)), DemoSpec("1000ml 保温24小时带提手", 0.44, (10, 10, 29)))),
    ("美妆个护", "电动剃须刀", "金属", "出差整理胡须，可USB充电", (DemoSpec("单刀头不可水洗", 0.09, (13, 4, 3)), DemoSpec("双刀头刀头可水洗", 0.12, (15, 5, 4)), DemoSpec("三刀头全身水洗", 0.16, (16, 6, 5)), DemoSpec("三刀头全身水洗带鬓角修剪器", 0.19, (17, 6, 5)))),
    ("美妆个护", "洗漱分装瓶", "玻璃", "洗护液分装携带，玻璃瓶体避免吸附气味", (DemoSpec("150ml 普通盖", 0.12, (5, 5, 14)), DemoSpec("100ml 普通盖", 0.09, (4, 4, 12)), DemoSpec("80ml 防漏旋盖", 0.08, (4, 4, 10)), DemoSpec("60ml 防漏旋盖带标签四只装", 0.22, (12, 8, 10)))),
    ("美妆个护", "旅行吹风机", "合成聚合物", "酒店洗头后吹干头发，可折叠手柄", (DemoSpec("600W 单电压", 0.22, (16, 6, 16)), DemoSpec("1000W 单电压", 0.28, (18, 7, 18)), DemoSpec("1200W 双电压", 0.35, (20, 8, 19)), DemoSpec("1600W 双电压负离子", 0.42, (22, 8, 20)))),
    ("美妆个护", "洁面巾", "天然纤维", "洗脸后擦拭水分，纯棉无香料", (DemoSpec("30抽薄款", 0.08, (15, 10, 4)), DemoSpec("50抽薄款", 0.12, (16, 11, 6)), DemoSpec("80抽加厚", 0.2, (18, 12, 8)), DemoSpec("100抽加厚独立包装", 0.3, (20, 12, 10)))),
    ("厨房餐饮", "密封餐盒", "玻璃", "上班带饭，硼硅玻璃盒体", (DemoSpec("400ml 不可微波", 0.25, (15, 10, 5)), DemoSpec("600ml 可微波无分隔", 0.4, (17, 12, 6)), DemoSpec("900ml 可微波双分隔", 0.55, (20, 14, 7)), DemoSpec("1200ml 可微波三分隔防漏", 0.7, (23, 16, 8)))),
    ("厨房餐饮", "便携餐具", "金属", "露营或午餐使用，可反复清洗", (DemoSpec("不锈钢勺单件", 0.04, (18, 4, 2)), DemoSpec("不锈钢叉勺两件", 0.075, (20, 5, 2)), DemoSpec("钛合金叉勺两件", 0.045, (20, 5, 2)), DemoSpec("钛合金筷叉勺三件带收纳盒", 0.08, (22, 6, 3)))),
    ("厨房餐饮", "手冲咖啡壶", "金属", "控制细水流冲泡咖啡，不锈钢细嘴", (DemoSpec("300ml 无温度显示", 0.25, (17, 10, 13)), DemoSpec("500ml 无温度显示", 0.35, (20, 12, 16)), DemoSpec("700ml 温度显示", 0.45, (23, 14, 18)), DemoSpec("900ml 温度显示电加热控温", 0.65, (25, 16, 20)))),
    ("厨房餐饮", "食品保鲜袋", "合成聚合物", "冰箱分装食材，食品级硅胶可重复使用", (DemoSpec("300ml 不可冷冻", 0.06, (15, 12, 3)), DemoSpec("500ml 可冷冻", 0.08, (18, 15, 4)), DemoSpec("1000ml 可冷冻密封", 0.12, (22, 18, 5)), DemoSpec("1500ml 可冷冻密封可洗碗机", 0.18, (25, 20, 6)))),
    ("办公学习", "阅读台灯", "金属", "书桌阅读照明，铝合金灯臂可调角度", (DemoSpec("单色温 300流明", 0.5, (25, 15, 35)), DemoSpec("双色温 400流明", 0.6, (28, 16, 38)), DemoSpec("三色温 600流明", 0.75, (30, 18, 42)), DemoSpec("无级色温 800流明显色指数95", 0.9, (32, 20, 45)))),
    ("办公学习", "笔记本支架", "金属", "抬高电脑屏幕改善桌面视线，铝合金底座", (DemoSpec("固定高度5cm", 0.45, (25, 22, 5)), DemoSpec("两档高度10cm", 0.55, (26, 23, 10)), DemoSpec("六档高度18cm", 0.65, (28, 24, 18)), DemoSpec("无级升降25cm旋转底座", 0.8, (30, 25, 25)))),
    ("办公学习", "无线鼠标", "合成聚合物", "办公控制光标，右手握持", (DemoSpec("仅2.4G 有声按键", 0.08, (11, 6, 4)), DemoSpec("仅蓝牙 静音按键", 0.085, (11, 6, 4)), DemoSpec("蓝牙2.4G双模 静音", 0.09, (12, 6, 4)), DemoSpec("蓝牙2.4G双模 静音多设备切换", 0.095, (12, 7, 4)))),
    ("办公学习", "文件收纳夹", "天然纤维", "分类整理纸张，牛皮纸内页和棉布封面", (DemoSpec("A5 4格", 0.12, (22, 16, 2)), DemoSpec("A4 6格", 0.18, (32, 24, 3)), DemoSpec("A4 12格", 0.28, (33, 25, 4)), DemoSpec("A4 24格防尘拉链", 0.4, (34, 26, 6)))),
    ("母婴宠物", "宠物航空包", "合成聚合物", "带猫出行，网面透气便于观察", (DemoSpec("承重3kg 单侧开门", 0.65, (38, 24, 26)), DemoSpec("承重5kg 双侧开门", 0.8, (42, 26, 28)), DemoSpec("承重7kg 可拆洗垫", 1, (46, 28, 30)), DemoSpec("承重9kg 可拆洗垫四面透气", 1.2, (50, 30, 32)))),
    ("母婴宠物", "宠物饮水器", "金属", "猫咪日常喝水，304不锈钢饮水盘", (DemoSpec("1L 无过滤", 0.6, (16, 16, 10)), DemoSpec("1.5L 单层过滤", 0.8, (18, 18, 12)), DemoSpec("2L 双层过滤", 1, (20, 20, 14)), DemoSpec("3L 三层过滤缺水断电", 1.2, (22, 22, 16)))),
    ("母婴宠物", "婴儿纱布浴巾", "天然纤维", "宝宝洗澡后包裹吸水，纯棉纱布", (DemoSpec("2层60cm", 0.15, (60, 60, 1)), DemoSpec("4层80cm", 0.22, (80, 80, 1)), DemoSpec("6层100cm", 0.3, (100, 100, 1)), DemoSpec("8层120cm无荧光剂", 0.4, (120, 120, 1)))),
    ("母婴宠物", "宠物梳毛刷", "金属", "梳理猫犬浮毛，不锈钢圆头梳齿", (DemoSpec("固定短齿", 0.07, (14, 7, 3)), DemoSpec("固定长齿", 0.08, (15, 8, 3)), DemoSpec("可调齿距", 0.09, (16, 9, 4)), DemoSpec("可调齿距一键退毛", 0.11, (17, 9, 4)))),
)


def _expanded_records():
    records = []
    for model_index in range(125):
        family, tier = divmod(model_index, 4)
        category, noun, material, purpose, specs = _EXPANDED_FAMILIES[family]
        specification = specs[tier]
        detail = specification.label()
        model = f"GX-{family+1:02d}-{tier+1}"
        for platform_index, platform in enumerate(_PLATFORMS):
            offset = model_index*4+platform_index
            pid = f"P{3000+offset}"
            currency = _CURRENCIES[(model_index+platform_index)%5]
            base_cny = (59,99,189,329)[tier] * (1 + family%3) + platform_index*7
            factor = {"CNY":1,"USD":7.1,"EUR":7.8,"JPY":.048,"SGD":5.3}[currency]
            fully_out = offset%10 == 0
            partially_out = offset%10 == 1
            records.append({
                "product_id":pid, "title":f"Roamix {noun} {model} {detail}",
                "brand":"Roamix", "category":category, "origin_country":_ORIGINS[family%8],
                "description":f"{purpose}。本型号规格：{detail}。商品功能以本型号为准，其他档位配置不包含在内。",
                "highlights":[{"label":"规格","detail":detail},{"label":"材质","detail":material}],
                "ships_to":(["JP","SG"] if offset%11==0 else ["CN","US","EU","JP","SG"]),
                "skus":[{"sku_id":f"{pid}-S{i+1}","spec":color,"price_major":round((base_cny+i*9)/factor,2),"currency":currency,
                         "stock":0 if fully_out or (partially_out and i==0) else 15+(offset*13+i*7)%100}
                        for i,color in enumerate(("石墨黑","雾灰"))],
                "source_platform":platform,"external_product_id":f"demo-{platform}-{pid}",
                "canonical_product_id":f"CAN-{model}","material_tags":[material],
                "weight_kg":specification.weight_kg,
                "dimensions_cm":dict(zip(("length","width","height"),specification.dimensions_cm)), "package_dimensions_cm":{},
                "tax_category":category,"rating_summary":{"average":round(4+(offset%9)/10,1),"review_count":20+offset*3},
                "updated_at":"2026-09-18","data_provenance":"synthetic","model_spec":detail,
                "evaluation_tags":["hard_negative"] if tier==0 or fully_out else [],
                "evaluation_family":family,"evaluation_tier":tier,
            })
    return records


def build_records() -> list[dict]:
    legacy = [_legacy_record(product, index) for index, product in enumerate(build_demo_seed_products())]
    generated = [_generated_record(index) for index in range(440)]
    return legacy + generated + _expanded_records()


def main() -> None:
    records = build_records()
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text("\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records) + "\n", encoding="utf-8")
    print(f"已写入 {_OUT}：{len(records)} SPU，{sum(len(record['skus']) for record in records)} SKU")


if __name__ == "__main__":
    main()
