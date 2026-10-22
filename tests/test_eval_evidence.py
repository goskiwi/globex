# -*- coding: utf-8 -*-
"""评测事件证据的确定性断言。"""
from tests.native_trace_helpers import tool_frames, final_frames

import pytest

from scripts.eval.evidence import AssertionDefinitionError, evaluate_trace_assertions


EVENTS = [
    {'type': 'TOOL_CALL_START', 'toolCallId': 'pending-test-call', 'toolCallName': 'product_search_tool'},
    *tool_frames('product_search_tool', {'hits': [{'product_id': 'P1008', 'price_major': 89, 'skus': [{'sku_id':'P1008-S1','stock':18}], 'material_tags': ['金属']}]}, status='success'),
    {'type': 'TOOL_CALL_START', 'toolCallId': 'pending-test-call', 'toolCallName': 'create_order_tool'},
    *tool_frames('create_order_tool', {'order': {'status': 'CONFIRMED'}}, status='success'),
]


def test_required_and_forbidden_tools_are_checked_from_trace():
    results = evaluate_trace_assertions(
        [
            {"criterion": "调商品检索", "kind": "required_tools", "tools": ["product_search_tool"]},
            {"criterion": "不下单", "kind": "forbidden_tools", "tools": ["handle_order"]},
        ],
        EVENTS,
    )

    assert [item["pass"] for item in results] == [True, True]


def test_product_constraint_leak_is_deterministically_captured():
    results = evaluate_trace_assertions(
        [
            {
                "criterion": "预算库存材质均不得泄漏",
                "kind": "product_hits", "price_max_major": 100, "require_in_stock": True,
                "excluded_material_tags": ["合成聚合物"],
            },
        ],
        EVENTS,
    )
    assert results[0]["pass"] is True

    bad_events=tool_frames('product_search_tool',{'hits':[{'product_id':'P-bad','price_major':101,
        'skus':[{'sku_id':'P-bad-S1','stock':0}],'material_tags':['合成聚合物']}]})
    bad = evaluate_trace_assertions(
        [{"criterion": "预算库存材质均不得泄漏", "kind": "product_hits", "price_max_major": 100,
          "require_in_stock": True, "excluded_material_tags": ["合成聚合物"]}],
        bad_events,
    )
    assert bad[0]["pass"] is False
    assert "P-bad" in bad[0]["reason"]


def test_product_constraint_reads_stock_from_real_product_card_skus():
    """线上商品卡没有顶层 stock，评测必须从 SKU 判断有货。"""
    events = [*tool_frames('product_search_tool', {'hits': [{'product_id': 'P-card', 'price_major': 99, 'material_tags': ['金属'], 'skus': [{'sku_id': 'P-card-S1', 'stock': 3}]}]}, status='success')]

    result = evaluate_trace_assertions(
        [{"criterion": "SKU 有货", "kind": "product_hits", "require_in_stock": True}],
        events,
    )

    assert result[0]["pass"] is True


def test_empty_but_successful_search_has_no_constraint_leak():
    events = [*tool_frames('product_search_tool', {'hits': []}, status='success')]

    result = evaluate_trace_assertions(
        [{"criterion": "不得泄漏禁用材质", "kind": "product_hits", "excluded_material_tags": ["合成聚合物"]}],
        events,
    )

    assert result[0]["pass"] is True
    assert "候选 0 个" in result[0]["reason"]


def test_order_status_and_unknown_assertion_fail_closed():
    results = evaluate_trace_assertions(
        [{"criterion": "订单已确认", "kind": "order_statuses", "expected": ["CONFIRMED"]}],
        EVENTS,
    )
    assert results[0]["pass"] is True

    with pytest.raises(AssertionDefinitionError):
        evaluate_trace_assertions([{"criterion": "坏断言", "kind": "unsupported"}], EVENTS)


def test_forbidden_tool_before_turn_checks_only_the_pre_confirmation_trace():
    assertion = [{
        "criterion": "确认前不下单", "kind": "forbidden_tools_before_turn",
        "tools": ["create_order_tool"], "before_turn": 1,
    }]
    safe_then_confirmed = [
        {'type': 'TOOL_CALL_START', 'toolCallId': 'pending-test-call', 'toolCallName': 'product_search_tool'},
        {"type": "eval.turn.complete", "payload": {"turn_index": 1}},
        {'type': 'TOOL_CALL_START', 'toolCallId': 'pending-test-call', 'toolCallName': 'create_order_tool'},
    ]
    premature_order = [
        {'type': 'TOOL_CALL_START', 'toolCallId': 'pending-test-call', 'toolCallName': 'create_order_tool'},
        {"type": "eval.turn.complete", "payload": {"turn_index": 1}},
    ]

    assert evaluate_trace_assertions(assertion, safe_then_confirmed)[0]["pass"] is True
    assert evaluate_trace_assertions(assertion, premature_order)[0]["pass"] is False


def test_tool_error_assertion_requires_the_expected_reproducible_failure():
    assertion = [{
        "criterion": "不支持目的国要如实报错", "kind": "tool_error",
        "tools": ["product_search_tool", "quote_products"], "contains": "不支持",
    }]
    events = [*tool_frames('product_search_tool', {'error': '暂不支持的目的国：BR'}, status='error')]

    assert evaluate_trace_assertions(assertion, events)[0]["pass"] is True
    quote = [*tool_frames('quote_products', {'error': '商品不支持配送至 BR：P2079'}, status='error')]
    assert evaluate_trace_assertions(assertion, quote)[0]["pass"] is True
    assert not evaluate_trace_assertions(assertion, [*tool_frames('quote_products', {'error': '网络超时'}, status='error')])[0]["pass"]
    assert not evaluate_trace_assertions(assertion, [*tool_frames('other', {'error': '不支持'}, status='error')])[0]["pass"]
    assert not evaluate_trace_assertions(assertion, [*final_frames(text='不支持配送至 BR')])[0]["pass"]


def test_landed_price_consistency_is_checked_from_tool_facts():
    assertion = [{"criterion": "到手价恒等式", "kind": "landed_price_consistency"}]
    good = [*tool_frames('product_search_tool', {'hits': [{'product_id': 'P1', 'landed_price': {'subtotal_minor': 21900, 'freight_minor': 915, 'tariff_minor': 0, 'total_amount_minor': 22815, 'currency': 'USD'}}]}, status='success')]
    bad=tool_frames('product_search_tool',{'hits':[{'product_id':'P1','landed_price':{
        'subtotal_minor':21900,'freight_minor':915,'tariff_minor':0,'total_amount_minor':23000,'currency':'USD'}}]})

    assert evaluate_trace_assertions(assertion, good)[0]["pass"] is True
    assert evaluate_trace_assertions(assertion, bad)[0]["pass"] is False


@pytest.mark.parametrize('mode',['inspect','prepare_cancel'])
def test_order_task_view_is_not_misreported_as_a_new_write(mode):
    data={'order':{'order_id':'existing','status':'CONFIRMED'},
          'operation':{'kind':mode,'status':'completed' if mode=='inspect' else 'awaiting_confirmation'},
          'confirmation':None}
    def check(payload):
        events=[{'type':'eval.confirmations.snapshot','payload':{'confirmations':[]}},*tool_frames('handle_order',payload)]
        return evaluate_trace_assertions([{'criterion':'先确认后写入','kind':'no_orders_before_http_confirmation'}],events)[0]['pass']
    assert check(data)
    assert not check({**data,'confirmation':{'result':{'order_id':'unexpected','status':'CONFIRMED'}}})
    assert not check({'order':{'order_id':'unexpected','status':'CONFIRMED'}})
