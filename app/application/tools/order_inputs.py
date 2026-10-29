"""下单工具与确认接口共用地址输入；不承载交易授权。"""
from pydantic import BaseModel, ConfigDict, Field, model_validator
from app.application.agents.shopping_state import Choice


class ShippingAddressInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    recipient_name: str = Field(min_length=1, max_length=100)
    country: str = Field(min_length=2, max_length=2, description="国家二位码，如 CN")
    state: str = Field(default="", max_length=100, description="省/州，未提供时省略，不填写说明文字")
    city: str = Field(min_length=1, max_length=100, description='城市名称；省/州单独填入state')
    address_line: str = Field(min_length=1, max_length=300, description='城市以下的详细地址：区县、街道、门牌、楼栋和房间；不重复country/state/city中已单独填写的部分，按买家提供的信息填写')
    postal_code: str = Field(default="", max_length=30, description="邮编，未提供时省略或传空字符串")
    phone: str = Field(default="", max_length=40,
                       description="可选联系电话；未提供或要求不填写时省略或传空字符串，不填说明文字")


class CreateOrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[Choice] = Field(min_length=1, description="本次明确选择的商品、SKU和正整数数量")
    shipping_address: ShippingAddressInput

    @model_validator(mode="after")
    def unique_skus(self):
        if len({item.sku_id for item in self.items}) != len(self.items):
            raise ValueError("items中的SKU不得重复")
        return self


class CancellationInput(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    reason: str = Field(min_length=1, max_length=500, description='买家提供的取消原因')


class HandleOrderInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    order_id: str = Field(min_length=1, max_length=32, description='当前买家的订单号')
    cancellation: CancellationInput | None = Field(default=None, description='需要准备取消时填写；省略或null只查看订单')
