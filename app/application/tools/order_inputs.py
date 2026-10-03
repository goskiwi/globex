"""下单工具与确认接口共用地址输入；不承载交易授权。"""
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShippingAddressInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    recipient_name: str = Field(min_length=1, max_length=100)
    country: str = Field(min_length=2, max_length=2, description="国家二位码，如 CN")
    state: str = Field(default="", max_length=100, description="省/州，未提供时省略，不填写说明文字")
    city: str = Field(min_length=1, max_length=100)
    address_line: str = Field(min_length=1, max_length=300)
    postal_code: str = Field(default="", max_length=30, description="邮编，未提供时省略或传空字符串")
    phone: str = Field(default="", max_length=40,
                       description="可选联系电话；未提供或要求不填写时省略或传空字符串，不填说明文字")


class CreateOrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sku_ids: list[str] = Field(min_length=1, description="已选 SKU，数量从服务端选购状态读取")
    shipping_address: ShippingAddressInput

    @model_validator(mode="after")
    def unique_skus(self):
        if any(not sku.strip() for sku in self.sku_ids) or len(set(self.sku_ids)) != len(self.sku_ids):
            raise ValueError("sku_ids 必须非空且不重复")
        return self
